"""Frame-to-frame Integer Linear Programming (ILP) Multi-Object Tracker.

Uses binary integer optimization to globally associate tracks and detections across
frames. Formulated using network flow constraints and solved via scipy.optimize.milp
with the HiGHS solver.

Specialized for unattended bag detection:
- Invariant to bag subcategory flips (backpack <-> handbag <-> suitcase).
- Stationary anchor locking prevents track drift from passing pedestrians.
- Tracks owner-object co-movement and explicit owner-separation events.
- Graceful coasting over temporary occlusions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import LinearConstraint, milp

BAG_NAMES = frozenset({"backpack", "handbag", "suitcase", "bag", "luggage", "travel bag"})


def box_iou(box1: list[float] | tuple[float, ...], box2: list[float] | tuple[float, ...]) -> float:
    """Compute Intersection over Union between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


@dataclass
class TrackState:
    track_id: int
    class_name: str
    bbox: list[float]  # [x1, y1, x2, y2]
    confidence: float
    center: tuple[float, float]
    velocity: tuple[float, float] = (0.0, 0.0)
    last_timestamp: float = 0.0
    frames_unseen: int = 0
    total_frames: int = 1
    stationary: bool = False
    stationary_anchor: tuple[float, float] | None = None
    stationary_frames: int = 0
    owner_id: int | None = None
    separated_from_owner: bool = False
    history: list[tuple[float, tuple[float, float]]] = field(default_factory=list)


@dataclass
class ILPTrackerConfig:
    max_lost_frames: int = 30  # Allow coasting over missing detections
    max_stationary_lost_frames: int = 90  # Stationary luggage held longer
    cost_miss: float = 0.70  # Cost penalty for unassigned track
    cost_birth: float = 0.85  # Cost penalty for new track creation
    max_distance_ratio: float = 0.25  # Max movement as fraction of frame diagonal
    iou_weight: float = 0.40  # Weight for (1 - IoU) in cost
    dist_weight: float = 0.60  # Weight for normalized Euclidean distance in cost
    stationary_velocity_threshold: float = 0.005  # Movement fraction per frame to be stationary
    stationary_anchor_bonus: float = 0.35  # Cost discount for matching stationary anchor


class ILPTracker:
    """Frame-to-frame ILP Multi-Object Tracker."""

    def __init__(self, config: ILPTrackerConfig | None = None):
        self.config = config or ILPTrackerConfig()
        self.tracks: dict[int, TrackState] = {}
        self.next_id: int = 1

    def update(
        self, detections: list[dict], width: int, height: int, timestamp: float
    ) -> list[dict]:
        """Associate detections with tracks using ILP and update track states.

        detections: list of dicts with keys:
            'class_id', 'class_name', 'confidence', 'bbox' ([x1, y1, x2, y2])
        Returns updated detections with 'track_id' filled in.
        """
        frame_diag = math.hypot(width, height) or 1.0

        active_track_ids = list(self.tracks.keys())
        num_tracks = len(active_track_ids)
        num_dets = len(detections)

        # Precompute centers for detections
        det_centers = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            det_centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

        if num_tracks == 0 and num_dets == 0:
            return detections

        if num_tracks == 0:
            # All detections initiate new tracks
            for i, det in enumerate(detections):
                tid = self._create_track(det, det_centers[i], timestamp)
                det["track_id"] = tid
            self._update_owner_associations(width, height)
            return detections

        if num_dets == 0:
            # All tracks are unassigned / coasted
            self._handle_unassigned_tracks(active_track_ids)
            return detections

        # Build ILP variables:
        # x_{i, j}: Track i matched to Detection j (size: num_tracks * num_dets)
        # u_i: Track i unassigned (size: num_tracks)
        # v_j: Detection j new track (size: num_dets)
        total_vars = (num_tracks * num_dets) + num_tracks + num_dets
        c = np.zeros(total_vars, dtype=np.float64)

        # Upper bounds for variables: default 1.0 (binary)
        ub = np.ones(total_vars, dtype=np.float64)
        lb = np.zeros(total_vars, dtype=np.float64)

        def var_x(i: int, j: int) -> int:
            return i * num_dets + j

        def var_u(i: int) -> int:
            return (num_tracks * num_dets) + i

        def var_v(j: int) -> int:
            return (num_tracks * num_dets) + num_tracks + j

        # Populate costs for x_{i, j}
        for i, tid in enumerate(active_track_ids):
            track = self.tracks[tid]
            dt = max(timestamp - track.last_timestamp, 1e-4) if track.last_timestamp > 0 else 1.0
            pred_cx = track.center[0] + track.velocity[0] * dt
            pred_cy = track.center[1] + track.velocity[1] * dt

            for j, det in enumerate(detections):
                idx = var_x(i, j)
                det_center = det_centers[j]

                # Disallow cross-category matching between Person and Bag
                track_is_bag = track.class_name in BAG_NAMES
                det_is_bag = det["class_name"] in BAG_NAMES

                if track_is_bag != det_is_bag:
                    # Incompatible class: cannot match
                    ub[idx] = 0.0
                    c[idx] = 100.0
                    continue

                # Distance calculation
                dist = math.dist((pred_cx, pred_cy), det_center)
                norm_dist = dist / frame_diag

                # Gating: if distance exceeds threshold and not a stationary anchor match
                max_dist = self.config.max_distance_ratio
                if track.stationary and track.stationary_anchor:
                    anchor_dist = math.dist(track.stationary_anchor, det_center) / frame_diag
                    if anchor_dist > max_dist:
                        ub[idx] = 0.0
                        c[idx] = 100.0
                        continue
                elif norm_dist > max_dist:
                    ub[idx] = 0.0
                    c[idx] = 100.0
                    continue

                # Compute IoU and composite cost
                iou = box_iou(track.bbox, det["bbox"])
                cost = self.config.dist_weight * min(
                    norm_dist / max_dist, 1.0
                ) + self.config.iou_weight * (1.0 - iou)

                # Stationary anchor bonus: lock track onto stationary position
                if track.stationary and track.stationary_anchor:
                    anchor_dist = math.dist(track.stationary_anchor, det_center) / frame_diag
                    if anchor_dist < 0.05:
                        cost = max(0.01, cost - self.config.stationary_anchor_bonus)

                # Small class consistency bonus if exact class match
                if track.class_name == det["class_name"]:
                    cost = max(0.01, cost * 0.9)

                c[idx] = cost

        # Costs for unassigned track u_i and new track v_j
        for i in range(num_tracks):
            c[var_u(i)] = self.config.cost_miss

        for j in range(num_dets):
            c[var_v(j)] = self.config.cost_birth

        # Constraints:
        # 1. For each track i: sum_j x_{i, j} + u_i = 1  (num_tracks constraints)
        # 2. For each detection j: sum_i x_{i, j} + v_j = 1  (num_dets constraints)
        num_constraints = num_tracks + num_dets
        A = np.zeros((num_constraints, total_vars), dtype=np.float64)
        rhs = np.ones(num_constraints, dtype=np.float64)

        for i in range(num_tracks):
            for j in range(num_dets):
                A[i, var_x(i, j)] = 1.0
            A[i, var_u(i)] = 1.0

        for j in range(num_dets):
            row = num_tracks + j
            for i in range(num_tracks):
                A[row, var_x(i, j)] = 1.0
            A[row, var_v(j)] = 1.0

        # Integrality: all variables must be binary (1)
        integrality = np.ones(total_vars, dtype=np.int32)
        constraints = LinearConstraint(A, lb=rhs, ub=rhs)
        bounds = (lb, ub)

        res = milp(c=c, integrality=integrality, constraints=constraints, bounds=bounds)

        if not res.success:
            # Fallback if solver fails: greedy assignment
            matched_pairs, unmatched_tracks, unmatched_dets = self._greedy_fallback(
                active_track_ids, detections, det_centers, frame_diag
            )
        else:
            sol = res.x
            matched_pairs = []
            unmatched_tracks = []
            unmatched_dets = []

            for i, tid in enumerate(active_track_ids):
                matched = False
                for j in range(num_dets):
                    if sol[var_x(i, j)] > 0.5:
                        matched_pairs.append((tid, j))
                        matched = True
                        break
                if not matched:
                    unmatched_tracks.append(tid)

            for j in range(num_dets):
                if sol[var_v(j)] > 0.5:
                    unmatched_dets.append(j)

        # Apply matches
        for tid, j in matched_pairs:
            det = detections[j]
            det["track_id"] = tid
            self._update_track(tid, det, det_centers[j], timestamp, frame_diag)

        # Handle unmatched tracks (coasting / expiration)
        self._handle_unassigned_tracks(unmatched_tracks)

        # Handle unmatched detections (new tracks)
        for j in unmatched_dets:
            det = detections[j]
            tid = self._create_track(det, det_centers[j], timestamp)
            det["track_id"] = tid

        # Update owner-bag associations
        self._update_owner_associations(width, height)

        return detections

    def _create_track(self, det: dict, center: tuple[float, float], timestamp: float) -> int:
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = TrackState(
            track_id=tid,
            class_name=det["class_name"],
            bbox=list(det["bbox"]),
            confidence=det["confidence"],
            center=center,
            velocity=(0.0, 0.0),
            last_timestamp=timestamp,
            frames_unseen=0,
            total_frames=1,
            history=[(timestamp, center)],
        )
        return tid

    def _update_track(
        self,
        tid: int,
        det: dict,
        center: tuple[float, float],
        timestamp: float,
        frame_diag: float,
    ):
        track = self.tracks[tid]
        dt = max(timestamp - track.last_timestamp, 1e-4)
        vx = (center[0] - track.center[0]) / dt
        vy = (center[1] - track.center[1]) / dt

        # Exponential moving average for velocity
        alpha = 0.6
        track.velocity = (
            alpha * vx + (1 - alpha) * track.velocity[0],
            alpha * vy + (1 - alpha) * track.velocity[1],
        )

        # Check stationarity
        speed = math.hypot(track.velocity[0], track.velocity[1]) / frame_diag
        if speed < self.config.stationary_velocity_threshold:
            track.stationary_frames += 1
            if track.stationary_frames >= 10:
                track.stationary = True
                if track.stationary_anchor is None:
                    track.stationary_anchor = center
        else:
            # Moved significantly
            if track.stationary and track.stationary_anchor:
                if math.dist(track.stationary_anchor, center) / frame_diag > 0.03:
                    track.stationary = False
                    track.stationary_anchor = None
                    track.stationary_frames = 0
            else:
                track.stationary_frames = max(0, track.stationary_frames - 2)

        # Bag class smoothing: if previous was suitcase and new is handbag, keep canonical
        if track.class_name in BAG_NAMES and det["class_name"] in BAG_NAMES:
            pass
        else:
            track.class_name = det["class_name"]

        track.bbox = list(det["bbox"])
        track.confidence = det["confidence"]
        track.center = center
        track.last_timestamp = timestamp
        track.frames_unseen = 0
        track.total_frames += 1
        track.history.append((timestamp, center))
        if len(track.history) > 60:
            track.history.pop(0)

    def _handle_unassigned_tracks(self, track_ids: list[int]):
        for tid in track_ids:
            track = self.tracks.get(tid)
            if not track:
                continue
            track.frames_unseen += 1
            # Stationary bags are granted extended lifetime to tolerate long occlusions
            max_unseen = (
                self.config.max_stationary_lost_frames
                if track.stationary
                else self.config.max_lost_frames
            )
            if track.frames_unseen > max_unseen:
                del self.tracks[tid]

    def _update_owner_associations(self, width: int, height: int):
        """Associate moving bags with nearby person tracks, and detect separation."""
        people = [
            t for t in self.tracks.values() if t.class_name == "person" and t.frames_unseen == 0
        ]
        bags = [t for t in self.tracks.values() if t.class_name in BAG_NAMES]

        for bag in bags:
            if not people:
                if bag.owner_id is not None:
                    bag.separated_from_owner = True
                continue

            # If bag is moving and has no owner, find closest person
            if not bag.stationary:
                best_person = None
                best_dist = float("inf")
                for person in people:
                    d = math.dist(bag.center, person.center)
                    p_height = max(person.bbox[3] - person.bbox[1], 20.0)
                    if d <= 1.2 * p_height and d < best_dist:
                        best_dist = d
                        best_person = person
                if best_person is not None:
                    bag.owner_id = best_person.track_id
                    bag.separated_from_owner = False
            elif bag.owner_id is not None:
                # Bag is stationary: check if owner is moving away
                owner = self.tracks.get(bag.owner_id)
                if owner is None or owner.frames_unseen > 15:
                    bag.separated_from_owner = True
                else:
                    d = math.dist(bag.center, owner.center)
                    o_height = max(owner.bbox[3] - owner.bbox[1], 20.0)
                    if d > 1.5 * o_height:
                        bag.separated_from_owner = True

    def _greedy_fallback(
        self,
        active_track_ids: list[int],
        detections: list[dict],
        det_centers: list[tuple[float, float]],
        frame_diag: float,
    ):
        matched = []
        used_tracks = set()
        used_dets = set()
        for _i, tid in enumerate(active_track_ids):
            track = self.tracks[tid]
            best_j = None
            best_cost = float("inf")
            for j, det in enumerate(detections):
                if j in used_dets:
                    continue
                if (track.class_name in BAG_NAMES) != (det["class_name"] in BAG_NAMES):
                    continue
                dist = math.dist(track.center, det_centers[j]) / frame_diag
                if dist < 0.25 and dist < best_cost:
                    best_cost = dist
                    best_j = j
            if best_j is not None:
                matched.append((tid, best_j))
                used_tracks.add(tid)
                used_dets.add(best_j)

        unmatched_tracks = [t for t in active_track_ids if t not in used_tracks]
        unmatched_dets = [j for j in range(len(detections)) if j not in used_dets]
        return matched, unmatched_tracks, unmatched_dets
