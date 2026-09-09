"""Conservative, video-time unattended-bag state machine for a fixed camera.

Person proximity is an attendance heuristic, never an ownership assertion.
Missing detections pause timers; long gaps reset them. No alert is raised for
an invisible bag. Tracker ID changes create a new observation history.
"""

import math
from dataclasses import dataclass

BAG_NAMES = frozenset({"backpack", "handbag", "suitcase", "bag", "luggage", "travel bag"})


@dataclass(frozen=True)
class MonitorConfig:
    alert_seconds: float = 300.0
    stationary_seconds: float = 3.0
    movement_fraction: float = 0.015
    proximity_ratio: float = 0.65
    max_missing_seconds: float = 2.0
    roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    def __post_init__(self):
        values = (
            self.alert_seconds,
            self.stationary_seconds,
            self.movement_fraction,
            self.proximity_ratio,
            self.max_missing_seconds,
        )
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Monitor thresholds must be finite and positive.")
        x1, y1, x2, y2 = self.roi
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("ROI must be normalized x1,y1,x2,y2 bounds within [0, 1].")


@dataclass
class BagState:
    track_id: int
    class_name: str
    anchor: tuple[float, float]
    last_seen: float
    stationary_elapsed: float = 0.0
    unattended_elapsed: float = 0.0
    status: str = "moving"
    visible: bool = True
    alert_id: int | None = None


class BagMonitor:
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.bags: dict[int, BagState] = {}
        self.events: list[dict] = []
        self.last_time: float | None = None
        self.next_alert = 1

    def _resolve(self, state: BagState, timestamp: float, reason: str):
        if state.alert_id is not None:
            self.events.append(
                {
                    "event": "resolved",
                    "alert_id": state.alert_id,
                    "track_id": state.track_id,
                    "timestamp_seconds": timestamp,
                    "reason": reason,
                }
            )
            state.alert_id = None

    def update(
        self, timestamp: float, detections: list[dict], width: int, height: int
    ) -> list[dict]:
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Timestamp must be finite and nonnegative.")
        if self.last_time is not None and timestamp <= self.last_time:
            raise ValueError("Video timestamps must increase strictly.")
        self.last_time = timestamp
        people = [d for d in detections if d["class_name"] == "person"]
        visible = set()
        rows = []
        for detection in detections:
            track_id = detection["track_id"]
            if detection["class_name"] not in BAG_NAMES or track_id is None:
                continue
            x1, y1, x2, y2 = detection["bbox"]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            rx1, ry1, rx2, ry2 = self.config.roi
            if not (rx1 <= center[0] / width <= rx2 and ry1 <= center[1] / height <= ry2):
                continue
            visible.add(track_id)
            state = self.bags.get(track_id)
            if state is None:
                state = BagState(track_id, detection["class_name"], center, timestamp)
                self.bags[track_id] = state
            gap = timestamp - state.last_seen
            if gap > self.config.max_missing_seconds:
                self._resolve(state, timestamp, "visibility gap")
                state.anchor = center
                state.stationary_elapsed = state.unattended_elapsed = 0.0
            dt = gap if state.visible and gap <= self.config.max_missing_seconds else 0.0
            moved = math.dist(center, state.anchor) > (
                self.config.movement_fraction * math.hypot(width, height)
            )
            nearby = False
            for person in people:
                px1, py1, px2, py2 = person["bbox"]
                # Shortest distance from bag center to the person's bounding rectangle.
                dx = max(px1 - center[0], 0, center[0] - px2)
                dy = max(py1 - center[1], 0, center[1] - py2)
                if math.hypot(dx, dy) <= self.config.proximity_ratio * max(py2 - py1, 1):
                    nearby = True
                    break
            if moved:
                state.anchor = center
                state.stationary_elapsed = state.unattended_elapsed = 0.0
                state.status = "moving"
                self._resolve(state, timestamp, "bag moved")
            else:
                previous_stationary = state.stationary_elapsed
                state.stationary_elapsed += dt
                if nearby:
                    state.unattended_elapsed = 0.0
                    state.status = "person nearby"
                    self._resolve(state, timestamp, "person nearby")
                elif state.stationary_elapsed >= self.config.stationary_seconds:
                    unattended_dt = 0.0 if state.status == "person nearby" else dt
                    # Count only the observed part of this interval after stationarity is confirmed.
                    state.unattended_elapsed += min(
                        unattended_dt,
                        max(
                            0.0,
                            state.stationary_elapsed
                            - max(previous_stationary, self.config.stationary_seconds),
                        ),
                    )
                    state.status = "unattended"
                    if state.unattended_elapsed >= self.config.alert_seconds:
                        state.status = "alert"
                        if state.alert_id is None:
                            state.alert_id = self.next_alert
                            self.next_alert += 1
                            self.events.append(
                                {
                                    "event": "alert",
                                    "alert_id": state.alert_id,
                                    "track_id": track_id,
                                    "class_name": state.class_name,
                                    "timestamp_seconds": timestamp,
                                    "unattended_seconds": state.unattended_elapsed,
                                    "reason": "stationary bag with no detected person nearby",
                                }
                            )
                else:
                    state.status = "confirming stationary"
            state.last_seen = timestamp
            state.visible = True
            rows.append(
                {
                    "timestamp_seconds": timestamp,
                    "track_id": track_id,
                    "class_name": state.class_name,
                    "status": state.status,
                    "unattended_seconds": round(state.unattended_elapsed, 3),
                    "person_nearby": nearby,
                    "alert_id": state.alert_id,
                }
            )
        for track_id in list(self.bags):
            state = self.bags[track_id]
            if track_id not in visible:
                state.visible = False
                if timestamp - state.last_seen > self.config.max_missing_seconds:
                    self._resolve(state, timestamp, "bag no longer visible")
                    del self.bags[track_id]
        return rows
