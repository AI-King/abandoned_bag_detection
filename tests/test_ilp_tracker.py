import pytest

from vision_lab.ilp_tracker import ILPTracker, ILPTrackerConfig, box_iou


def test_box_iou():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [10, 10, 20, 20]) == 0.0
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_ilp_tracker_continuous_motion():
    tracker = ILPTracker()
    # Frame 1: person and suitcase moving right
    dets1 = [
        {"class_id": 0, "class_name": "person", "confidence": 0.9, "bbox": [100, 100, 150, 250]},
        {
            "class_id": 28,
            "class_name": "suitcase",
            "confidence": 0.85,
            "bbox": [140, 200, 180, 240],
        },
    ]
    tracked1 = tracker.update(dets1, 640, 480, 0.0)
    assert tracked1[0]["track_id"] == 1
    assert tracked1[1]["track_id"] == 2

    # Frame 2: both moved right by 10 pixels
    dets2 = [
        {"class_id": 0, "class_name": "person", "confidence": 0.9, "bbox": [110, 100, 160, 250]},
        {
            "class_id": 28,
            "class_name": "suitcase",
            "confidence": 0.85,
            "bbox": [150, 200, 190, 240],
        },
    ]
    tracked2 = tracker.update(dets2, 640, 480, 0.033)
    # IDs must persist across frames
    assert tracked2[0]["track_id"] == 1
    assert tracked2[1]["track_id"] == 2


def test_ilp_tracker_bag_class_flip():
    """Verify that if detector flips handbag/suitcase, ILP tracker preserves track ID."""
    tracker = ILPTracker()
    dets1 = [
        {
            "class_id": 28,
            "class_name": "suitcase",
            "confidence": 0.85,
            "bbox": [200, 200, 260, 260],
        },
    ]
    tracked1 = tracker.update(dets1, 640, 480, 0.0)
    tid = tracked1[0]["track_id"]

    # Next frame, detector calls it 'handbag' at nearly the same spot
    dets2 = [
        {"class_id": 26, "class_name": "handbag", "confidence": 0.75, "bbox": [202, 201, 259, 261]},
    ]
    tracked2 = tracker.update(dets2, 640, 480, 0.033)
    assert tracked2[0]["track_id"] == tid, "Track ID should not change when bag subcategory flips"


def test_ilp_tracker_coasting_and_recovery():
    """Verify that a track coasts through missing frames and recovers its ID."""
    tracker = ILPTracker(ILPTrackerConfig(max_lost_frames=5))
    dets1 = [
        {"class_id": 28, "class_name": "suitcase", "confidence": 0.9, "bbox": [300, 300, 350, 350]},
    ]
    tracked1 = tracker.update(dets1, 640, 480, 0.0)
    tid = tracked1[0]["track_id"]

    # 3 frames with no detection (occlusion)
    for t in [0.033, 0.066, 0.099]:
        tracked = tracker.update([], 640, 480, t)
        assert len(tracked) == 0

    # Bag re-appears
    dets2 = [
        {"class_id": 28, "class_name": "suitcase", "confidence": 0.9, "bbox": [301, 302, 351, 352]},
    ]
    tracked2 = tracker.update(dets2, 640, 480, 0.132)
    assert tracked2[0]["track_id"] == tid, "Recovered detection should retain its original track ID"


def test_ilp_tracker_no_cross_class_matching():
    """Verify that a person detection never claims a bag track."""
    tracker = ILPTracker()
    dets1 = [
        {"class_id": 28, "class_name": "suitcase", "confidence": 0.9, "bbox": [300, 300, 350, 350]},
    ]
    tracker.update(dets1, 640, 480, 0.0)

    # Next frame, only a person appears near that area
    dets2 = [
        {"class_id": 0, "class_name": "person", "confidence": 0.9, "bbox": [300, 200, 350, 380]},
    ]
    tracked2 = tracker.update(dets2, 640, 480, 0.033)
    assert tracked2[0]["track_id"] != 1, "Person must not take over the suitcase's track ID"
