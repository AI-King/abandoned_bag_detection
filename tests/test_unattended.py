"""Behavioral tests for the video-time alert policy, independent of YOLO accuracy."""

import pytest

from vision_lab.unattended import BagMonitor, MonitorConfig


def bag(track_id=7, x=400, name="suitcase"):
    return {"track_id": track_id, "class_name": name, "bbox": [x, 300, x + 40, 350]}


def person(near=True):
    x = 350 if near else 10
    return {"track_id": 1, "class_name": "person", "bbox": [x, 200, x + 50, 350]}


def tick(monitor, t, detections):
    return monitor.update(t, detections, 1000, 600)


def test_five_minutes_of_observed_video_time_and_no_duplicate_alerts():
    monitor = BagMonitor(MonitorConfig())
    for t in range(303):
        tick(monitor, t, [bag()])
    assert monitor.events == []
    rows = tick(monitor, 303, [bag()])
    assert rows[0]["status"] == "alert"
    assert rows[0]["unattended_seconds"] == 300
    for t in range(304, 320):
        tick(monitor, t, [bag()])
    assert len(monitor.events) == 1


def test_nearby_person_prevents_alert_and_return_resets_timer():
    monitor = BagMonitor(MonitorConfig(alert_seconds=3, stationary_seconds=1))
    for t in range(10):
        tick(monitor, t, [bag(), person()])
    assert monitor.events == []
    for t in range(10, 14):
        tick(monitor, t, [bag(), person(False)])
    assert monitor.events[0]["event"] == "alert"
    rows = tick(monitor, 14, [bag(), person()])
    assert rows[0]["unattended_seconds"] == 0
    assert monitor.events[-1]["reason"] == "person nearby"


def test_motion_resets_timer_even_when_slow_drift_accumulates():
    monitor = BagMonitor(MonitorConfig(alert_seconds=10, stationary_seconds=1))
    for t in range(20):
        tick(monitor, t, [bag(x=400 + t * 5)])
    assert monitor.events == []


def test_missing_frames_do_not_accrue_time_or_raise_alert():
    monitor = BagMonitor(MonitorConfig(alert_seconds=3, stationary_seconds=1))
    for t in range(3):
        tick(monitor, t, [bag()])
    before = monitor.bags[7].unattended_elapsed
    tick(monitor, 3, [])
    rows = tick(monitor, 4, [bag()])
    assert rows[0]["unattended_seconds"] == before
    assert monitor.events == []
    tick(monitor, 8, [bag()])
    assert monitor.bags[7].unattended_elapsed == 0


def test_disappearance_resolves_as_unknown_not_collected():
    monitor = BagMonitor(MonitorConfig(alert_seconds=1, stationary_seconds=1))
    for t in range(3):
        tick(monitor, t, [bag()])
    tick(monitor, 5, [])
    assert monitor.events[-1]["reason"] == "bag no longer visible"
    assert not monitor.bags


def test_new_id_does_not_inherit_old_timer():
    monitor = BagMonitor(MonitorConfig(alert_seconds=3, stationary_seconds=1))
    for t in range(3):
        tick(monitor, t, [bag()])
    rows = tick(monitor, 3, [bag(track_id=8)])
    assert rows[0]["unattended_seconds"] == 0
    assert monitor.events == []


@pytest.mark.parametrize("name", ["suitcase", "handbag", "backpack", "bag", "travel bag"])
def test_supported_luggage_classes(name):
    monitor = BagMonitor(MonitorConfig())
    assert tick(monitor, 0, [bag(name=name)])


def test_untracked_objects_and_outside_roi_are_not_monitored():
    monitor = BagMonitor(MonitorConfig(roi=(0, 0, 0.3, 1)))
    assert not tick(monitor, 0, [bag(), bag(track_id=None, x=100)])


def test_multiple_bags_have_independent_clocks():
    monitor = BagMonitor(MonitorConfig(alert_seconds=3, stationary_seconds=1))
    for t in range(5):
        tick(monitor, t, [bag(), bag(track_id=8, x=700 + 30 * t)])
    assert [e["track_id"] for e in monitor.events] == [7]


@pytest.mark.parametrize("timestamp", [0, -1, float("nan"), float("inf")])
def test_invalid_or_nonmonotonic_time_rejected(timestamp):
    monitor = BagMonitor(MonitorConfig())
    tick(monitor, 0, [])
    with pytest.raises(ValueError):
        tick(monitor, timestamp, [])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alert_seconds": 0},
        {"proximity_ratio": -1},
        {"stationary_seconds": float("nan")},
        {"roi": (1, 0, 0, 1)},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        MonitorConfig(**kwargs)
