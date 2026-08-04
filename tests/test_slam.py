"""Real LaserScan SLAM node, matching, mapping, and optimization tests."""
from __future__ import annotations

import math

import numpy as np

import blacknode  # noqa: F401
from blacknode.node import _NODE_REGISTRY
from blacknode.pkg.blacknode_cuda import slam_runtime, warp_points


def _points() -> np.ndarray:
    horizontal = [[x, 0.0, 0.0] for x in np.linspace(-2.0, 2.0, 25)]
    vertical = [[1.25, y, 0.0] for y in np.linspace(-1.0, 2.0, 19)]
    diagonal = [[-1.5 + value, 1.0 + value * 0.4, 0.0] for value in np.linspace(0.0, 1.3, 13)]
    return np.asarray(horizontal + vertical + diagonal, dtype=np.float32)


def _source() -> dict:
    return {
        "kind": "blacknode.message-stream",
        "schema_version": 1,
        "stream_id": "topic-subscriber:/scan",
        "protocol": "ros2",
        "topic": "/scan",
        "message_type": "sensor_msgs/msg/LaserScan",
    }


def _message(second: int) -> dict:
    return {
        "message": {
            "header": {"stamp": {"sec": second, "nanosec": 0}, "frame_id": "laser"},
            "angle_min": -math.pi,
            "angle_max": math.pi,
            "angle_increment": math.pi / 8.0,
            "range_min": 0.1,
            "range_max": 12.0,
            "ranges": [1.0] * 16,
            "intensities": [],
        }
    }


def test_correlative_match_recovers_metric_pose_near_prior():
    local = _points()
    expected = np.asarray([0.35, -0.20, math.radians(6.0)])
    reference = slam_runtime._transform_points(local, expected)

    pose, score = slam_runtime.correlative_match(
        local,
        reference,
        np.asarray([0.22, -0.10, math.radians(2.0)]),
        resolution=0.05,
        linear_window=0.4,
        angular_window=math.radians(10.0),
    )

    assert np.linalg.norm(pose[:2] - expected[:2]) < 0.09
    assert abs(slam_runtime._wrap_angle(float(pose[2] - expected[2]))) < math.radians(3.0)
    assert score > 0.8


def test_pose_graph_loop_constraint_corrects_drift():
    poses = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.25, 0.15, 0.04],
    ])
    edges = [
        {"i": 0, "j": 1, "measurement": np.asarray([1.0, 0.0, 0.0]), "weight": [10.0, 10.0, 20.0]},
        {"i": 1, "j": 2, "measurement": np.asarray([1.0, 0.0, 0.0]), "weight": [10.0, 10.0, 20.0]},
        {"i": 0, "j": 2, "measurement": np.asarray([2.0, 0.0, 0.0]), "weight": [100.0, 100.0, 150.0]},
    ]

    optimized = slam_runtime.optimize_pose_graph(poses, edges)

    assert np.linalg.norm(optimized[2, :2] - np.asarray([2.0, 0.0])) < 0.03
    assert abs(float(optimized[2, 2])) < 0.02


def test_slam_node_maps_live_scans_and_pause_keeps_localization_active(monkeypatch):
    slam_runtime.stop_slam()
    state = {"messages": [_message(1)]}
    local = _points()

    def reader(_source_value):
        return {
            "message": state["messages"][-1],
            "messages": list(state["messages"]),
            "received": len(state["messages"]),
            "status": {"state": "ready", "source_fresh": True, "last_message_time_ns": state["messages"][-1]["message"]["header"]["stamp"]["sec"] * 1_000_000_000},
        }

    monkeypatch.setattr(
        warp_points,
        "process_laser_scan",
        lambda *_args, **_kwargs: {
            "ok": True,
            "filtered_points": local.astype(float).tolist(),
            "kernel_ms": 0.2,
        },
    )

    started = _NODE_REGISTRY["SLAM"]({
        "action": "start",
        "source": _source(),
        "slam_id": "live-map",
        "mode": "editor",
        "device": "cpu",
        "keyframe_interval_s": 0.1,
        "__node_id__": "slam-node",
        "__message_stream_reader__": reader,
    })
    state["messages"].append(_message(2))
    mapped = slam_runtime.slam_status("live-map")
    paused = _NODE_REGISTRY["SLAM"]({"action": "pause", "slam_id": "live-map"})
    state["messages"].append(_message(3))
    localized = slam_runtime.slam_status("live-map")
    runtime = slam_runtime.runtime_status()
    cleared = _NODE_REGISTRY["SLAM"]({"action": "clear", "slam_id": "live-map"})

    assert started["running"] is True
    assert mapped["scene"]["slam"]["keyframes"] == 2
    assert mapped["map"]["point_count"] > 0
    assert mapped["pose"]["kind"] == "blacknode.slam-pose"
    assert paused["scene"]["history_paused"] is True
    assert localized["live"] is True
    assert localized["scene"]["slam"]["keyframes"] == 2
    assert localized["status"]["mapping"] is False
    assert runtime["node_outputs"][0]["node_id"] == "slam-node"
    assert cleared["map"]["point_count"] == 0
    assert cleared["map"]["keyframes"] == 0
    slam_runtime.stop_slam()


def test_slam_node_declares_generic_stream_contract():
    fn = _NODE_REGISTRY["SLAM"]

    assert fn._bn_package == "blacknode-cuda"
    assert fn._bn_input_types["source"] == "Dict"
    assert fn._bn_input_types["odometry"] == "Dict"
    assert fn._bn_output_types["scene"] == "Dict"
    assert fn._bn_output_types["pose"] == "Dict"
    assert fn._bn_output_types["map"] == "Dict"
    assert fn._bn_live_capable is True


def test_slam_displays_sparse_live_scan_before_it_can_localize(monkeypatch):
    slam_runtime.stop_slam()
    sparse = _points()[:3]
    monkeypatch.setattr(
        warp_points,
        "process_laser_scan",
        lambda *_args, **_kwargs: {
            "ok": True,
            "filtered_points": sparse.astype(float).tolist(),
            "kernel_ms": 0.1,
        },
    )

    result = _NODE_REGISTRY["SLAM"]({
        "action": "start",
        "source": _source(),
        "slam_id": "sparse-live",
        "mode": "editor",
        "device": "cpu",
        "__message_stream_reader__": lambda _source_value: {
            **_message(0),
            "messages": [_message(0)["message"]],
            "received": 1,
            "status": {
                "state": "ready",
                "source_fresh": True,
                "received": 1,
                "last_message_time_ns": 1,
            },
        },
    })

    assert result["live"] is True
    assert result["scene"]["current_point_count"] == 3
    assert result["scene"]["point_count"] == 0
    assert result["status"]["state"] == "waiting"
    assert "at least 8 valid returns" in result["report"]
    slam_runtime.stop_slam()


def test_slam_uses_stream_counter_when_sensor_timestamp_moves_backwards(monkeypatch):
    slam_runtime.stop_slam()
    state = {"received": 1, "second": 10}
    processed = []

    def process(*_args, **_kwargs):
        processed.append(state["received"])
        return {
            "ok": True,
            "filtered_points": _points().astype(float).tolist(),
            "kernel_ms": 0.1,
        }

    def reader(_source_value):
        message = _message(state["second"])["message"]
        return {
            "message": message,
            "messages": [message],
            "received": state["received"],
            "status": {
                "state": "ready",
                "source_fresh": True,
                "received": state["received"],
                "last_message_time_ns": state["received"],
            },
        }

    monkeypatch.setattr(warp_points, "process_laser_scan", process)
    _NODE_REGISTRY["SLAM"]({
        "action": "start",
        "source": _source(),
        "slam_id": "counter-sequenced",
        "mode": "editor",
        "device": "cpu",
        "__message_stream_reader__": reader,
    })
    state.update(received=2, second=1)
    result = slam_runtime.slam_status("counter-sequenced")

    assert processed == [1, 2]
    assert result["live"] is True
    assert result["status"]["received"] == 2
    slam_runtime.stop_slam()
