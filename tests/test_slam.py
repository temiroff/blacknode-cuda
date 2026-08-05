"""Real LaserScan SLAM node, matching, mapping, and optimization tests."""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

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


def test_particle_cpu_reference_scores_equal_work_pose_candidates():
    local = _points()
    reference = slam_runtime._transform_points(
        local,
        np.asarray([0.25, -0.1, math.radians(4.0)]),
    )
    candidates = np.asarray([
        [0.25, -0.1, math.radians(4.0)],
        [0.8, 0.6, math.radians(30.0)],
    ], dtype=np.float32)

    scores = slam_runtime.score_particle_candidates_cpu(
        local,
        reference,
        candidates,
        0.05,
    )

    assert scores.shape == (2,)
    assert scores[0] > 0.9
    assert scores[0] > scores[1]


def test_deskew_registers_beams_into_first_beam_frame():
    points = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

    deskewed = slam_runtime._deskew_points(
        points,
        [0, 3],
        4,
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, math.pi / 2.0]),
    )

    assert deskewed[0] == pytest.approx([1.0, 0.0, 0.0], abs=1.0e-6)
    assert deskewed[1] == pytest.approx([0.0, 1.0, 0.0], abs=1.0e-6)


def test_delayed_scan_match_correction_is_rate_limited():
    pose, limited = slam_runtime._limit_pose_correction(
        np.zeros(3, dtype=np.float64),
        np.asarray([0.15, -0.05, math.radians(12.0)], dtype=np.float64),
    )

    assert limited is True
    assert math.hypot(float(pose[0]), float(pose[1])) == pytest.approx(0.08)
    assert pose[2] == pytest.approx(math.radians(3.0))


def test_map_cells_average_repeated_registered_observations():
    session = {
        "keyframes": [
            {"points": np.asarray([[1.01, 0.0, 0.0]]), "pose": np.zeros(3)},
            {"points": np.asarray([[1.03, 0.0, 0.0]]), "pose": np.zeros(3)},
        ],
        "options": {"map_resolution_m": 0.1, "max_map_points": 100},
    }

    result = slam_runtime._map_points(session)

    assert result.shape == (1, 3)
    assert result[0] == pytest.approx([1.02, 0.0, 0.0], abs=1.0e-6)


def test_scan_matching_reuses_expanded_cells_until_reference_changes():
    session = {}
    reference = _points()

    first = slam_runtime._cached_expanded_cells(session, reference, 0.05)
    second = slam_runtime._cached_expanded_cells(session, reference, 0.05)
    replacement = slam_runtime._cached_expanded_cells(session, reference.copy(), 0.05)

    assert second is first
    assert replacement is not first
    assert replacement == first


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
    assert mapped["scene"]["slam"]["keyframes"] == 1
    assert mapped["map"]["point_count"] > 0
    assert mapped["scene"]["robot"] == {
        "length_m": 0.25,
        "width_m": 0.22,
        "height_m": 0.08,
    }
    assert mapped["scene"]["animation"]["enabled"] is True
    assert mapped["scene"]["animation"]["show_rays"] is True
    assert mapped["scene"]["colors"] == []
    assert mapped["scene"]["current_colors"] == []
    assert mapped["scene"]["map_render_mode"] == "occupancy-texture"
    assert mapped["scene"]["occupancy"]["backend"] == "warp"
    assert mapped["scene"]["occupancy"]["fixed_origin"] is True
    assert mapped["scene"]["occupancy"]["rays"] == len(local)
    assert mapped["scene"]["occupancy"]["encoding"] == "u2-base64"
    assert mapped["scene"]["occupancy"]["data"]
    assert "data" not in mapped["map"]["occupancy"]
    assert mapped["scene"]["floor_point_count"] > 0
    assert "occupied_points" in mapped["scene"]
    assert "occupied_colors" in mapped["scene"]
    assert mapped["scene"]["occupied_point_count"] == mapped["scene"]["occupancy"]["occupied_cells"]
    assert mapped["pose"]["kind"] == "blacknode.slam-pose"
    assert paused["scene"]["history_paused"] is True
    assert localized["live"] is True
    assert localized["scene"]["slam"]["keyframes"] == 1
    assert localized["status"]["mapping"] is False
    assert runtime["node_outputs"][0]["node_id"] == "slam-node"
    assert cleared["map"]["point_count"] == 0
    assert cleared["map"]["keyframes"] == 0
    assert cleared["scene"]["points"] == []
    assert cleared["scene"]["floor_points"] == []
    assert cleared["scene"]["occupied_points"] == []
    assert cleared["scene"]["occupancy"]["free_cells"] == 0
    assert cleared["scene"]["occupancy"]["occupied_cells"] == 0
    assert cleared["scene"]["current_points"] == localized["scene"]["current_points"]
    assert cleared["scene"]["view"] == localized["scene"]["view"]
    assert cleared["scene"]["sensor"] == localized["scene"]["sensor"]
    assert cleared["pose"] == localized["pose"]
    assert cleared["scene"]["history_paused"] is True
    slam_runtime.stop_slam()


def test_slam_node_declares_generic_stream_contract():
    fn = _NODE_REGISTRY["SLAM"]

    assert fn._bn_package == "blacknode-cuda"
    assert fn._bn_input_types["source"] == "Dict"
    assert fn._bn_input_types["odometry"] == "Dict"
    assert fn._bn_input_types["particle_localization"] == "Dict"
    assert fn._bn_input_types["dynamic_occupancy"] == "Dict"
    assert fn._bn_input_types["robot_length_m"] == "Float"
    assert fn._bn_input_types["robot_width_m"] == "Float"
    assert fn._bn_input_defaults["downsample_stride"] == 1
    assert fn._bn_input_defaults["tracking_min_score"] == 0.2
    assert fn._bn_input_defaults["mapping_min_score"] == 0.3
    assert fn._bn_input_defaults["occupancy_radius_m"] == 20.0
    assert fn._bn_output_types["scene"] == "Dict"
    assert fn._bn_output_types["pose"] == "Dict"
    assert fn._bn_output_types["map"] == "Dict"
    assert fn._bn_live_capable is True


def test_dynamic_occupancy_stage_is_explicit():
    result = _NODE_REGISTRY["WarpDynamicOccupancy"]({
        "enabled": True,
        "stable_radius_m": 0.07,
        "tracking_radius_m": 0.4,
        "minimum_speed_mps": 0.15,
        "maximum_points": 5_000,
        "display_points": 600,
        "trail_seconds": 0.25,
    })

    assert result["stage"]["kind"] == "blacknode.warp-dynamic-occupancy"
    assert result["stage"]["stable_radius_m"] == pytest.approx(0.07)
    assert result["stage"]["tracking_radius_m"] == pytest.approx(0.4)
    assert result["stage"]["maximum_points"] == 5_000
    assert result["stage"]["display_points"] == 600
    assert result["workload"] == "up to 5,000 registered returns per fresh scan"


def test_dynamic_stage_publishes_scene_motion_and_excludes_it_from_map(monkeypatch):
    slam_runtime.stop_slam()
    fixed = np.asarray([[float(index), 0.0, 0.0] for index in range(10)], dtype=np.float32)
    moved = fixed.copy()
    moved[4, 0] += 0.2
    processed_scans = [fixed, moved]
    process_index = 0

    def process_scan(*_args, **_kwargs):
        nonlocal process_index
        points = processed_scans[min(process_index, len(processed_scans) - 1)]
        process_index += 1
        return {
            "ok": True,
            "filtered_points": points.astype(float).tolist(),
            "filtered_indices": list(range(len(points))),
            "kernel_ms": 0.2,
        }

    monkeypatch.setattr(warp_points, "process_laser_scan", process_scan)
    stage = _NODE_REGISTRY["WarpDynamicOccupancy"]({
        "enabled": True,
        "stable_radius_m": 0.05,
        "tracking_radius_m": 0.3,
        "minimum_speed_mps": 0.1,
        "display_points": 32,
    })["stage"]
    reader_count = 0

    def read_scan(_source_value):
        nonlocal reader_count
        reader_count += 1
        sample_index = min(reader_count, 2)
        message = _message(1)
        message["message"]["header"]["stamp"]["nanosec"] = (
            100_000_000 if sample_index == 2 else 0
        )
        return {
            "messages": [message],
            "received": sample_index,
            "status": {
                "state": "ready",
                "source_fresh": True,
                "last_message_time_ns": 1_000_000_000 + (sample_index - 1) * 100_000_000,
            },
        }

    _NODE_REGISTRY["SLAM"]({
        "action": "start",
        "source": _source(),
        "dynamic_occupancy": stage,
        "slam_id": "dynamic-occupancy",
        "mode": "editor",
        "device": "cpu",
        "__node_id__": "dynamic-slam-node",
        "__message_stream_reader__": read_scan,
    })

    started = slam_runtime.slam_status("dynamic-occupancy")
    for _ in range(50):
        dynamic = started.get("scene", {}).get("dynamic_occupancy", {})
        if dynamic.get("state") == "ready":
            break
        time.sleep(0.02)
        started = slam_runtime.slam_status("dynamic-occupancy")

    dynamic = started["scene"]["dynamic_occupancy"]
    assert dynamic["state"] == "ready"
    assert dynamic["backend"] == "warp-hash-grid"
    assert dynamic["dynamic_points"] == 1
    assert len(started["scene"]["dynamic_points"]) == 1
    assert len(started["scene"]["dynamic_velocities"]) == 1
    assert "_dynamic_mask" not in dynamic
    assert started["scene"]["dynamic_points"][0][0] == pytest.approx(4.2)
    assert started["scene"]["occupancy"]["rays"] == 9
    slam_runtime.stop_slam()


def test_particle_stage_is_explicit_and_slam_executes_it(monkeypatch):
    slam_runtime.stop_slam()
    local = _points()

    stage_result = _NODE_REGISTRY["WarpParticleLocalization"]({
        "enabled": True,
        "particles": 96,
        "position_spread_m": 0.4,
        "yaw_spread_deg": 8.0,
        "display_particles": 64,
        "random_seed": 11,
    })

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
        "particle_localization": stage_result["stage"],
        "slam_id": "particle-localization",
        "mode": "editor",
        "device": "cpu",
        "__node_id__": "particle-slam-node",
        "__message_stream_reader__": lambda _source_value: {
            "message": _message(1),
            "messages": [_message(1)],
            "received": 1,
            "status": {
                "state": "ready",
                "source_fresh": True,
                "last_message_time_ns": 1_000_000_000,
            },
        },
    })

    localization = started["scene"]["localization"]
    assert stage_result["stage"]["kind"] == "blacknode.warp-particle-localization"
    assert stage_result["workload"] == "96 pose hypotheses per fresh scan"
    assert localization["state"] == "ready"
    assert localization["backend"] == "numpy"
    assert localization["requested_particles"] == 96
    assert localization["evaluated_particles"] == 96
    assert localization["beam_count"] == len(local)
    assert localization["work_items"] == 96 * len(local)
    assert 1.0 <= localization["effective_sample_size"] <= 96.0
    assert len(started["scene"]["particles"]) == 64
    assert len(started["scene"]["particle_scores"]) == 64
    assert started["scene"]["particles"] != started["scene"]["current_points"]
    slam_runtime.stop_slam()


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


def test_stationary_robot_does_not_grow_pose_graph_or_drift(monkeypatch):
    slam_runtime.stop_slam()
    state = {"received": 1}
    local = _points()

    def scan_reader(_source_value):
        message = _message(state["received"])["message"]
        return {
            "message": message,
            "messages": [message],
            "received": state["received"],
            "status": {"state": "ready", "source_fresh": True, "received": state["received"]},
        }

    monkeypatch.setattr(
        warp_points,
        "process_laser_scan",
        lambda *_args, **_kwargs: {
            "ok": True,
            "filtered_points": local.astype(float).tolist(),
            "kernel_ms": 0.1,
        },
    )
    _NODE_REGISTRY["SLAM"]({
        "action": "start",
        "source": _source(),
        "slam_id": "stationary",
        "mode": "editor",
        "device": "cpu",
        "keyframe_interval_s": 0.1,
        "__message_stream_reader__": scan_reader,
    })
    for received in range(2, 40):
        state["received"] = received
        result = slam_runtime.slam_status("stationary")

    assert result["scene"]["slam"]["keyframes"] == 1
    assert result["scene"]["slam"]["loop_closures"] == 0
    assert abs(result["pose"]["x_m"]) < 1.0e-9
    assert abs(result["pose"]["y_m"]) < 1.0e-9
    assert abs(result["pose"]["yaw_rad"]) < 1.0e-9
    assert result["status"]["received"] == 39
    slam_runtime.stop_slam()


def test_scan_matching_moves_robot_when_odometry_prior_is_unchanged(monkeypatch):
    slam_runtime.stop_slam()
    state = {"received": 1}
    local = _points()

    def scan_reader(_source_value):
        message = _message(state["received"])["message"]
        return {
            "message": message,
            "messages": [message],
            "received": state["received"],
            "status": {"state": "ready", "source_fresh": True, "received": state["received"]},
        }

    monkeypatch.setattr(
        warp_points,
        "process_laser_scan",
        lambda *_args, **_kwargs: {
            "ok": True,
            "filtered_points": local.astype(float).tolist(),
            "kernel_ms": 0.1,
        },
    )
    monkeypatch.setattr(
        slam_runtime,
        "_read_odometry",
        lambda _session, _scan: np.zeros(3, dtype=np.float64),
    )
    matched_pose = np.asarray([0.15, -0.05, math.radians(12.0)], dtype=np.float64)
    monkeypatch.setattr(
        slam_runtime,
        "correlative_match",
        lambda *_args, **_kwargs: (matched_pose.copy(), 0.9),
    )

    _NODE_REGISTRY["SLAM"]({
        "action": "start",
        "source": _source(),
        "odometry": _source(),
        "slam_id": "stationary-odom-prior",
        "mode": "editor",
        "device": "cpu",
        "__message_stream_reader__": scan_reader,
    })
    state["received"] = 2
    result = slam_runtime.slam_status("stationary-odom-prior")

    assert math.hypot(result["pose"]["x_m"], result["pose"]["y_m"]) == pytest.approx(0.08)
    assert result["pose"]["yaw_rad"] == pytest.approx(math.radians(3.0))
    assert result["status"]["tracking_correction_limited"] is True
    assert result["scene"]["sensor"]["yaw_rad"] == result["pose"]["yaw_rad"]
    assert result["scene"]["frame"] == "map"
    slam_runtime.stop_slam()


def test_fresh_stationary_odometry_blocks_dynamic_object_pose_jump(monkeypatch):
    slam_runtime.stop_slam()
    state = {"received": 1}
    local = _points()

    def scan_reader(_source_value):
        message = _message(state["received"])["message"]
        return {
            "message": message,
            "messages": [message],
            "received": state["received"],
            "status": {"state": "ready", "source_fresh": True, "received": state["received"]},
        }

    def stationary_odometry(session, _scan):
        session["odometry_sample_time_ns"] = state["received"] * 1_000_000_000
        return np.zeros(3, dtype=np.float64)

    monkeypatch.setattr(
        warp_points,
        "process_laser_scan",
        lambda *_args, **_kwargs: {
            "ok": True,
            "filtered_points": local.astype(float).tolist(),
            "kernel_ms": 0.1,
        },
    )
    monkeypatch.setattr(slam_runtime, "_read_odometry", stationary_odometry)
    monkeypatch.setattr(
        slam_runtime,
        "correlative_match",
        lambda *_args, **_kwargs: (
            np.asarray([0.25, -0.1, math.radians(8.0)], dtype=np.float64),
            0.9,
        ),
    )

    _NODE_REGISTRY["SLAM"]({
        "action": "start",
        "source": _source(),
        "odometry": _source(),
        "slam_id": "dynamic-hand",
        "mode": "editor",
        "device": "cpu",
        "__message_stream_reader__": scan_reader,
    })
    state["received"] = 2
    result = slam_runtime.slam_status("dynamic-hand")

    assert result["pose"]["x_m"] == pytest.approx(0.0)
    assert result["pose"]["y_m"] == pytest.approx(0.0)
    assert result["pose"]["yaw_rad"] == pytest.approx(0.0)
    assert result["status"]["stationary_odometry_locked"] is True
    assert result["status"]["scan_motion_override"] is False
    assert result["scene"]["slam"]["stationary_odometry_locked"] is True
    assert result["map"]["keyframes"] == 1
    slam_runtime.stop_slam()


def test_mapping_off_keeps_world_fixed_and_allows_strong_scan_rotation_to_move_robot(monkeypatch):
    slam_runtime.stop_slam()
    state = {"received": 1}
    local = _points()
    matched_pose = np.asarray([0.0, 0.0, math.radians(8.0)], dtype=np.float64)

    def scan_reader(_source_value):
        message = _message(state["received"])["message"]
        return {
            "message": message,
            "messages": [message],
            "received": state["received"],
            "status": {"state": "ready", "source_fresh": True, "received": state["received"]},
        }

    monkeypatch.setattr(
        warp_points,
        "process_laser_scan",
        lambda *_args, **_kwargs: {
            "ok": True,
            "filtered_points": local.astype(float).tolist(),
            "kernel_ms": 0.1,
        },
    )

    def stationary_odometry(session, _scan):
        session["odometry_sample_time_ns"] = state["received"] * 1_000_000_000
        return np.zeros(3, dtype=np.float64)

    monkeypatch.setattr(slam_runtime, "_read_odometry", stationary_odometry)
    monkeypatch.setattr(slam_runtime, "_pose_match_score", lambda *_args, **_kwargs: 0.1)
    monkeypatch.setattr(
        slam_runtime,
        "correlative_match",
        lambda *_args, **_kwargs: (matched_pose.copy(), 0.9),
    )

    started = _NODE_REGISTRY["SLAM"]({
        "action": "start",
        "source": _source(),
        "odometry": _source(),
        "slam_id": "mapping-off-rotation",
        "mode": "editor",
        "device": "cpu",
        "__message_stream_reader__": scan_reader,
    })
    frozen_points = started["scene"]["points"]
    _NODE_REGISTRY["SLAM"]({"action": "pause", "slam_id": "mapping-off-rotation"})
    state["received"] = 2
    results = [slam_runtime.slam_status("mapping-off-rotation")]
    for received in (3, 4):
        state["received"] = received
        results.append(slam_runtime.slam_status("mapping-off-rotation"))
    result = results[-1]

    assert result["scene"]["history_paused"] is True
    assert result["scene"]["points"] == frozen_points
    assert result["pose"]["yaw_rad"] == pytest.approx(matched_pose[2])
    assert result["scene"]["sensor"]["yaw_rad"] == pytest.approx(matched_pose[2])
    assert result["status"]["scan_motion_override"] is True
    assert result["status"]["stationary_odometry_locked"] is False
    assert result["map"]["keyframes"] == 1
    yaws = [value["pose"]["yaw_rad"] for value in results]
    assert yaws == pytest.approx([math.radians(3.0), math.radians(6.0), math.radians(8.0)])
    assert results[0]["status"]["tracking_correction_limited"] is True
    slam_runtime.stop_slam()


def test_clear_keeps_visible_map_empty_but_retains_hidden_localization(monkeypatch):
    slam_runtime.stop_slam()
    state = {"received": 1}
    local = _points()
    matched_pose = np.asarray([0.0, 0.0, math.radians(7.0)], dtype=np.float64)

    def scan_reader(_source_value):
        message = _message(state["received"])["message"]
        return {
            "message": message,
            "messages": [message],
            "received": state["received"],
            "status": {"state": "ready", "source_fresh": True, "received": state["received"]},
        }

    monkeypatch.setattr(
        warp_points,
        "process_laser_scan",
        lambda *_args, **_kwargs: {
            "ok": True,
            "filtered_points": local.astype(float).tolist(),
            "kernel_ms": 0.1,
        },
    )
    monkeypatch.setattr(slam_runtime, "_pose_match_score", lambda *_args, **_kwargs: 0.1)
    monkeypatch.setattr(
        slam_runtime,
        "correlative_match",
        lambda *_args, **_kwargs: (matched_pose.copy(), 0.9),
    )

    _NODE_REGISTRY["SLAM"]({
        "action": "start",
        "source": _source(),
        "slam_id": "clear-localization",
        "mode": "editor",
        "device": "cpu",
        "__message_stream_reader__": scan_reader,
    })
    _NODE_REGISTRY["SLAM"]({"action": "clear", "slam_id": "clear-localization"})
    state["received"] = 2
    seeded = slam_runtime.slam_status("clear-localization")
    state["received"] = 3
    localized = slam_runtime.slam_status("clear-localization")

    assert seeded["map"]["point_count"] == 0
    assert seeded["scene"]["points"] == []
    assert seeded["scene"]["current_point_count"] == len(local)
    assert localized["map"]["point_count"] == 0
    assert localized["scene"]["points"] == []
    assert 0.0 < localized["pose"]["yaw_rad"] <= matched_pose[2]
    assert localized["scene"]["sensor"]["yaw_rad"] == pytest.approx(localized["pose"]["yaw_rad"])
    assert localized["map"]["keyframes"] == 0
    slam_runtime.stop_slam()


def test_low_confidence_match_keeps_odometry_pose_and_rejects_map_update(monkeypatch):
    slam_runtime.stop_slam()
    state = {"received": 1}
    local = _points()

    def scan_reader(_source_value):
        message = _message(state["received"])["message"]
        return {
            "message": message,
            "messages": [message],
            "received": state["received"],
            "status": {"state": "ready", "source_fresh": True, "received": state["received"]},
        }

    monkeypatch.setattr(
        warp_points,
        "process_laser_scan",
        lambda *_args, **_kwargs: {
            "ok": True,
            "filtered_points": local.astype(float).tolist(),
            "kernel_ms": 0.1,
        },
    )
    monkeypatch.setattr(
        slam_runtime,
        "_read_odometry",
        lambda _session, _scan: np.asarray(
            [0.2 if state["received"] > 1 else 0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
    )
    monkeypatch.setattr(
        slam_runtime,
        "correlative_match",
        lambda *_args, **_kwargs: (np.asarray([2.0, 2.0, 1.0]), 0.1),
    )

    _NODE_REGISTRY["SLAM"]({
        "action": "start",
        "source": _source(),
        "odometry": _source(),
        "slam_id": "reject-uncertain-map",
        "mode": "editor",
        "device": "cpu",
        "tracking_min_score": 0.2,
        "mapping_min_score": 0.3,
        "__message_stream_reader__": scan_reader,
    })
    state["received"] = 2
    result = slam_runtime.slam_status("reject-uncertain-map")

    assert result["pose"]["x_m"] == pytest.approx(0.2)
    assert result["pose"]["y_m"] == pytest.approx(0.0)
    assert result["pose"]["yaw_rad"] == pytest.approx(0.0)
    assert result["status"]["tracking_accepted"] is False
    assert result["status"]["map_update_rejected"] is True
    assert result["map"]["keyframes"] == 1
    slam_runtime.stop_slam()


def test_runtime_status_uses_cached_snapshot_without_running_scan_matching(monkeypatch):
    snapshot = {
        "running": True,
        "live": True,
        "status": {"state": "ready"},
        "scene": {"point_count": 12},
    }
    session = {"node_id": "slam-node", "snapshot": snapshot}
    with slam_runtime._LOCK:
        slam_runtime._SESSIONS["cached-status"] = session
    monkeypatch.setattr(
        slam_runtime,
        "_update_session",
        lambda _session: (_ for _ in ()).throw(AssertionError("status must not process scans")),
    )
    try:
        runtime = slam_runtime.runtime_status()
    finally:
        with slam_runtime._LOCK:
            slam_runtime._SESSIONS.pop("cached-status", None)

    assert runtime["active"] is True
    assert runtime["node_outputs"][0]["outputs"] == snapshot
