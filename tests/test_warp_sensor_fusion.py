"""Warp LiDAR, metric-depth, RGB, pose, and Viewer fusion coverage."""
from __future__ import annotations

import math
from pathlib import Path

import blacknode  # noqa: F401
from blacknode.node import _NODE_REGISTRY
from blacknode.pkg.blacknode_cuda import viewer_runtime
from blacknode.pkg.blacknode_cuda.warp_fusion import process_sensor_fusion


def _stage(**values) -> dict:
    return _NODE_REGISTRY["WarpSensorFusion"]({
        "require_pose": False,
        "synchronization_tolerance_s": 0.1,
        "maximum_alignment_distance_m": 0.3,
        "minimum_depth_confidence": 0.1,
        "maximum_points": 1_000,
        "calibration_search": True,
        "calibration_translation_m": 0.1,
        "calibration_yaw_deg": 0.0,
        "calibration_steps": 3,
        **values,
    })["stage"]


def _projector() -> dict:
    return _NODE_REGISTRY["WarpDepthProjector"]({
        "minimum_depth_m": 0.1,
        "maximum_depth_m": 4.0,
        "downsample_stride": 1,
        "maximum_points": 100,
        "target_frame": "base_link",
    })["stage"]


def _point_cloud(offset_x: float = 0.1) -> dict:
    return {
        "points_xyz": [[1.0 + offset_x, y, 0.0] for y in (-0.2, 0.0, 0.2)],
        "colors_rgb": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "confidence": [1.0, 1.0, 1.0],
        "processing": {"sensor_extrinsics": {}},
    }


def _depth_stream(source_time_ns: int = 42) -> dict:
    return {
        "kind": "blacknode.depth-stream",
        "schema_version": 1,
        "frame": "depth_optical",
        "encoding": "32FC1",
        "depth_scale": 1.0,
        "calibration": {
            "kind": "blacknode.camera-calibration",
            "schema_version": 1,
            "camera_model": "pinhole",
            "width": 3, "height": 1,
            "fx": 5.0, "fy": 5.0, "cx": 1.0, "cy": 0.0,
        },
        "frame_source": {
            "kind": "blacknode.depth-frame-source",
            "schema_version": 1,
            "transport": "inline",
            "frame": "depth_optical",
            "width": 3, "height": 1,
            "encoding": "32FC1", "depth_scale": 1.0,
            "source_time_ns": source_time_ns,
            "depth_m": [1.0, 1.0, 1.0],
        },
    }


def _lidar_scan(source_time_ns: int = 42) -> dict:
    angle = math.atan(0.2)
    distance = math.sqrt(1.0 + 0.2 ** 2)
    return {
        "kind": "blacknode.laser-scan-stream",
        "schema_version": 1,
        "frame": "base_link",
        "source_time_ns": source_time_ns,
        "receive_time_ns": source_time_ns,
        "angle_min": -angle,
        "angle_max": angle,
        "angle_increment": angle,
        "range_min": 0.05,
        "range_max": 10.0,
        "ranges": [distance, 1.0, distance],
        "intensities": [],
    }


def _color_stream() -> dict:
    return {
        "kind": "blacknode.frame-stream",
        "schema_version": 1,
        "stream_id": "aligned-rgb",
        "width": 3, "height": 1,
        "pixels_rgb": [255, 0, 0, 0, 255, 0, 0, 0, 255],
    }


def test_fusion_stage_is_graph_visible_and_viewer_ports_are_typed():
    fusion = _NODE_REGISTRY["WarpSensorFusion"]
    viewer = _NODE_REGISTRY["Viewer"]

    assert fusion._bn_package == "blacknode-cuda"
    assert fusion._bn_output_types["stage"] == "Dict"
    assert _stage()["kind"] == "blacknode.warp-sensor-fusion"
    assert viewer._bn_input_types["lidar_source"] == "Dict"
    assert viewer._bn_input_types["sensor_fusion"] == "Dict"


def test_cpu_hash_grid_calibration_recovers_depth_translation():
    result = process_sensor_fusion(
        [[1.0, y, 0.0] for y in (-0.2, 0.0, 0.2)],
        _point_cloud(offset_x=0.1),
        pose={"x_m": 0.0, "y_m": 0.0, "z_m": 0.0, "yaw_rad": 0.0},
        stage=_stage(compare_cpu=True),
        device="cpu",
    )

    assert result["ok"] is True
    report = result["report"]
    assert report["backend"] == "warp-hash-grid"
    assert report["calibration_hypotheses"] == 27
    assert report["matched_points"] == 3
    assert abs(report["correction"]["x_m"] + 0.1) < 1.0e-5
    assert report["mean_residual_m"] < 1.0e-5
    assert report["correction_error"] < 1.0e-6


def test_managed_viewer_publishes_a_colorized_sensor_fusion_scene():
    viewer_runtime.stop_viewer()
    try:
        result = _NODE_REGISTRY["Viewer"]({
            "action": "start",
            "source": _depth_stream(),
            "lidar_source": _lidar_scan(),
            "color_source": _color_stream(),
            "depth_projection": _projector(),
            "sensor_fusion": _stage(calibration_translation_m=0.0),
            "viewer_id": "sensor-fusion",
            "mode": "editor",
            "device": "cpu",
            "filter_min_m": 0.05,
            "filter_max_m": 4.0,
        })

        assert result["live"] is True
        assert result["scene"]["primitive"] == "sensor-fusion"
        assert result["scene"]["sensor_fusion"]["lidar_points"] == 3
        assert result["scene"]["sensor_fusion"]["depth_points"] == 3
        assert result["scene"]["sensor_fusion"]["matched_points"] == 3
        assert result["scene"]["depth_projection"]["color_registered"] is True
        assert result["scene"]["synchronization"]["delta_seconds"] == 0.0
        assert "mean residual" in result["report"]

        unchanged = viewer_runtime.viewer_status("sensor-fusion")
        assert unchanged["scene"]["sequence"] == result["scene"]["sequence"]
    finally:
        viewer_runtime.stop_viewer()


def test_managed_fusion_rejects_unsynchronized_sensor_frames():
    viewer_runtime.stop_viewer()
    try:
        result = _NODE_REGISTRY["Viewer"]({
            "action": "start",
            "source": _depth_stream(2_000_000_000),
            "lidar_source": _lidar_scan(1_000_000_000),
            "depth_projection": _projector(),
            "sensor_fusion": _stage(synchronization_tolerance_s=0.05),
            "viewer_id": "sensor-fusion-stale",
            "mode": "editor",
            "device": "cpu",
        })

        assert result["live"] is False
        assert result["status"]["state"] == "stale"
        assert "apart" in result["report"]
    finally:
        viewer_runtime.stop_viewer()


def test_fusion_requires_a_lidar_source():
    result = _NODE_REGISTRY["Viewer"]({
        "action": "start",
        "source": _depth_stream(),
        "depth_projection": _projector(),
        "sensor_fusion": _stage(),
        "viewer_id": "sensor-fusion-missing-lidar",
        "mode": "editor",
        "device": "cpu",
    })

    assert result["running"] is False
    assert "LiDAR" in result["report"]


def test_roadmap_marks_cross_sensor_fusion_as_delivered():
    roadmap = Path(__file__).resolve().parents[1] / "SENSOR_GPU_ROADMAP.md"
    content = roadmap.read_text(encoding="utf-8")
    assert "Status: implemented in `WarpSensorFusion`." in content
