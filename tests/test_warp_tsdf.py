"""Persistent Warp RGB-D reconstruction and managed Viewer coverage."""
from __future__ import annotations

import numpy as np
from pathlib import Path

import blacknode  # noqa: F401
from blacknode.node import _NODE_REGISTRY
from blacknode.pkg.blacknode_cuda import viewer_runtime
from blacknode.pkg.blacknode_cuda.warp_tsdf import WarpTSDFVolume


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
            "width": 3, "height": 2,
            "fx": 2.0, "fy": 2.0, "cx": 1.0, "cy": 0.5,
        },
        "frame_source": {
            "kind": "blacknode.depth-frame-source",
            "schema_version": 1,
            "transport": "inline",
            "frame": "depth_optical",
            "width": 3, "height": 2,
            "encoding": "32FC1", "depth_scale": 1.0,
            "source_time_ns": source_time_ns,
            "depth_m": [1.0, 1.0, 1.0, 1.0, 0.0, 2.0],
        },
    }


def _projector() -> dict:
    return _NODE_REGISTRY["WarpDepthProjector"]({
        "minimum_depth_m": 0.1,
        "maximum_depth_m": 4.0,
        "downsample_stride": 1,
        "maximum_points": 100,
    })["stage"]


def _integration(**values) -> dict:
    return _NODE_REGISTRY["WarpTSDFIntegration"]({
        "require_pose": False,
        "voxel_size_m": 0.15,
        "truncation_m": 0.3,
        "volume_radius_m": 1.5,
        "volume_origin_x_m": 1.0,
        "volume_origin_z_m": 0.0,
        "maximum_voxels": 8_000,
        "samples_per_ray": 5,
        **values,
    })["stage"]


def _extraction(**values) -> dict:
    return _NODE_REGISTRY["WarpSurfaceExtraction"]({
        "surface_band": 1.0,
        "minimum_weight": 1.0,
        "maximum_points": 1_000,
        **values,
    })["stage"]


def _color_stream() -> dict:
    return {
        "kind": "blacknode.frame-stream",
        "schema_version": 1,
        "stream_id": "rgb",
        "width": 3,
        "height": 2,
        "pixels_rgb": [
            255, 0, 0, 0, 255, 0, 0, 0, 255,
            255, 255, 0, 255, 0, 255, 0, 255, 255,
        ],
    }


def _pose_stream() -> dict:
    return {
        "kind": "blacknode.message-stream",
        "schema_version": 1,
        "protocol": "ros2",
        "topic": "/odom",
        "message_type": "nav_msgs/msg/Odometry",
    }


def _pose_outputs() -> dict:
    message = {
        "header": {"stamp": {"sec": 0, "nanosec": 42}, "frame_id": "odom"},
        "child_frame_id": "base_link",
        "pose": {"pose": {
            "position": {"x": 0.5, "y": 0.25, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }},
    }
    return {
        "message": message,
        "messages": [message],
        "status": {"state": "ready", "source_fresh": True, "received": 1, "error": ""},
        "received": 1,
    }


def test_reconstruction_stages_are_graph_visible_and_typed():
    integration = _NODE_REGISTRY["WarpTSDFIntegration"]
    extraction = _NODE_REGISTRY["WarpSurfaceExtraction"]

    assert integration._bn_package == "blacknode-cuda"
    assert integration._bn_output_types["stage"] == "Dict"
    assert extraction._bn_output_types["stage"] == "Dict"
    assert _integration()["kind"] == "blacknode.warp-tsdf-integration"
    assert _extraction()["kind"] == "blacknode.warp-surface-extraction"
    viewer = _NODE_REGISTRY["Viewer"]
    assert viewer._bn_input_types["color_source"] == "Dict"
    assert viewer._bn_input_types["tsdf_integration"] == "Dict"
    assert viewer._bn_input_types["surface_extraction"] == "Dict"


def test_cpu_tsdf_volume_persists_frames_and_extracts_a_surface():
    stage = _integration()
    volume = WarpTSDFVolume(stage, device="cpu")
    point_cloud = {
        "points_xyz": [[1.0, -0.2, 0.0], [1.0, 0.0, 0.0], [1.0, 0.2, 0.0]],
        "colors_rgb": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "processing": {"sensor_extrinsics": {}},
    }
    pose = {"x_m": 0.0, "y_m": 0.0, "z_m": 0.0, "yaw_rad": 0.0}

    first = volume.integrate(point_cloud, pose=pose, stage=stage)
    second = volume.integrate(point_cloud, pose=pose, stage=stage)
    surface = volume.extract(_extraction())

    assert first["ok"] is True
    assert second["report"]["frames_integrated"] == 2
    assert surface["ok"] is True
    assert surface["report"]["observed_voxels"] > 0
    assert surface["report"]["surface_voxels"] > 0
    assert len(surface["points"]) == len(surface["colors"])
    assert np.asarray(surface["points"]).shape[1] == 3


def test_managed_viewer_builds_colored_surface_once_per_fresh_depth_frame():
    viewer_runtime.stop_viewer()
    try:
        result = _NODE_REGISTRY["Viewer"]({
            "action": "start",
            "source": _depth_stream(),
            "color_source": _color_stream(),
            "depth_projection": _projector(),
            "tsdf_integration": _integration(),
            "surface_extraction": _extraction(),
            "viewer_id": "rgbd-reconstruction",
            "mode": "editor",
            "device": "cpu",
        })

        assert result["live"] is True
        assert result["scene"]["primitive"] == "rgbd-surface"
        assert result["scene"]["reconstruction"]["color_registered"] is True
        assert result["scene"]["reconstruction"]["integration"]["frames_integrated"] == 1
        assert result["scene"]["reconstruction"]["extraction"]["observed_voxels"] > 0
        assert result["scene"]["history_registered"] is True
        assert "reconstructed" in result["report"]

        unchanged = viewer_runtime.viewer_status("rgbd-reconstruction")
        assert unchanged["scene"]["reconstruction"]["integration"]["frames_integrated"] == 1

        cleared = viewer_runtime.clear_viewer("rgbd-reconstruction")
        assert cleared["scene"]["point_count"] == 0
        assert cleared["scene"]["history_paused"] is True
        viewer_runtime.resume_viewer("rgbd-reconstruction")
        rebuilt = viewer_runtime.viewer_status("rgbd-reconstruction")
        assert rebuilt["scene"]["reconstruction"]["integration"]["frames_integrated"] == 1
        assert rebuilt["scene"]["point_count"] > 0
    finally:
        viewer_runtime.stop_viewer()


def test_pose_registered_reconstruction_requires_a_connected_pose_by_default():
    result = _NODE_REGISTRY["Viewer"]({
        "action": "start",
        "source": _depth_stream(),
        "depth_projection": _projector(),
        "tsdf_integration": _integration(require_pose=True),
        "surface_extraction": _extraction(),
        "viewer_id": "rgbd-missing-pose",
        "mode": "editor",
        "device": "cpu",
    })

    assert result["running"] is False
    assert "pose" in result["report"].lower()


def test_managed_reconstruction_registers_depth_in_the_pose_frame():
    viewer_runtime.stop_viewer()
    try:
        result = _NODE_REGISTRY["Viewer"]({
            "action": "start",
            "source": _depth_stream(),
            "depth_projection": _projector(),
            "tsdf_integration": _integration(require_pose=True),
            "surface_extraction": _extraction(),
            "pose": _pose_stream(),
            "pose_child_frame": "base_link",
            "viewer_id": "rgbd-pose",
            "mode": "editor",
            "device": "cpu",
            "__message_stream_reader__": lambda _source: _pose_outputs(),
        })

        assert result["live"] is True
        assert result["scene"]["frame"] == "odom"
        assert result["scene"]["registration"]["x_m"] == 0.5
        assert result["scene"]["registration"]["y_m"] == 0.25
        assert result["scene"]["reconstruction"]["pose_registered"] is True
    finally:
        viewer_runtime.stop_viewer()


def test_roadmap_marks_rgbd_reconstruction_as_delivered():
    roadmap = Path(__file__).resolve().parents[1] / "SENSOR_GPU_ROADMAP.md"
    content = roadmap.read_text(encoding="utf-8")
    assert "Status: implemented in `WarpTSDFIntegration` and `WarpSurfaceExtraction`." in content
