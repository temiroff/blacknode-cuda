"""Provider-neutral metric depth projection and managed Viewer integration."""
from __future__ import annotations

import json
import struct
import time
from pathlib import Path

import numpy as np

import blacknode  # noqa: F401
from blacknode.node import _NODE_REGISTRY
from blacknode.pkg.blacknode_cuda import viewer_runtime, warp_depth


def _depth_stream() -> dict:
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
            "width": 3,
            "height": 2,
            "fx": 2.0,
            "fy": 2.0,
            "cx": 1.0,
            "cy": 0.5,
        },
        "frame_source": {
            "kind": "blacknode.depth-frame-source",
            "schema_version": 1,
            "transport": "inline",
            "frame": "depth_optical",
            "width": 3,
            "height": 2,
            "encoding": "32FC1",
            "depth_scale": 1.0,
            "source_time_ns": 42,
            "depth_m": [1.0, 1.0, 1.0, 1.0, 0.0, 2.0],
        },
    }


def _stage(**values) -> dict:
    return _NODE_REGISTRY["WarpDepthProjector"]({
        "minimum_depth_m": 0.1,
        "maximum_depth_m": 4.0,
        "downsample_stride": 1,
        "maximum_points": 100,
        **values,
    })["stage"]


def _uniform_depth_stream(width: int, height: int, values: list[float], source_time_ns: int) -> dict:
    return {
        "kind": "blacknode.depth-stream",
        "schema_version": 1,
        "frame": "depth_optical",
        "encoding": "32FC1",
        "calibration": {
            "kind": "blacknode.camera-calibration",
            "schema_version": 1,
            "width": width,
            "height": height,
            "fx": float(max(width, 1)),
            "fy": float(max(height, 1)),
            "cx": (width - 1) * 0.5,
            "cy": (height - 1) * 0.5,
        },
        "frame_source": {
            "kind": "blacknode.depth-frame-source",
            "schema_version": 1,
            "transport": "inline",
            "frame": "depth_optical",
            "width": width,
            "height": height,
            "encoding": "32FC1",
            "source_time_ns": source_time_ns,
            "depth_m": values,
        },
    }


def test_depth_projector_registers_as_visible_managed_stage():
    fn = _NODE_REGISTRY["WarpDepthProjector"]

    assert fn._bn_package == "blacknode-cuda"
    assert fn._bn_component == "spatial-processing"
    assert fn._bn_output_types["stage"] == "Dict"
    assert _stage()["kind"] == "blacknode.warp-depth-projector"
    assert _NODE_REGISTRY["Viewer"]._bn_input_types["depth_projection"] == "Dict"


def test_inline_metric_depth_projects_to_forward_left_up_on_cpu():
    result = warp_depth.process_depth_stream(_depth_stream(), _stage(), device="cpu")

    assert result["ok"] is True
    assert result["report"]["input_pixels"] == 6
    assert result["report"]["valid_points"] == 5
    assert result["report"]["display_points"] == 5
    np.testing.assert_allclose(result["filtered_points"][0], [1.0, 0.5, 0.25], atol=1.0e-6)
    assert result["point_cloud"]["frame"] == "base_link"
    assert result["point_cloud"]["source_frame"] == "depth_optical"
    assert len(result["normals"]) == 5
    assert len(result["confidence"]) == 5


def test_depth_projection_applies_selected_ir_intensity_to_points():
    pixels = [
        16, 16, 16,
        32, 32, 32,
        64, 64, 64,
        96, 96, 96,
        128, 128, 128,
        255, 255, 255,
    ]
    result = warp_depth.process_depth_stream(
        _depth_stream(),
        _stage(),
        device="cpu",
        color_stream={
            "kind": "blacknode.frame-stream",
            "width": 3,
            "height": 2,
            "pixels_rgb": pixels,
        },
        color_mode="ir",
    )

    assert result["ok"] is True
    assert result["report"]["color_mode"] == "ir"
    assert result["report"]["color_applied"] == "ir"
    assert result["report"]["color_registered"] is True
    np.testing.assert_allclose(result["colors"][0], [16 / 255] * 3, atol=1.0e-6)


def test_depth_projection_applies_calibrated_sensor_extrinsics():
    result = warp_depth.process_depth_stream(
        _depth_stream(),
        _stage(sensor_x_m=0.25, sensor_z_m=0.5, sensor_yaw_rad=np.pi / 2),
        device="cpu",
    )

    np.testing.assert_allclose(result["filtered_points"][0], [-0.25, 1.0, 0.75], atol=1.0e-5)
    assert result["report"]["sensor_extrinsics"]["x_m"] == 0.25


def test_depth_cleanup_fills_small_hole_with_consistent_neighbors():
    values = [1.0] * 9
    values[4] = 0.0
    result = warp_depth.process_depth_stream(
        _uniform_depth_stream(3, 3, values, 100),
        _stage(minimum_neighbors=4, temporal_smoothing=0.0),
        device="cpu",
    )

    assert result["ok"] is True
    assert result["report"]["valid_points"] == 9
    assert result["report"]["cleanup"]["holes_filled"] == 1


def test_temporal_cleanup_reuses_buffers_but_does_not_smear_large_motion():
    state = warp_depth.WarpDepthProcessingState()
    stage = _stage(
        spatial_filter=False,
        hole_fill=False,
        outlier_rejection=False,
        temporal_smoothing=0.5,
        temporal_max_delta_m=0.08,
    )
    first = warp_depth.process_depth_stream(
        _uniform_depth_stream(3, 3, [1.0] * 9, 100),
        stage,
        device="cpu",
        processing_state=state,
    )
    stable = warp_depth.process_depth_stream(
        _uniform_depth_stream(3, 3, [1.04] * 9, 200),
        stage,
        device="cpu",
        processing_state=state,
    )
    moved = warp_depth.process_depth_stream(
        _uniform_depth_stream(3, 3, [2.0] * 9, 300),
        stage,
        device="cpu",
        processing_state=state,
    )

    assert first["report"]["buffers_reused"] is False
    assert stable["report"]["buffers_reused"] is True
    assert stable["report"]["cleanup"]["temporally_blended"] == 9
    np.testing.assert_allclose(stable["filtered_points"][4][0], 1.02, atol=1.0e-6)
    assert moved["report"]["cleanup"]["temporally_blended"] == 0
    np.testing.assert_allclose(moved["filtered_points"][4][0], 2.0, atol=1.0e-6)


def test_projection_increases_stride_before_allocating_more_than_point_budget():
    result = warp_depth.process_depth_stream(
        _uniform_depth_stream(20, 20, [1.0] * 400, 100),
        _stage(maximum_points=64, downsample_stride=1),
        device="cpu",
    )

    assert result["ok"] is True
    assert result["report"]["requested_stride"] == 1
    assert result["report"]["stride"] == 3
    assert result["report"]["candidate_points"] == 49
    assert result["report"]["display_points"] == 49


def test_binary_metric_depth_decoder_preserves_pixels_without_json_lists():
    pixels = struct.pack("<HHHH", 100, 500, 1000, 0)
    header = json.dumps({
        "kind": "blacknode.metric-depth-frame",
        "schema_version": 1,
        "width": 2,
        "height": 2,
        "step": 4,
        "encoding": "16UC1",
        "is_bigendian": False,
    }, separators=(",", ":")).encode("utf-8")
    payload = b"BNDEPTH1" + struct.pack("<I", len(header)) + header + pixels

    depth, metadata = warp_depth._decode_binary(payload)

    assert metadata["encoding"] == "16UC1"
    np.testing.assert_array_equal(depth, [[100.0, 500.0], [1000.0, 0.0]])


def test_binary_depth_source_applies_provider_metric_scale(monkeypatch):
    pixels = struct.pack("<HHHH", 100, 500, 1000, 0)
    header = json.dumps({
        "kind": "blacknode.metric-depth-frame",
        "schema_version": 1,
        "width": 2,
        "height": 2,
        "step": 4,
        "encoding": "16UC1",
        "is_bigendian": False,
        "received_at_ns": time.time_ns(),
    }, separators=(",", ":")).encode("utf-8")
    payload = b"BNDEPTH1" + struct.pack("<I", len(header)) + header + pixels

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit):
            return payload

    monkeypatch.setattr(warp_depth.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    depth, _, _ = warp_depth.load_metric_depth({
        "kind": "blacknode.depth-stream",
        "depth_scale": 0.001,
        "frame_source": {
            "kind": "blacknode.depth-frame-source",
            "transport": "http-binary",
            "url": "http://127.0.0.1/frame.bin",
            "depth_scale": 0.001,
        },
    })

    np.testing.assert_allclose(depth, [[0.1, 0.5], [1.0, 0.0]], atol=1.0e-6)


def test_live_binary_depth_distinguishes_worker_presence_from_stale_source(monkeypatch):
    stream = _depth_stream()
    stream["frame_source"] = {
        "kind": "blacknode.depth-frame-source",
        "schema_version": 1,
        "transport": "http-binary",
        "url": "http://127.0.0.1:9/frame.bin",
        "depth_scale": 1.0,
    }
    monkeypatch.setattr(
        warp_depth,
        "load_metric_depth",
        lambda _stream: (
            np.ones((2, 3), dtype=np.float32),
            {
                "width": 3,
                "height": 2,
                "encoding": "32FC1",
                "received_at_ns": time.time_ns() - 5_000_000_000,
            },
            0.1,
        ),
    )

    result = warp_depth.process_depth_stream(stream, _stage(stale_after_seconds=1.0), device="cpu")

    assert result["ok"] is False
    assert result["report"]["state"] == "stale"
    assert result["report"]["worker_alive"] is True
    assert result["report"]["source_fresh"] is False


def test_managed_viewer_renders_depth_surface_with_projection_metrics():
    viewer_runtime.stop_viewer()
    try:
        result = _NODE_REGISTRY["Viewer"]({
            "action": "start",
            "source": _depth_stream(),
            "depth_projection": _stage(),
            "viewer_id": "depth-surface",
            "mode": "editor",
            "device": "cpu",
        })

        assert result["live"] is True
        assert result["scene"]["projection"] == "xyz"
        assert result["scene"]["point_count"] == 5
        assert result["scene"]["depth_projection"]["backend"] == "warp"
        assert result["scene"]["depth_projection"]["input_pixels"] == 6
        assert "metric depth points" in result["report"]
        refreshed = viewer_runtime.viewer_status("depth-surface")
        assert refreshed["scene"]["depth_projection"]["buffers_reused"] is True
        assert refreshed["scene"]["depth_projection"]["duplicate_frame"] is True
    finally:
        viewer_runtime.stop_viewer()


def test_depth_viewer_requires_visible_projector_stage():
    result = _NODE_REGISTRY["Viewer"]({
        "action": "start",
        "source": _depth_stream(),
        "viewer_id": "missing-stage",
        "mode": "editor",
        "device": "cpu",
    })

    assert result["running"] is False
    assert "WarpDepthProjector" in result["report"]


def test_roadmap_marks_live_depth_projection_as_delivered():
    roadmap = Path(__file__).resolve().parents[1] / "SENSOR_GPU_ROADMAP.md"
    assert "Status: implemented in `WarpDepthProjector`." in roadmap.read_text(encoding="utf-8")
