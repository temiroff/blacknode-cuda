"""Warp LaserScan processing and managed viewer contracts."""
import math
import json
from pathlib import Path

import numpy as np
import pytest

import blacknode  # noqa: F401
from blacknode.node import _NODE_REGISTRY
from blacknode.pkg.blacknode_cuda import warp_points
from blacknode.pkg.blacknode_cuda import warp_viewer_runtime


def _scan():
    return {
        "kind": "blacknode.laser-scan-stream",
        "schema_version": 1,
        "frame": "laser",
        "source_time_ns": 42,
        "angle_min": 0.0,
        "angle_max": 1.5 * math.pi,
        "angle_increment": 0.5 * math.pi,
        "range_min": 0.1,
        "range_max": 10.0,
        "ranges": [1.0, 2.0, float("nan"), 4.0],
    }


def test_warp_nodes_register_to_spatial_component():
    for name in ("WarpLaserScanFilter", "WarpLiDARViewer"):
        fn = _NODE_REGISTRY[name]
        assert fn._bn_package == "blacknode-cuda"
        assert fn._bn_component == "spatial-processing"


def test_warp_filter_converts_filters_transforms_and_colors_on_cpu():
    result = warp_points.process_laser_scan(
        _scan(),
        device="cpu",
        filter_min_m=0.5,
        filter_max_m=3.0,
        stride=1,
        sensor_pose=(1.0, 0.0, 0.0),
    )

    assert result["ok"] is True
    assert result["raw_count"] == 3
    assert result["filtered_count"] == 2
    assert result["filtered_points"][0] == [2.0, 0.0, 0.0]
    assert abs(result["filtered_points"][1][0] - 1.0) < 1e-6
    assert abs(result["filtered_points"][1][1] - 2.0) < 1e-6
    assert result["colors"][0][0] == 0.0
    assert result["colors"][0][2] == 1.0
    assert result["point_cloud"]["frame"] == "base_link"
    assert result["report"]["implementation"] == "NVIDIA Warp kernel"


def test_partial_scan_forward_heading_uses_angular_sector_center():
    heading = warp_points._scan_forward_yaw(
        {"angle_min": math.radians(72.0), "angle_max": math.radians(291.0)},
        0.0,
    )

    assert abs(math.atan2(math.sin(heading - math.pi), math.cos(heading - math.pi))) < math.radians(2.0)
    assert warp_points._scan_forward_yaw(
        {"angle_min": -math.pi, "angle_max": math.pi},
        0.35,
    ) == pytest.approx(0.35)


def test_warp_filter_downsamples_and_invalid_input_is_structured():
    downsampled = _NODE_REGISTRY["WarpLaserScanFilter"]({
        "laser_scan": _scan(),
        "device": "cpu",
        "downsample_stride": 2,
        "filter_max_m": 10.0,
    })
    missing = _NODE_REGISTRY["WarpLaserScanFilter"]({"laser_scan": {}, "device": "cpu"})

    assert downsampled["filtered_count"] == 1
    assert missing["ok"] is False
    assert "error" in missing["report"]


def test_warp_filter_can_skip_raw_python_output_for_dense_scans():
    result = warp_points.process_laser_scan(
        _scan(),
        device="cpu",
        filter_max_m=10.0,
        include_raw_points=False,
    )

    assert result["ok"] is True
    assert result["raw_count"] == 3
    assert result["raw_points"] == []
    assert result["report"]["raw_output_count"] == 0
    assert result["filtered_count"] == 3


def test_warp_filter_compares_warmed_numpy_reference():
    result = warp_points.process_laser_scan(
        _scan(),
        device="cpu",
        filter_max_m=10.0,
        include_raw_points=False,
        compare_numpy=True,
    )

    benchmark = result["report"]["benchmark"]
    assert benchmark["dtype"] == "float32"
    assert benchmark["warmup_runs"] == 1
    assert benchmark["measured_runs"] == 3
    assert benchmark["warp_gpu_ms"] > 0.0
    assert benchmark["numpy_cpu_ms"] > 0.0
    assert benchmark["warp_kernel_speedup"] > 0.0
    assert benchmark["warp_end_to_end_ms"] > 0.0
    assert benchmark["end_to_end_speedup"] > 0.0
    assert benchmark["max_abs_error_m"] < 1.0e-5


def test_cuda_opengl_kernel_writes_current_and_bounded_history_on_cpu():
    wp = warp_points.wp
    if wp is None:
        return
    ranges = wp.array(
        np.asarray([1.0, np.nan, 2.0, 20.0], dtype=np.float32),
        dtype=wp.float32,
        device="cpu",
    )
    current = wp.zeros(4, dtype=wp.vec3, device="cpu")
    history = wp.zeros(8, dtype=wp.vec3, device="cpu")
    history_count = wp.zeros(1, dtype=wp.int32, device="cpu")

    wp.launch(
        warp_points._laser_scan_interop_kernel,
        dim=4,
        inputs=[ranges, 0.0, math.pi / 2.0, 0.1, 10.0, 0.0, 0.0, 0.0, 1, 1, 8],
        outputs=[current, history, history_count],
        device="cpu",
    )

    current_values = current.numpy()
    assert history_count.numpy()[0] == 2
    np.testing.assert_allclose(current_values[0], [1.0, 0.0, 0.0], atol=1.0e-6)
    assert current_values[1, 2] == -1000.0
    assert current_values[2, 0] == pytest.approx(-2.0, abs=1.0e-5)
    assert current_values[3, 2] == -1000.0


def test_native_slam_viewer_uses_registered_gl_buffers_with_fallback():
    source = Path(warp_points.__file__).read_text(encoding="utf-8")

    assert "RegisteredGLBuffer" in source
    assert "fallback_to_copy=False" in source
    assert "CUDA/OpenGL interop unavailable; using renderer fallback" in source


def test_static_viewer_node_uses_managed_runtime(monkeypatch):
    captured = {}

    def fake_start(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "running": True, "viewer_id": "static"}

    monkeypatch.setattr(warp_points.viewer_rt, "start_viewer", fake_start)
    result = _NODE_REGISTRY["WarpLiDARViewer"]({
        "action": "start",
        "viewer_id": "static",
        "laser_scan": _scan(),
        "device": "cuda:0",
        "animate_scan": True,
        "scan_hz": 0.4,
        "show_rays": True,
        "ray_trail_count": 128,
        "accumulate_hits": False,
        "compare_numpy": True,
    })

    assert result["running"] is True
    assert captured["viewer_id"] == "static"
    assert captured["scan"]["kind"] == "blacknode.laser-scan-stream"
    assert captured["options"]["animate_scan"] is True
    assert captured["options"]["scan_hz"] == 0.4
    assert captured["options"]["show_rays"] is True
    assert captured["options"]["ray_trail_count"] == 128
    assert captured["options"]["accumulate_hits"] is False
    assert captured["options"]["compare_numpy"] is True
    assert result["viewer"]["controls"]["p"] == "pause or resume scan sweep"


def test_static_viewer_rejects_cpu_before_launch(monkeypatch):
    launched = False

    def fake_popen(*_args, **_kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("CPU viewer must not launch an unsafe OpenGL worker")

    monkeypatch.setattr(warp_viewer_runtime.subprocess, "Popen", fake_popen)
    result = warp_viewer_runtime.start_viewer(
        viewer_id="cpu_viewer",
        scan=_scan(),
        options={"device": "cpu"},
    )

    assert result["ok"] is False
    assert "requires a CUDA device" in result["error"]
    assert launched is False


def test_dense_scan_viewer_uses_file_handoff_instead_of_command_payload(monkeypatch):
    captured = {}

    class FakeProcess:
        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            del timeout
            return 0

    def fake_popen(arguments, **kwargs):
        captured["arguments"] = arguments
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(warp_viewer_runtime.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(warp_viewer_runtime.time, "sleep", lambda _seconds: None)
    scan = _scan()
    scan["ranges"] = [2.0] * 4096
    result = warp_viewer_runtime.start_viewer(
        viewer_id="dense_handoff",
        scan=scan,
        options={"device": "cuda:0", "animate_scan": True, "show_rays": True},
    )

    try:
        arguments = captured["arguments"]
        assert result["ok"] is True
        assert "--scan-file" in arguments
        assert "--scan-base64" not in arguments
        assert "--persist-scans" in arguments
        assert "--max-accumulated-points" in arguments
        assert arguments[arguments.index("--robot-length") + 1] == "0.25"
        assert arguments[arguments.index("--robot-width") + 1] == "0.22"
        scan_path = Path(arguments[arguments.index("--scan-file") + 1])
        assert json.loads(scan_path.read_text(encoding="utf-8"))["ranges"] == scan["ranges"]
        assert max(len(str(value)) for value in arguments) < 1024
    finally:
        warp_viewer_runtime.stop_viewer("dense_handoff")
