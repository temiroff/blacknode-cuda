"""Warp LaserScan processing and managed viewer contracts."""
import math
import json
from pathlib import Path

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
        scan_path = Path(arguments[arguments.index("--scan-file") + 1])
        assert json.loads(scan_path.read_text(encoding="utf-8"))["ranges"] == scan["ranges"]
        assert max(len(str(value)) for value in arguments) < 1024
    finally:
        warp_viewer_runtime.stop_viewer("dense_handoff")
