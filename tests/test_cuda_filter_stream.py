"""CUDAImageFilterStream — node contract.

The no-source/no-GPU contract (structured error, never raises) is always
exercised. Real subprocess/GPU behavior is covered manually (see README) --
these tests monkeypatch cuda_stream_runtime so they run fast and portable.
"""
import blacknode  # noqa: F401  triggers package discovery
from blacknode.node import _NODE_REGISTRY
from blacknode.pkg.blacknode_cuda import cuda_stream_runtime as stream_rt


def test_registered_with_category():
    assert "CUDAImageFilterStream" in _NODE_REGISTRY
    assert _NODE_REGISTRY["CUDAImageFilterStream"]._bn_category == "NVIDIA GPU"
    assert _NODE_REGISTRY["CUDAImageFilterStream"]._bn_package == "blacknode-cuda"


def test_start_requires_source_url():
    result = _NODE_REGISTRY["CUDAImageFilterStream"]({"action": "start", "source_url": ""})
    assert result["streaming"] is False
    assert result["stream_url"] == ""
    assert "source_url is required" in result["report"]


def test_start_rejects_unknown_filter():
    result = _NODE_REGISTRY["CUDAImageFilterStream"]({
        "action": "start", "source_url": "http://127.0.0.1:9000/snapshot.jpg", "op": "not_a_real_filter",
    })
    assert result["streaming"] is False
    assert "unknown filter" in result["report"]


def test_start_calls_runtime_with_resolved_params(monkeypatch):
    calls = {}

    def fake_start(**kwargs):
        calls.update(kwargs)
        return {
            "ok": True,
            "stream_url": "http://127.0.0.1:9020/stream.mjpg",
            "snapshot_url": "http://127.0.0.1:9020/snapshot.jpg",
            "health_url": "http://127.0.0.1:9020/health.json",
            "port": 9020,
        }

    monkeypatch.setattr(stream_rt, "start_filter_stream", fake_start)
    result = _NODE_REGISTRY["CUDAImageFilterStream"]({
        "action": "start",
        "stream_id": "cf1",
        "source_url": "http://127.0.0.1:58793/snapshot.jpg",
        "op": "sobel_edges",
        "amount": 2.0,
        "max_fps": 15.0,
        "max_width": 640,
    })
    assert result["preview"] == "http://127.0.0.1:9020/stream.mjpg"
    assert result["streaming"] is True
    assert result["stream_url"] == result["preview"]
    assert result["snapshot_url"] == "http://127.0.0.1:9020/snapshot.jpg"
    assert calls["stream_id"] == "cf1"
    assert calls["source_url"] == "http://127.0.0.1:58793/snapshot.jpg"
    assert calls["op"] == "sobel_edges"
    assert calls["amount"] == 2.0
    assert calls["max_fps"] == 15.0
    assert calls["max_width"] == 640


def test_start_surfaces_runtime_failure(monkeypatch):
    monkeypatch.setattr(stream_rt, "start_filter_stream", lambda **kwargs: {"ok": False, "error": "no CUDA device"})
    result = _NODE_REGISTRY["CUDAImageFilterStream"]({
        "action": "start", "source_url": "http://127.0.0.1:58793/snapshot.jpg",
    })
    assert result["streaming"] is False
    assert "no CUDA device" in result["report"]


def test_stop_calls_runtime():
    captured = {}

    def fake_stop(stream_id=""):
        captured["stream_id"] = stream_id
        return {"ok": True, "stopped": 1}

    import unittest.mock
    with unittest.mock.patch.object(stream_rt, "stop_filter_stream", fake_stop):
        result = _NODE_REGISTRY["CUDAImageFilterStream"]({"action": "stop", "stream_id": "cf1"})
    assert captured["stream_id"] == "cf1"
    assert result["streaming"] is False
    assert result["preview"] == ""
    assert "stopped 1" in result["report"]


def test_apply_image_filter_math_matches_node():
    """The standalone stream script duplicates the node's filter math on
    purpose (subprocess helpers are self-contained) -- pin that the two
    stay numerically identical for a couple of ops."""
    import numpy as np
    import sys
    from pathlib import Path

    script_dir = Path(stream_rt.__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(script_dir))
    try:
        import cuda_filter_stream_server as server_mod
    finally:
        sys.path.remove(str(script_dir))

    g = np.random.default_rng(0).random((4, 4, 3)).astype(np.float32)
    from blacknode.pkg.blacknode_cuda.cuda import _apply_image_filter as node_filter

    for op in ["grayscale", "invert", "brighten", "gaussian_blur"]:
        node_out = node_filter(np, op, g, 1.5)
        server_out = server_mod.apply_image_filter(np, op, g, 1.5)
        assert np.allclose(node_out, server_out), op
