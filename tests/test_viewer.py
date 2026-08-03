"""Generic live Viewer node and managed Warp session tests."""
from __future__ import annotations

import blacknode  # noqa: F401
from blacknode.node import _NODE_REGISTRY
from blacknode.pkg.blacknode_cuda import viewer_runtime
from blacknode.pkg.blacknode_cuda import warp_points


def _source():
    return {
        "kind": "blacknode.message-stream",
        "schema_version": 1,
        "stream_id": "topic-subscriber:/scan",
        "protocol": "ros2",
        "topic": "/scan",
        "message_type": "sensor_msgs/msg/LaserScan",
    }


def _outputs(received=4, *, fresh=True):
    return {
        "running": True,
        "message": {
            "header": {"stamp": {"sec": received, "nanosec": 5}, "frame_id": "laser"},
            "angle_min": 0.0,
            "angle_max": 1.0,
            "angle_increment": 0.5,
            "range_min": 0.1,
            "range_max": 10.0,
            "ranges": [1.0, 2.0, None],
        },
        "status": {
            "state": "ready" if fresh else "stale",
            "source_fresh": fresh,
            "received": received,
            "last_message_time_ns": received * 1_000_000_000 + 5,
            "error": "",
        },
        "received": received,
    }


def _processed(scan, **kwargs):
    assert scan["kind"] == "blacknode.laser-scan-stream"
    assert scan["frame"] == "laser"
    return {
        "ok": True,
        "filtered_points": [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        "colors": [[0.0, 1.0, 1.0], [0.0, 0.5, 1.0]],
        "device": kwargs["device"],
        "kernel_ms": 0.25,
        "report": {"state": "ready"},
    }


def test_viewer_registers_as_generic_live_spatial_node():
    fn = _NODE_REGISTRY["Viewer"]

    assert fn._bn_package == "blacknode-cuda"
    assert fn._bn_component == "spatial-processing"
    assert fn._bn_live_capable is True
    assert fn._bn_input_types["source"] == "Dict"
    assert fn._bn_input_types["accumulate_hits"] == "Bool"
    assert "clear" in fn._bn_input_choices["action"]
    assert fn._bn_output_types["scene"] == "Dict"
    assert _NODE_REGISTRY["WarpLiDARViewer"]._bn_hidden is True
    assert _NODE_REGISTRY["WarpSLAMDiscoveryViewer"]._bn_hidden is True


def test_editor_viewer_normalizes_stream_and_publishes_live_scene(monkeypatch):
    viewer_runtime.stop_viewer()
    monkeypatch.setattr(warp_points, "process_laser_scan", _processed)

    result = _NODE_REGISTRY["Viewer"]({
        "action": "start",
        "source": _source(),
        "viewer_id": "scan",
        "mode": "editor",
        "device": "cpu",
        "__node_id__": "viewer-node",
        "__message_stream_reader__": lambda _source_value: _outputs(),
    })
    runtime = viewer_runtime.runtime_status()

    assert result["running"] is True
    assert result["live"] is True
    assert result["scene"]["kind"] == "blacknode.viewer-scene"
    assert result["scene"]["primitive"] == "point-cloud"
    assert result["scene"]["point_count"] == 2
    assert result["scene"]["sensor"] == {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}
    assert result["scene"]["scan"]["angle_max_rad"] == 1.0
    assert result["scene"]["view"] == {"radius_m": 12.0, "units": "meters"}
    assert result["scene"]["animation"]["show_rays"] is True
    assert result["scene"]["current_point_count"] == 2
    assert result["scene"]["accumulated_scan_count"] == 1
    assert result["status"]["kernel_ms"] == 0.25
    assert runtime["node_outputs"][0]["node_id"] == "viewer-node"
    assert runtime["node_outputs"][0]["outputs"]["scene"]["sequence"] == 4

    viewer_runtime.stop_viewer()


def test_viewer_accumulates_real_scans_and_clears_history(monkeypatch):
    viewer_runtime.stop_viewer()
    monkeypatch.setattr(warp_points, "process_laser_scan", _processed)
    state = {"received": 1}

    def reader(_source_value):
        return _outputs(state["received"])

    _NODE_REGISTRY["Viewer"]({
        "action": "start",
        "source": _source(),
        "viewer_id": "history",
        "mode": "editor",
        "device": "cpu",
        "accumulate_hits": True,
        "max_accumulated_points": 1000,
        "__message_stream_reader__": reader,
    })
    state["received"] = 2
    accumulated = viewer_runtime.viewer_status("history")
    cleared = _NODE_REGISTRY["Viewer"]({"action": "clear", "viewer_id": "history"})

    assert accumulated["scene"]["point_count"] == 4
    assert accumulated["scene"]["accumulated_scan_count"] == 2
    assert cleared["scene"]["point_count"] == 2
    assert cleared["scene"]["accumulated_scan_count"] == 1
    assert cleared["report"] == "Viewer scan history cleared"
    viewer_runtime.stop_viewer()


def test_device_viewer_updates_native_worker_from_new_scans(monkeypatch):
    viewer_runtime.stop_viewer()
    monkeypatch.setattr(warp_points, "process_laser_scan", _processed)
    calls = []
    state = {"received": 1}

    monkeypatch.setattr(
        viewer_runtime.warp_viewer_runtime,
        "start_viewer",
        lambda **kwargs: calls.append(("start", kwargs["scan"]["source_time_ns"]))
        or {"ok": True, "running": True},
    )
    monkeypatch.setattr(
        viewer_runtime.warp_viewer_runtime,
        "update_viewer_scan",
        lambda _viewer_id, scan: calls.append(("update", scan["source_time_ns"]))
        or {"ok": True},
    )
    monkeypatch.setattr(
        viewer_runtime.warp_viewer_runtime,
        "stop_viewer",
        lambda viewer_id="": calls.append(("stop", viewer_id)) or {"ok": True, "stopped": 1},
    )

    def reader(_source_value):
        return _outputs(state["received"])

    started = _NODE_REGISTRY["Viewer"]({
        "action": "start",
        "source": _source(),
        "viewer_id": "device-scan",
        "mode": "device",
        "device": "cuda:0",
        "__node_id__": "device-viewer",
        "__message_stream_reader__": reader,
    })
    state["received"] = 2
    status = viewer_runtime.viewer_status("device-scan")
    stopped = _NODE_REGISTRY["Viewer"]({"action": "stop", "viewer_id": "device-scan"})

    assert started["live"] is True
    assert status["scene"]["sequence"] == 2
    assert [item[0] for item in calls] == ["start", "update", "stop"]
    assert stopped["running"] is False


def test_viewer_does_not_present_stale_scan_as_live(monkeypatch):
    viewer_runtime.stop_viewer()
    monkeypatch.setattr(warp_points, "process_laser_scan", _processed)

    result = _NODE_REGISTRY["Viewer"]({
        "action": "start",
        "source": _source(),
        "viewer_id": "stale",
        "mode": "editor",
        "device": "cpu",
        "__message_stream_reader__": lambda _source_value: _outputs(fresh=False),
    })

    assert result["running"] is True
    assert result["live"] is False
    assert result["status"]["state"] == "stale"
    viewer_runtime.stop_viewer()
