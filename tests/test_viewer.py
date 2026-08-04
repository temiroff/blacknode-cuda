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


def _pose_source():
    return {
        "kind": "blacknode.message-stream",
        "schema_version": 1,
        "stream_id": "topic-subscriber:/odom",
        "protocol": "ros2",
        "topic": "/odom",
        "message_type": "nav_msgs/msg/Odometry",
    }


def _tf_source():
    return {
        "kind": "blacknode.message-stream",
        "schema_version": 1,
        "stream_id": "topic-subscriber:/tf",
        "protocol": "ros2",
        "topic": "/tf",
        "message_type": "tf2_msgs/msg/TFMessage",
    }


def _tf_static_source():
    return {**_tf_source(), "stream_id": "topic-subscriber:/tf_static", "topic": "/tf_static"}


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


def _pose_outputs(received=4, *, fresh=True, x=2.0, y=3.0):
    message = {
        "header": {"stamp": {"sec": received, "nanosec": 5}, "frame_id": "odom"},
        "child_frame_id": "base_link",
        "pose": {
            "pose": {
                "position": {"x": x, "y": y, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }
        },
    }
    return {
        "running": True,
        "message": message,
        "messages": [message],
        "status": {
            "state": "ready" if fresh else "stale",
            "source_fresh": fresh,
            "received": received,
            "error": "",
        },
        "received": received,
    }


def _tf_outputs(received=4):
    def transform(parent, child, x, y, z, w):
        return {
            "header": {"stamp": {"sec": received, "nanosec": 5}, "frame_id": parent},
            "child_frame_id": child,
            "transform": {
                "translation": {"x": x, "y": y, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": z, "w": w},
            },
        }

    message = {"transforms": [
        transform("map", "camera", 99.0, 99.0, 0.0, 1.0),
        transform("odom", "base_link", 1.0, 2.0, 2 ** -0.5, 2 ** -0.5),
    ]}
    return {
        "running": True,
        "message": message,
        "messages": [message],
        "status": {"state": "ready", "source_fresh": True, "received": received, "error": ""},
        "received": received,
    }


def _tf_static_outputs():
    message = {"transforms": [{
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "base_link"},
        "child_frame_id": "laser",
        "transform": {
            "translation": {"x": 0.2, "y": 0.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 2 ** -0.5, "w": 2 ** -0.5},
        },
    }]}
    return {
        "message": message,
        "messages": [message],
        "status": {"state": "stale", "source_fresh": False, "received": 1, "error": ""},
        "received": 1,
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
    assert fn._bn_input_types["pose"] == "Dict"
    assert fn._bn_input_types["pose_parent_frame"] == "Text"
    assert fn._bn_input_types["pose_child_frame"] == "Text"
    assert fn._bn_input_types["accumulate_hits"] == "Bool"
    assert "clear" in fn._bn_input_choices["action"]
    assert "pause" in fn._bn_input_choices["action"]
    assert "resume" in fn._bn_input_choices["action"]
    assert fn._bn_output_types["scene"] == "Dict"
    assert _NODE_REGISTRY["WarpLiDARViewer"]._bn_hidden is True
    assert _NODE_REGISTRY["WarpSLAMDiscoveryViewer"]._bn_hidden is True


def test_editor_viewer_normalizes_stream_and_publishes_live_scene(monkeypatch):
    viewer_runtime.stop_viewer()


def test_viewer_registers_scan_history_with_generic_pose_stream(monkeypatch):
    viewer_runtime.stop_viewer()
    captured = {}

    def process(scan, **kwargs):
        captured["sensor_pose"] = kwargs["sensor_pose"]
        return _processed(scan, **kwargs)

    def reader(source):
        return _pose_outputs() if source.get("topic") == "/odom" else _outputs()

    monkeypatch.setattr(warp_points, "process_laser_scan", process)
    result = _NODE_REGISTRY["Viewer"]({
        "action": "start",
        "source": _source(),
        "pose": _pose_source(),
        "viewer_id": "registered",
        "mode": "editor",
        "device": "cpu",
        "sensor_x_m": 0.25,
        "__message_stream_reader__": reader,
    })

    assert result["live"] is True
    assert captured["sensor_pose"] == (2.25, 3.0, 0.0)
    assert result["scene"]["frame"] == "odom"
    assert result["scene"]["source_frame"] == "laser"
    assert result["scene"]["history_registered"] is True
    assert result["scene"]["pose_source"] == "nav_msgs/msg/Odometry"
    assert result["scene"]["registration"]["child_frame"] == "base_link"
    assert result["status"]["pose_fresh"] is True
    viewer_runtime.stop_viewer()


def test_viewer_waits_instead_of_accumulating_with_stale_pose(monkeypatch):
    viewer_runtime.stop_viewer()


def test_viewer_resolves_multihop_tf_to_scan_frame_and_uses_tf_forward(monkeypatch):
    viewer_runtime.stop_viewer()
    captured = {}

    def process(scan, **kwargs):
        captured["sensor_pose"] = kwargs["sensor_pose"]
        return _processed(scan, **kwargs)

    def reader(source):
        if source.get("topic") == "/tf":
            return _tf_outputs()
        if source.get("topic") == "/tf_static":
            return _tf_static_outputs()
        return _outputs()

    monkeypatch.setattr(warp_points, "process_laser_scan", process)
    result = _NODE_REGISTRY["Viewer"]({
        "action": "start",
        "source": _source(),
        "pose": _tf_source(),
        "pose_static": _tf_static_source(),
        "pose_parent_frame": "odom",
        "pose_child_frame": "auto",
        "sensor_x_m": 9.0,
        "viewer_id": "tf-registered",
        "mode": "editor",
        "device": "cpu",
        "__message_stream_reader__": reader,
    })

    sensor_x, sensor_y, sensor_yaw = captured["sensor_pose"]
    assert abs(sensor_x - 1.0) < 1e-6
    assert abs(sensor_y - 2.2) < 1e-6
    assert abs(abs(sensor_yaw) - 3.141592653589793) < 1e-6
    assert result["scene"]["registration"]["tf_path"] == ["odom", "base_link", "laser"]
    assert result["scene"]["registration"]["child_frame"] == "laser"
    viewer_runtime.stop_viewer()
    called = False

    def process(scan, **kwargs):
        del scan, kwargs
        nonlocal called
        called = True
        return {}

    def reader(source):
        return _pose_outputs(fresh=False) if source.get("topic") == "/odom" else _outputs()

    monkeypatch.setattr(warp_points, "process_laser_scan", process)
    result = _NODE_REGISTRY["Viewer"]({
        "action": "start",
        "source": _source(),
        "pose": _pose_source(),
        "viewer_id": "stale-pose",
        "mode": "editor",
        "device": "cpu",
        "__message_stream_reader__": reader,
    })

    assert result["live"] is False
    assert result["status"]["pose_connected"] is True
    assert result["status"]["pose_fresh"] is False
    assert called is False
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
    cached = viewer_runtime.viewer_status("history")
    state["received"] = 3
    still_paused = viewer_runtime.viewer_status("history")
    resumed = _NODE_REGISTRY["Viewer"]({"action": "resume", "viewer_id": "history"})
    state["received"] = 4
    rebuilding = viewer_runtime.viewer_status("history")

    assert accumulated["scene"]["point_count"] == 4
    assert accumulated["scene"]["accumulated_scan_count"] == 2
    assert cleared["scene"]["point_count"] == 0
    assert cleared["scene"]["current_point_count"] == 2
    assert cleared["scene"]["accumulated_scan_count"] == 0
    assert cleared["scene"]["history_paused"] is True
    assert cached["scene"]["point_count"] == 0
    assert still_paused["scene"]["point_count"] == 2
    assert still_paused["scene"]["accumulated_scan_count"] == 1
    assert still_paused["scene"]["history_paused"] is True
    assert still_paused["scene"]["animation"]["accumulate_hits"] is False
    assert resumed["scene"]["history_paused"] is False
    assert rebuilding["scene"]["point_count"] == 4
    assert rebuilding["scene"]["accumulated_scan_count"] == 2
    assert cleared["report"] == "Viewer scan history cleared; accumulation is off"
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
