"""Managed live point-cloud sessions for the generic Viewer node."""
from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable

from . import warp_viewer_runtime


_SESSIONS: dict[str, dict[str, Any]] = {}
_LOCK = threading.RLock()
_MAX_EDITOR_POINTS = 20_000


def _safe_id(value: str) -> str:
    return "".join(
        character
        for character in str(value or "")
        if character.isalnum() or character in "_-"
    )[:80]


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _local_stream_reader(source: dict[str, Any]) -> dict[str, Any]:
    if str(source.get("protocol") or "") != "ros2":
        return {"status": {"state": "unavailable", "error": "unsupported message-stream protocol"}}
    try:
        from blacknode.pkg.blacknode_ros2 import ros2_runtime
    except Exception as exc:
        return {
            "status": {
                "state": "unavailable",
                "error": f"blacknode-ros2 is unavailable ({type(exc).__name__}: {exc})",
            }
        }
    topic = str(source.get("topic") or "").strip()
    status = ros2_runtime.topic_subscriber_status(topic)
    return ros2_runtime.ros2_topic_outputs(status)


def _normalize_laser_scan(outputs: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    envelope = outputs.get("message") if isinstance(outputs.get("message"), dict) else {}
    message = envelope.get("message") if isinstance(envelope.get("message"), dict) else envelope
    ranges = message.get("ranges") if isinstance(message.get("ranges"), list) else []
    if not ranges:
        return {}
    header = message.get("header") if isinstance(message.get("header"), dict) else {}
    stamp = header.get("stamp") if isinstance(header.get("stamp"), dict) else {}
    source_time_ns = (
        int(_finite(stamp.get("sec"), 0.0)) * 1_000_000_000
        + int(_finite(stamp.get("nanosec"), 0.0))
    )
    clean_ranges = [
        _finite(value, float("nan"))
        for value in ranges[:100_000]
    ]
    status = outputs.get("status") if isinstance(outputs.get("status"), dict) else {}
    return {
        "kind": "blacknode.laser-scan-stream",
        "schema_version": 1,
        "frame": str(header.get("frame_id") or "laser").strip() or "laser",
        "topic": str(source.get("topic") or ""),
        "message_type": str(source.get("message_type") or "sensor_msgs/msg/LaserScan"),
        "source_time_ns": source_time_ns or int(status.get("last_message_time_ns") or 0),
        "receive_time_ns": time.time_ns(),
        "angle_min": _finite(message.get("angle_min"), -math.pi),
        "angle_max": _finite(message.get("angle_max"), math.pi),
        "angle_increment": _finite(message.get("angle_increment"), 0.0),
        "range_min": max(0.0, _finite(message.get("range_min"), 0.0)),
        "range_max": max(0.0, _finite(message.get("range_max"), 0.0)),
        "scan_time": max(0.0, _finite(message.get("scan_time"), 0.0)),
        "time_increment": max(0.0, _finite(message.get("time_increment"), 0.0)),
        "ranges": clean_ranges,
        "intensities": list(message.get("intensities") or [])[:100_000],
    }


def _viewer_outputs(session: dict[str, Any]) -> dict[str, Any]:
    status = dict(session.get("status") or {})
    return {
        "running": bool(session.get("running")),
        "live": bool(session.get("live")),
        "scene": dict(session.get("scene") or {}),
        "status": status,
        "viewer": {
            "kind": "blacknode.viewer",
            "schema_version": 1,
            "viewer_id": session["viewer_id"],
            "mode": session["mode"],
            "processor": "warp",
            "device": session["device"],
            "state": status.get("state", "waiting"),
        },
        "report": str(session.get("report") or "Viewer is waiting for source data"),
    }


def _scene_from_processed(
    processed: dict[str, Any],
    scan: dict[str, Any],
    source_outputs: dict[str, Any],
) -> dict[str, Any]:
    points = list(processed.get("filtered_points") or [])
    colors = list(processed.get("colors") or [])
    display_stride = max(1, math.ceil(len(points) / _MAX_EDITOR_POINTS))
    display_points = points[::display_stride]
    display_colors = colors[::display_stride]
    return {
        "kind": "blacknode.viewer-scene",
        "schema_version": 1,
        "primitive": "point-cloud",
        "projection": "xy",
        "frame": str(scan.get("frame") or "laser"),
        "source_time_ns": int(scan.get("source_time_ns") or 0),
        "receive_time_ns": int(scan.get("receive_time_ns") or 0),
        "sequence": int(source_outputs.get("received") or 0),
        "points": display_points,
        "colors": display_colors,
        "point_count": len(points),
        "display_count": len(display_points),
        "display_stride": display_stride,
        "device": str(processed.get("device") or ""),
        "kernel_ms": float(processed.get("kernel_ms") or 0.0),
    }


def _update_session(session: dict[str, Any]) -> None:
    reader = session.get("source_reader")
    if not callable(reader):
        reader = _local_stream_reader
    try:
        source_outputs = reader(dict(session["source"]))
    except Exception as exc:
        source_outputs = {
            "status": {
                "state": "unavailable",
                "error": f"message stream read failed ({type(exc).__name__}: {exc})",
            }
        }
    if not isinstance(source_outputs, dict):
        source_outputs = {"status": {"state": "unavailable", "error": "message stream returned invalid data"}}
    source_status = (
        source_outputs.get("status")
        if isinstance(source_outputs.get("status"), dict)
        else {}
    )
    source_fresh = bool(source_status.get("source_fresh"))
    scan = _normalize_laser_scan(source_outputs, session["source"])
    if not source_fresh or not scan:
        source_state = str(source_status.get("state") or "waiting")
        error = str(source_status.get("error") or "").strip()
        session.update(
            live=False,
            status={
                "kind": "blacknode.viewer-status",
                "schema_version": 1,
                "state": "stale" if source_state == "stale" else "waiting",
                "source_fresh": False,
                "received": int(source_outputs.get("received") or 0),
                "error": error,
            },
            report=error or "Viewer is waiting for a fresh LaserScan message",
        )
        return

    marker = (
        int(scan.get("source_time_ns") or 0),
        int(source_outputs.get("received") or 0),
    )
    if marker == session.get("source_marker") and session.get("scene"):
        status = dict(session.get("status") or {})
        if status.get("state") == "error":
            status.update(source_fresh=True, received=marker[1])
            session.update(live=False, status=status)
            return
        status.update(
            state="ready",
            source_fresh=True,
            received=marker[1],
            error="",
        )
        session.update(live=True, status=status)
        return

    from .warp_points import process_laser_scan

    options = session["options"]
    processed = process_laser_scan(
        scan,
        device=session["device"],
        filter_min_m=options["filter_min_m"],
        filter_max_m=options["filter_max_m"],
        stride=options["stride"],
        sensor_pose=(
            options["sensor_x_m"],
            options["sensor_y_m"],
            options["sensor_yaw_rad"],
        ),
        include_raw_points=False,
        compare_numpy=False,
    )
    if not processed.get("ok"):
        error = str(processed.get("report", {}).get("error") or "Warp processing failed")
        session.update(
            live=False,
            status={
                "kind": "blacknode.viewer-status",
                "schema_version": 1,
                "state": "error",
                "source_fresh": True,
                "received": int(source_outputs.get("received") or 0),
                "error": error,
            },
            report=error,
        )
        return

    scene = _scene_from_processed(processed, scan, source_outputs)
    if session["mode"] == "device":
        native = session.get("native") if isinstance(session.get("native"), dict) else {}
        if not native.get("running"):
            native = warp_viewer_runtime.start_viewer(
                viewer_id=session["viewer_id"],
                scan=scan,
                options={**options, "device": session["device"], "live": True},
            )
            session["native"] = native
        else:
            native_update = warp_viewer_runtime.update_viewer_scan(session["viewer_id"], scan)
            if not native_update.get("ok"):
                native = native_update
        if not native.get("ok", True):
            error = str(native.get("error") or "native viewer failed")
            session.update(
                live=False,
                scene=scene,
                source_marker=marker,
                status={
                    "kind": "blacknode.viewer-status",
                    "schema_version": 1,
                    "state": "error",
                    "source_fresh": True,
                    "received": int(source_outputs.get("received") or 0),
                    "error": error,
                },
                report=error,
            )
            return
    session.update(
        live=True,
        scene=scene,
        source_time_ns=int(scan.get("source_time_ns") or 0),
        source_marker=marker,
        status={
            "kind": "blacknode.viewer-status",
            "schema_version": 1,
            "state": "ready",
            "source_fresh": True,
            "received": int(source_outputs.get("received") or 0),
            "point_count": int(scene.get("point_count") or 0),
            "kernel_ms": float(scene.get("kernel_ms") or 0.0),
            "error": "",
        },
        report=(
            f"Viewer {session['mode']} rendered {int(scene.get('point_count') or 0):,} "
            f"points with Warp in {float(scene.get('kernel_ms') or 0.0):.3f} ms"
        ),
    )


def _device_worker(viewer_id: str, stop_event: threading.Event, interval: float) -> None:
    while not stop_event.wait(interval):
        with _LOCK:
            session = _SESSIONS.get(viewer_id)
            if session is None or session.get("stop_event") is not stop_event:
                return
            _update_session(session)


def start_viewer(
    *,
    viewer_id: str,
    node_id: str,
    source: dict[str, Any],
    mode: str,
    device: str,
    options: dict[str, Any],
    source_reader: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    clean_id = _safe_id(viewer_id)
    if not clean_id:
        return {
            "running": False,
            "live": False,
            "scene": {},
            "status": {"state": "error", "error": "viewer_id is required"},
            "viewer": {},
            "report": "viewer_id is required",
        }
    if source.get("kind") != "blacknode.message-stream":
        return {
            "running": False,
            "live": False,
            "scene": {},
            "status": {"state": "error", "error": "source must be a blacknode.message-stream"},
            "viewer": {},
            "report": "Connect ROS2.stream to Viewer.source",
        }
    selected_mode = str(mode or "editor").strip().lower()
    if selected_mode not in {"editor", "device"}:
        selected_mode = "editor"
    with _LOCK:
        previous = _SESSIONS.pop(clean_id, None)
        if previous:
            previous_event = previous.get("stop_event")
            if isinstance(previous_event, threading.Event):
                previous_event.set()
            if previous.get("mode") == "device":
                warp_viewer_runtime.stop_viewer(clean_id)
        stop_event = threading.Event()
        session = {
            "viewer_id": clean_id,
            "node_id": str(node_id or ""),
            "source": dict(source),
            "source_reader": source_reader,
            "mode": selected_mode,
            "device": str(device or "cuda:0"),
            "options": dict(options),
            "running": True,
            "live": False,
            "scene": {},
            "status": {
                "kind": "blacknode.viewer-status",
                "schema_version": 1,
                "state": "waiting",
                "source_fresh": False,
                "received": 0,
                "error": "",
            },
            "report": "Viewer started; waiting for a fresh LaserScan message",
            "stop_event": stop_event,
        }
        _SESSIONS[clean_id] = session
        _update_session(session)
        if selected_mode == "device" and session.get("running"):
            interval = max(
                0.05,
                1.0 / max(1.0, min(120.0, _finite(options.get("fps"), 30.0))),
            )
            worker = threading.Thread(
                target=_device_worker,
                args=(clean_id, stop_event, interval),
                name=f"blacknode-viewer-{clean_id}",
                daemon=True,
            )
            session["worker"] = worker
            worker.start()
        return _viewer_outputs(session)


def viewer_status(viewer_id: str) -> dict[str, Any]:
    clean_id = _safe_id(viewer_id)
    with _LOCK:
        session = _SESSIONS.get(clean_id)
        if session is None:
            return {
                "running": False,
                "live": False,
                "scene": {},
                "status": {"kind": "blacknode.viewer-status", "schema_version": 1, "state": "stopped"},
                "viewer": {"viewer_id": clean_id, "state": "stopped"},
                "report": "Viewer is stopped",
            }
        _update_session(session)
        return _viewer_outputs(session)


def stop_viewer(viewer_id: str = "") -> dict[str, Any]:
    clean_id = _safe_id(viewer_id)
    with _LOCK:
        ids = [clean_id] if clean_id else list(_SESSIONS)
        stopped = 0
        for session_id in ids:
            session = _SESSIONS.pop(session_id, None)
            if session is None:
                continue
            stop_event = session.get("stop_event")
            if isinstance(stop_event, threading.Event):
                stop_event.set()
            if session.get("mode") == "device":
                warp_viewer_runtime.stop_viewer(session_id)
            stopped += 1
    return {"ok": True, "stopped": stopped}


def runtime_status() -> dict[str, Any]:
    node_outputs: list[dict[str, Any]] = []
    with _LOCK:
        for viewer_id, session in list(_SESSIONS.items()):
            _update_session(session)
            node_outputs.append({
                "node_type": "Viewer",
                "node_id": session.get("node_id", ""),
                "run_id": viewer_id,
                "outputs": _viewer_outputs(session),
            })
    return {
        "ok": all(item["outputs"].get("status", {}).get("state") != "error" for item in node_outputs),
        "active": bool(node_outputs),
        "streams": [],
        "managed_runs": [],
        "node_outputs": node_outputs,
        "detached_count": 0,
        "report": f"{len(node_outputs)} Viewer session(s) active" if node_outputs else "no Viewer sessions active",
    }


def stop_runtime_services() -> dict[str, Any]:
    stopped = int(stop_viewer().get("stopped") or 0)
    return {
        "ok": True,
        "stopped": {"streams": stopped, "managed_runs": 0, "detached": 0},
        "report": f"stopped {stopped} Viewer session(s)",
    }
