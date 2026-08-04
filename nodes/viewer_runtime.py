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
    header_time_ns = (
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
        "source_time_ns": header_time_ns or int(status.get("last_message_time_ns") or 0),
        "header_time_ns": header_time_ns,
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


def _message_payload(value: Any) -> dict[str, Any]:
    """Unwrap local and paired-device ROS2 message envelopes."""
    if not isinstance(value, dict):
        return {}
    nested = value.get("message")
    return nested if isinstance(nested, dict) else value


def _stamp_ns(value: dict[str, Any]) -> int:
    header = value.get("header") if isinstance(value.get("header"), dict) else {}
    stamp = header.get("stamp") if isinstance(header.get("stamp"), dict) else {}
    return (
        int(_finite(stamp.get("sec"), 0.0)) * 1_000_000_000
        + int(_finite(stamp.get("nanosec"), 0.0))
    )


def _quaternion_yaw(value: dict[str, Any]) -> float:
    x = _finite(value.get("x"), 0.0)
    y = _finite(value.get("y"), 0.0)
    z = _finite(value.get("z"), 0.0)
    w = _finite(value.get("w"), 1.0)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _frame_id(value: Any) -> str:
    return str(value or "").strip().lstrip("/")


def _compose_pose(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
    yaw = float(first.get("yaw_rad") or 0.0)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return {
        "x_m": float(first.get("x_m") or 0.0) + cosine * float(second.get("x_m") or 0.0) - sine * float(second.get("y_m") or 0.0),
        "y_m": float(first.get("y_m") or 0.0) + sine * float(second.get("x_m") or 0.0) + cosine * float(second.get("y_m") or 0.0),
        "z_m": float(first.get("z_m") or 0.0) + float(second.get("z_m") or 0.0),
        "yaw_rad": math.atan2(math.sin(yaw + float(second.get("yaw_rad") or 0.0)), math.cos(yaw + float(second.get("yaw_rad") or 0.0))),
    }


def _inverse_pose(value: dict[str, Any]) -> dict[str, float]:
    yaw = float(value.get("yaw_rad") or 0.0)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    x = float(value.get("x_m") or 0.0)
    y = float(value.get("y_m") or 0.0)
    return {
        "x_m": -cosine * x - sine * y,
        "y_m": sine * x - cosine * y,
        "z_m": -float(value.get("z_m") or 0.0),
        "yaw_rad": -yaw,
    }


def _pose_candidates(value: Any, source: dict[str, Any]) -> list[dict[str, Any]]:
    message = _message_payload(value)
    if not message:
        return []
    transforms = message.get("transforms")
    if isinstance(transforms, list):
        candidates: list[dict[str, Any]] = []
        for transform in transforms:
            candidates.extend(_pose_candidates(transform, source))
        return candidates

    pose_container = message.get("pose") if isinstance(message.get("pose"), dict) else {}
    pose = pose_container.get("pose") if isinstance(pose_container.get("pose"), dict) else pose_container
    transform = message.get("transform") if isinstance(message.get("transform"), dict) else {}
    position = pose.get("position") if isinstance(pose.get("position"), dict) else {}
    orientation = pose.get("orientation") if isinstance(pose.get("orientation"), dict) else {}
    if transform:
        position = transform.get("translation") if isinstance(transform.get("translation"), dict) else {}
        orientation = transform.get("rotation") if isinstance(transform.get("rotation"), dict) else {}
    if not position or not orientation:
        return []

    header = message.get("header") if isinstance(message.get("header"), dict) else {}
    message_type = str(source.get("message_type") or "").strip()
    return [{
        "x_m": _finite(position.get("x"), 0.0),
        "y_m": _finite(position.get("y"), 0.0),
        "z_m": _finite(position.get("z"), 0.0),
        "yaw_rad": _quaternion_yaw(orientation),
        "source_time_ns": _stamp_ns(message),
        "frame": _frame_id(header.get("frame_id") or "map") or "map",
        "child_frame": _frame_id(message.get("child_frame_id")),
        "message_type": message_type or ("tf2_msgs/msg/TFMessage" if transform else "geometry_msgs/msg/PoseStamped"),
        "is_transform": bool(transform),
    }]


def _tf_tree_pose(
    candidates: list[dict[str, Any]],
    parent_frame: str,
    child_frame: str,
    scan_frame: str,
    scan_time_ns: int,
    tolerance_seconds: float,
) -> tuple[dict[str, Any], str]:
    best_edges: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        parent = _frame_id(candidate.get("frame"))
        child = _frame_id(candidate.get("child_frame"))
        if not parent or not child or parent == child:
            continue
        stamp_ns = int(candidate.get("source_time_ns") or 0)
        delta = abs(stamp_ns - scan_time_ns) / 1_000_000_000.0 if stamp_ns and scan_time_ns else 0.0
        if stamp_ns and scan_time_ns and delta > tolerance_seconds:
            continue
        edge = {**candidate, "time_delta_seconds": delta}
        key = (parent, child)
        previous = best_edges.get(key)
        if previous is None or delta < float(previous.get("time_delta_seconds") or 0.0):
            best_edges[key] = edge
    if not best_edges:
        return {}, "TF stream has no transforms synchronized with the scan"

    adjacency: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for (parent, child), edge in best_edges.items():
        adjacency.setdefault(parent, []).append((child, edge))
        adjacency.setdefault(child, []).append((parent, {
            **edge,
            **_inverse_pose(edge),
            "frame": child,
            "child_frame": parent,
        }))

    queue: list[tuple[str, dict[str, float], list[str], list[dict[str, Any]]]] = [(
        parent_frame,
        {"x_m": 0.0, "y_m": 0.0, "z_m": 0.0, "yaw_rad": 0.0},
        [parent_frame],
        [],
    )]
    visited = {parent_frame}
    while queue:
        frame, transform, path, edges = queue.pop(0)
        if frame == child_frame:
            timed_edges = [edge for edge in edges if int(edge.get("source_time_ns") or 0) > 0]
            closest_edge = min(
                timed_edges,
                key=lambda edge: float(edge.get("time_delta_seconds") or 0.0),
            ) if timed_edges else {}
            return {
                **transform,
                "source_time_ns": int(closest_edge.get("source_time_ns") or 0),
                "time_delta_seconds": max(
                    (float(edge.get("time_delta_seconds") or 0.0) for edge in timed_edges),
                    default=None,
                ),
                "frame": parent_frame,
                "child_frame": child_frame,
                "message_type": "tf2_msgs/msg/TFMessage",
                "tf_path": path,
                "includes_sensor_extrinsics": child_frame == _frame_id(scan_frame),
            }, ""
        for next_frame, edge in adjacency.get(frame, []):
            if next_frame in visited:
                continue
            visited.add(next_frame)
            queue.append((
                next_frame,
                _compose_pose(transform, edge),
                [*path, next_frame],
                [*edges, edge],
            ))
    return {}, f"TF has no path from {parent_frame!r} to {child_frame!r}"


def _normalize_pose(
    outputs: dict[str, Any],
    source: dict[str, Any],
    scan_time_ns: int,
    tolerance_seconds: float,
    parent_frame: str,
    child_frame: str,
    scan_frame: str,
) -> tuple[dict[str, Any], str]:
    values = list(outputs.get("messages") or [])
    latest = outputs.get("message")
    if latest:
        values.append(latest)
    candidates = [candidate for value in values for candidate in _pose_candidates(value, source)]
    if not candidates:
        return {}, "Pose stream has no supported pose message"
    transform_candidates = [candidate for candidate in candidates if candidate.get("is_transform")]
    if transform_candidates:
        parent = _frame_id(parent_frame) or "odom"
        configured_child = _frame_id(child_frame)
        child = _frame_id(scan_frame) if not configured_child or configured_child == "auto" else configured_child
        if not child:
            return {}, "pose_child_frame=auto requires a frame_id in the scan message"
        return _tf_tree_pose(
            transform_candidates,
            parent,
            child,
            scan_frame,
            scan_time_ns,
            tolerance_seconds,
        )
    timed = [candidate for candidate in candidates if int(candidate.get("source_time_ns") or 0) > 0]
    if scan_time_ns > 0 and timed:
        pose = min(timed, key=lambda item: abs(int(item["source_time_ns"]) - scan_time_ns))
        delta_seconds = abs(int(pose["source_time_ns"]) - scan_time_ns) / 1_000_000_000.0
        if delta_seconds > tolerance_seconds:
            return {}, (
                f"Closest pose is {delta_seconds:.3f}s from the scan; "
                f"increase pose_sync_tolerance_s only if both topics use the same clock"
            )
    else:
        pose = candidates[-1]
        delta_seconds = None
    return {
        **pose,
        "time_delta_seconds": delta_seconds,
        "receive_time_ns": time.time_ns(),
        "tf_path": [],
        "includes_sensor_extrinsics": False,
    }, ""


def _combined_sensor_pose(options: dict[str, Any], pose: dict[str, Any]) -> tuple[float, float, float]:
    sensor_x = float(options.get("sensor_x_m") or 0.0)
    sensor_y = float(options.get("sensor_y_m") or 0.0)
    sensor_yaw = float(options.get("sensor_yaw_rad") or 0.0)
    if not pose:
        return sensor_x, sensor_y, sensor_yaw
    if pose.get("includes_sensor_extrinsics"):
        return (
            float(pose.get("x_m") or 0.0),
            float(pose.get("y_m") or 0.0),
            float(pose.get("yaw_rad") or 0.0),
        )
    robot_yaw = float(pose.get("yaw_rad") or 0.0)
    cosine = math.cos(robot_yaw)
    sine = math.sin(robot_yaw)
    return (
        float(pose.get("x_m") or 0.0) + cosine * sensor_x - sine * sensor_y,
        float(pose.get("y_m") or 0.0) + sine * sensor_x + cosine * sensor_y,
        robot_yaw + sensor_yaw,
    )


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
    options: dict[str, Any],
    history_points: list[Any],
    history_colors: list[Any],
    accumulated_scan_count: int,
    sensor_pose: tuple[float, float, float],
    pose: dict[str, Any],
) -> dict[str, Any]:
    current_points = list(processed.get("filtered_points") or [])
    current_colors = list(processed.get("colors") or [])
    display_stride = max(1, math.ceil(len(history_points) / _MAX_EDITOR_POINTS))
    current_stride = max(1, math.ceil(len(current_points) / _MAX_EDITOR_POINTS))
    display_points = history_points[::display_stride]
    display_colors = history_colors[::display_stride]
    return {
        "kind": "blacknode.viewer-scene",
        "schema_version": 1,
        "primitive": "point-cloud",
        "projection": "xy",
        "frame": str(pose.get("frame") or scan.get("frame") or "laser"),
        "source_frame": str(scan.get("frame") or "laser"),
        "source_time_ns": int(scan.get("source_time_ns") or 0),
        "receive_time_ns": int(scan.get("receive_time_ns") or 0),
        "sequence": int(source_outputs.get("received") or 0),
        "sensor": {
            "x_m": sensor_pose[0],
            "y_m": sensor_pose[1],
            "yaw_rad": sensor_pose[2],
        },
        "scan": {
            "angle_min_rad": float(scan.get("angle_min") or 0.0),
            "angle_max_rad": float(scan.get("angle_max") or 0.0),
            "angle_increment_rad": float(scan.get("angle_increment") or 0.0),
            "range_min_m": float(scan.get("range_min") or 0.0),
            "range_max_m": float(scan.get("range_max") or 0.0),
        },
        "view": {
            "radius_m": max(0.1, float(options.get("filter_max_m") or 12.0)),
            "units": "meters",
        },
        "points": display_points,
        "colors": display_colors,
        "current_points": current_points[::current_stride],
        "current_colors": current_colors[::current_stride],
        "point_count": len(history_points),
        "current_point_count": len(current_points),
        "accumulated_scan_count": accumulated_scan_count,
        "display_count": len(display_points),
        "display_stride": display_stride,
        "history_registered": bool(pose),
        "history_paused": bool(options.get("history_paused", False)),
        "pose_source": str(pose.get("message_type") or "sensor-local"),
        "registration": ({
            "method": "external-pose",
            "frame": str(pose.get("frame") or "map"),
            "child_frame": str(pose.get("child_frame") or ""),
            "x_m": float(pose.get("x_m") or 0.0),
            "y_m": float(pose.get("y_m") or 0.0),
            "z_m": float(pose.get("z_m") or 0.0),
            "yaw_rad": float(pose.get("yaw_rad") or 0.0),
            "source_time_ns": int(pose.get("source_time_ns") or 0),
            "time_delta_seconds": pose.get("time_delta_seconds"),
            "tf_path": list(pose.get("tf_path") or []),
        } if pose else {}),
        "animation": {
            "enabled": bool(options.get("animate_scan", True)),
            "show_rays": bool(options.get("show_rays", True)),
            "ray_trail_count": int(options.get("ray_trail_count") or 96),
            "pulse_hz": float(options.get("scan_hz") or 1.0),
            "sweep_direction": "counterclockwise",
            "accumulate_hits": bool(options.get("accumulate_hits", True))
            and not bool(options.get("history_paused", False)),
        },
        "device": str(processed.get("device") or ""),
        "kernel_ms": float(processed.get("kernel_ms") or 0.0),
    }


def _append_scan_history(
    session: dict[str, Any],
    processed: dict[str, Any],
) -> tuple[list[Any], list[Any], int]:
    points = list(processed.get("filtered_points") or [])
    colors = list(processed.get("colors") or [])
    if len(colors) < len(points):
        colors.extend([[0.0, 0.78, 1.0]] * (len(points) - len(colors)))
    options = session["options"]
    if session.get("history_paused"):
        # Keep the registered world cloud fixed while the latest sweep remains
        # visible through scene.current_points. Replacing history with each scan
        # makes the environment appear to rotate around a stationary robot.
        return (
            session.setdefault("history_points", []),
            session.setdefault("history_colors", []),
            int(session.get("accumulated_scan_count") or 0),
        )
    if not options.get("accumulate_hits", True):
        session["history_points"] = points
        session["history_colors"] = colors
        session["accumulated_scan_count"] = 1 if points else 0
        return points, colors, int(session["accumulated_scan_count"])

    history_points = session.setdefault("history_points", [])
    history_colors = session.setdefault("history_colors", [])
    history_points.extend(points)
    history_colors.extend(colors[:len(points)])
    maximum = max(1_000, min(250_000, int(options.get("max_accumulated_points") or 50_000)))
    if len(history_points) > maximum:
        del history_points[:len(history_points) - maximum]
        del history_colors[:len(history_colors) - maximum]
    session["accumulated_scan_count"] = int(session.get("accumulated_scan_count") or 0) + 1
    return history_points, history_colors, int(session["accumulated_scan_count"])


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
    options["history_paused"] = bool(session.get("history_paused"))
    pose_source = session.get("pose_source") if isinstance(session.get("pose_source"), dict) else {}
    pose: dict[str, Any] = {}
    if pose_source:
        try:
            pose_outputs = reader(dict(pose_source))
        except Exception as exc:
            pose_outputs = {
                "status": {
                    "state": "unavailable",
                    "source_fresh": False,
                    "error": f"pose stream read failed ({type(exc).__name__}: {exc})",
                }
            }
        if not isinstance(pose_outputs, dict):
            pose_outputs = {"status": {"state": "unavailable", "source_fresh": False}}
        pose_status = pose_outputs.get("status") if isinstance(pose_outputs.get("status"), dict) else {}
        pose_fresh = bool(pose_status.get("source_fresh"))
        pose_error = str(pose_status.get("error") or "").strip()
        if pose_fresh:
            pose_static_source = session.get("pose_static_source") if isinstance(session.get("pose_static_source"), dict) else {}
            if pose_static_source:
                try:
                    static_outputs = reader(dict(pose_static_source))
                except Exception:
                    static_outputs = {}
                if isinstance(static_outputs, dict):
                    pose_outputs = {
                        **pose_outputs,
                        "messages": [
                            *(pose_outputs.get("messages") or []),
                            *(static_outputs.get("messages") or []),
                            *([static_outputs.get("message")] if static_outputs.get("message") else []),
                        ],
                    }
            pose, pose_error = _normalize_pose(
                pose_outputs,
                pose_source,
                int(scan.get("header_time_ns") or 0),
                float(options.get("pose_sync_tolerance_s") or 0.25),
                str(options.get("pose_parent_frame") or "odom"),
                str(options.get("pose_child_frame") or "auto"),
                str(scan.get("frame") or ""),
            )
        if not pose_fresh or not pose:
            pose_state = str(pose_status.get("state") or "waiting")
            error = pose_error or "Viewer is waiting for a fresh pose message"
            session.update(
                live=False,
                status={
                    "kind": "blacknode.viewer-status",
                    "schema_version": 1,
                    "state": "stale" if pose_state == "stale" or pose_fresh else "waiting",
                    "source_fresh": True,
                    "pose_connected": True,
                    "pose_fresh": False,
                    "received": int(source_outputs.get("received") or 0),
                    "pose_received": int(pose_outputs.get("received") or 0),
                    "error": error,
                },
                report=error,
            )
            return
    sensor_pose = _combined_sensor_pose(options, pose)
    processed = process_laser_scan(
        scan,
        device=session["device"],
        filter_min_m=options["filter_min_m"],
        filter_max_m=options["filter_max_m"],
        stride=options["stride"],
        sensor_pose=sensor_pose,
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

    history_points, history_colors, accumulated_scan_count = _append_scan_history(
        session,
        processed,
    )
    scene = _scene_from_processed(
        processed,
        scan,
        source_outputs,
        {
            **options,
            "history_paused": bool(session.get("history_paused")),
        },
        history_points,
        history_colors,
        accumulated_scan_count,
        sensor_pose,
        pose,
    )
    if session["mode"] == "device":
        native_scan = {
            **scan,
            "viewer_pose": {
                "x_m": sensor_pose[0],
                "y_m": sensor_pose[1],
                "yaw_rad": sensor_pose[2],
            },
            "viewer_robot_pose": ({
                "x_m": float(pose.get("x_m") or 0.0),
                "y_m": float(pose.get("y_m") or 0.0),
                "yaw_rad": float(pose.get("yaw_rad") or 0.0),
            } if pose else {}),
            "history_registered": bool(pose),
            "viewer_frame": str(pose.get("frame") or scan.get("frame") or "laser"),
        }
        native = session.get("native") if isinstance(session.get("native"), dict) else {}
        if not native.get("running"):
            native = warp_viewer_runtime.start_viewer(
                viewer_id=session["viewer_id"],
                scan=native_scan,
                options={
                    **options,
                    "device": session["device"],
                    "live": True,
                    "accumulate_hits": bool(options.get("accumulate_hits", True))
                    and not bool(session.get("history_paused")),
                },
            )
            session["native"] = native
        else:
            native_update = warp_viewer_runtime.update_viewer_scan(session["viewer_id"], native_scan)
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
            "pose_connected": bool(pose_source),
            "pose_fresh": bool(pose) if pose_source else False,
            "pose_time_delta_seconds": pose.get("time_delta_seconds") if pose else None,
            "received": int(source_outputs.get("received") or 0),
            "point_count": int(scene.get("point_count") or 0),
            "current_point_count": int(scene.get("current_point_count") or 0),
            "accumulated_scan_count": int(scene.get("accumulated_scan_count") or 0),
            "history_paused": bool(session.get("history_paused")),
            "kernel_ms": float(scene.get("kernel_ms") or 0.0),
            "error": "",
        },
        report=(
            f"Viewer {session['mode']} retained {int(scene.get('point_count') or 0):,} "
            f"hits from {int(scene.get('accumulated_scan_count') or 0):,} scan(s); "
            f"{'pose-registered; ' if pose else 'sensor-local; '}"
            f"Warp {float(scene.get('kernel_ms') or 0.0):.3f} ms"
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
    pose_source: dict[str, Any] | None,
    pose_static_source: dict[str, Any] | None,
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
            "history_points": [],
            "history_colors": [],
            "accumulated_scan_count": 0,
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
    if pose_source and pose_source.get("kind") != "blacknode.message-stream":
        return {
            "running": False,
            "live": False,
            "scene": {},
            "status": {"state": "error", "error": "pose must be a blacknode.message-stream"},
            "viewer": {},
            "report": "Connect a pose-producing ROS2.stream to Viewer.pose",
        }
    if pose_static_source and pose_static_source.get("kind") != "blacknode.message-stream":
        return {
            "running": False, "live": False, "scene": {},
            "status": {"state": "error", "error": "pose_static must be a blacknode.message-stream"},
            "viewer": {}, "report": "Connect ROS2.stream to Viewer.pose_static",
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
            "pose_source": dict(pose_source or {}),
            "pose_static_source": dict(pose_static_source or {}),
            "source_reader": source_reader,
            "mode": selected_mode,
            "device": str(device or "cuda:0"),
            "options": dict(options),
            "running": True,
            "live": False,
            "history_paused": False,
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


def clear_viewer(viewer_id: str) -> dict[str, Any]:
    clean_id = _safe_id(viewer_id)
    with _LOCK:
        session = _SESSIONS.get(clean_id)
        if session is None:
            return viewer_status(clean_id)
        session["history_points"] = []
        session["history_colors"] = []
        session["accumulated_scan_count"] = 0
        session["history_paused"] = True
        scene = dict(session.get("scene") or {})
        if scene:
            scene.update(
                points=[],
                colors=[],
                point_count=0,
                accumulated_scan_count=0,
                display_count=0,
                display_stride=1,
                history_paused=True,
            )
        session["scene"] = scene
        if session.get("mode") == "device":
            warp_viewer_runtime.stop_viewer(clean_id)
            session["native"] = {}
        status = dict(session.get("status") or {})
        status["history_paused"] = True
        session["status"] = status
        session["report"] = "Viewer scan history cleared; accumulation is off"
        return _viewer_outputs(session)


def resume_viewer(viewer_id: str) -> dict[str, Any]:
    clean_id = _safe_id(viewer_id)
    with _LOCK:
        session = _SESSIONS.get(clean_id)
        if session is None:
            return viewer_status(clean_id)
        session["history_paused"] = False
        scene = dict(session.get("scene") or {})
        if scene:
            scene["history_paused"] = False
        session["scene"] = scene
        status = dict(session.get("status") or {})
        status["history_paused"] = False
        session["status"] = status
        if session.get("mode") == "device":
            warp_viewer_runtime.stop_viewer(clean_id)
            session["native"] = {}
        session["report"] = "Viewer accumulation enabled"
        return _viewer_outputs(session)


def pause_viewer(viewer_id: str) -> dict[str, Any]:
    clean_id = _safe_id(viewer_id)
    with _LOCK:
        session = _SESSIONS.get(clean_id)
        if session is None:
            return viewer_status(clean_id)
        session["history_paused"] = True
        scene = dict(session.get("scene") or {})
        if scene:
            scene["history_paused"] = True
        session["scene"] = scene
        status = dict(session.get("status") or {})
        status["history_paused"] = True
        session["status"] = status
        if session.get("mode") == "device":
            warp_viewer_runtime.stop_viewer(clean_id)
            session["native"] = {}
        session["report"] = "Viewer accumulation disabled; showing the latest scan"
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
