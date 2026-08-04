"""Managed real-time 2D SLAM sessions for generic LaserScan streams."""
from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable

try:
    import numpy as np
except Exception:  # pragma: no cover - dependency diagnostics cover this path
    np = None

from . import viewer_runtime, warp_viewer_runtime


_SESSIONS: dict[str, dict[str, Any]] = {}
_LOCK = threading.RLock()
_MAX_EDITOR_POINTS = 20_000
_MIN_MATCH_SCORE_GAIN = 0.02
_STATIONARY_TRANSLATION_M = 0.01
_STATIONARY_ROTATION_RAD = math.radians(0.5)


def _safe_id(value: str) -> str:
    return "".join(
        character for character in str(value or "")
        if character.isalnum() or character in "_-"
    )[:80]


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _compose(first: Any, second: Any) -> Any:
    yaw = float(first[2])
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.asarray([
        float(first[0]) + cosine * float(second[0]) - sine * float(second[1]),
        float(first[1]) + sine * float(second[0]) + cosine * float(second[1]),
        _wrap_angle(yaw + float(second[2])),
    ], dtype=np.float64)


def _relative(first: Any, second: Any) -> Any:
    dx = float(second[0]) - float(first[0])
    dy = float(second[1]) - float(first[1])
    cosine = math.cos(float(first[2]))
    sine = math.sin(float(first[2]))
    return np.asarray([
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
        _wrap_angle(float(second[2]) - float(first[2])),
    ], dtype=np.float64)


def _transform_points(points: Any, pose: Any) -> Any:
    if points is None or len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    cosine = math.cos(float(pose[2]))
    sine = math.sin(float(pose[2]))
    result = np.asarray(points, dtype=np.float32).copy()
    x = result[:, 0].copy()
    y = result[:, 1].copy()
    result[:, 0] = cosine * x - sine * y + float(pose[0])
    result[:, 1] = sine * x + cosine * y + float(pose[1])
    return result


def _expanded_cells(points: Any, resolution: float) -> set[tuple[int, int]]:
    if points is None or len(points) == 0:
        return set()
    cells = np.rint(np.asarray(points)[:, :2] / resolution).astype(np.int32)
    occupied: set[tuple[int, int]] = set()
    for cell_x, cell_y in cells:
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                occupied.add((int(cell_x + offset_x), int(cell_y + offset_y)))
    return occupied


def _match_score(local_points: Any, occupied: set[tuple[int, int]], pose: Any, resolution: float) -> float:
    if not occupied or local_points is None or len(local_points) == 0:
        return 0.0
    transformed = _transform_points(local_points, pose)
    cells = np.rint(transformed[:, :2] / resolution).astype(np.int32)
    hits = sum((int(x), int(y)) in occupied for x, y in cells)
    return hits / max(1, len(cells))


def correlative_match(
    local_points: Any,
    reference_points: Any,
    initial_pose: Any,
    *,
    resolution: float,
    linear_window: float,
    angular_window: float,
) -> tuple[Any, float]:
    """Find the highest occupancy-overlap pose near an odometry/motion prior."""
    if np is None:
        raise RuntimeError("NumPy is required for SLAM scan matching")
    local = np.asarray(local_points, dtype=np.float32)
    reference = np.asarray(reference_points, dtype=np.float32)
    initial = np.asarray(initial_pose, dtype=np.float64)
    if len(local) == 0 or len(reference) == 0:
        return initial.copy(), 0.0
    # Keep runtime bounded for dense 360-degree sensors while retaining the
    # whole angular field of view.
    local = local[::max(1, math.ceil(len(local) / 720))]
    occupied = _expanded_cells(reference, resolution)
    best = initial.copy()
    best_score = _match_score(local, occupied, best, resolution)

    def search(center: Any, linear_step: float, angular_step: float, radius: int) -> tuple[Any, float]:
        candidate_best = center.copy()
        candidate_score = _match_score(local, occupied, candidate_best, resolution)
        for x_index in range(-radius, radius + 1):
            for y_index in range(-radius, radius + 1):
                for yaw_index in range(-radius, radius + 1):
                    candidate = np.asarray([
                        float(center[0]) + x_index * linear_step,
                        float(center[1]) + y_index * linear_step,
                        _wrap_angle(float(center[2]) + yaw_index * angular_step),
                    ])
                    score = _match_score(local, occupied, candidate, resolution)
                    # Prefer the prior when overlap is tied; this prevents a
                    # stationary scan from drifting between equivalent cells.
                    # LaserScan overlap is quantized and frequently has many
                    # almost-equivalent poses. Do not turn tiny score changes
                    # from sensor noise into visible robot motion.
                    if score > candidate_score + _MIN_MATCH_SCORE_GAIN:
                        candidate_best = candidate
                        candidate_score = score
        return candidate_best, candidate_score

    coarse_linear = max(resolution * 2.0, linear_window / 4.0)
    coarse_angular = max(math.radians(0.5), angular_window / 4.0)
    coarse_radius = max(1, min(4, math.ceil(linear_window / coarse_linear)))
    best, best_score = search(best, coarse_linear, coarse_angular, coarse_radius)
    best, best_score = search(best, max(resolution * 0.5, coarse_linear / 2.0), coarse_angular / 2.0, 2)
    return best, float(best_score)


def _edge_error(first: Any, second: Any, measurement: Any) -> Any:
    predicted = _relative(first, second)
    error = predicted - measurement
    error[2] = _wrap_angle(float(error[2]))
    return error


def optimize_pose_graph(poses: Any, edges: list[dict[str, Any]], iterations: int = 5) -> Any:
    """Optimize SE(2) keyframe poses while anchoring the first keyframe."""
    if np is None:
        raise RuntimeError("NumPy is required for SLAM pose-graph optimization")
    optimized = np.asarray(poses, dtype=np.float64).copy()
    if len(optimized) < 2 or not edges:
        return optimized
    variable_count = (len(optimized) - 1) * 3
    epsilon = 1.0e-5
    for _ in range(max(1, iterations)):
        hessian = np.eye(variable_count, dtype=np.float64) * 1.0e-6
        gradient = np.zeros(variable_count, dtype=np.float64)
        for edge in edges:
            first_index = int(edge["i"])
            second_index = int(edge["j"])
            measurement = np.asarray(edge["measurement"], dtype=np.float64)
            residual = _edge_error(optimized[first_index], optimized[second_index], measurement)
            information = np.diag(np.asarray(edge.get("weight") or [1.0, 1.0, 1.0], dtype=np.float64))
            jacobians: dict[int, Any] = {}
            for node_index in (first_index, second_index):
                jacobian = np.zeros((3, 3), dtype=np.float64)
                for axis in range(3):
                    perturbed = optimized.copy()
                    perturbed[node_index, axis] += epsilon
                    changed = _edge_error(perturbed[first_index], perturbed[second_index], measurement)
                    delta = changed - residual
                    delta[2] = _wrap_angle(float(delta[2]))
                    jacobian[:, axis] = delta / epsilon
                jacobians[node_index] = jacobian
            for left_index, left_jacobian in jacobians.items():
                if left_index == 0:
                    continue
                left_slice = slice((left_index - 1) * 3, left_index * 3)
                gradient[left_slice] += left_jacobian.T @ information @ residual
                for right_index, right_jacobian in jacobians.items():
                    if right_index == 0:
                        continue
                    right_slice = slice((right_index - 1) * 3, right_index * 3)
                    hessian[left_slice, right_slice] += left_jacobian.T @ information @ right_jacobian
        try:
            step = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        optimized[1:] += step.reshape((-1, 3))
        optimized[:, 2] = np.arctan2(np.sin(optimized[:, 2]), np.cos(optimized[:, 2]))
        if float(np.linalg.norm(step)) < 1.0e-5:
            break
    return optimized


def _scan_values(outputs: dict[str, Any], source: dict[str, Any]) -> list[dict[str, Any]]:
    values = list(outputs.get("messages") or [])
    if outputs.get("message"):
        values.append(outputs["message"])
    scans: dict[tuple[int, int], dict[str, Any]] = {}
    for index, value in enumerate(values):
        scan = viewer_runtime._normalize_laser_scan(
            {**outputs, "message": value}, source,
        )
        if scan:
            scans[(int(scan.get("source_time_ns") or 0), index if not scan.get("source_time_ns") else 0)] = scan
    return [scans[key] for key in sorted(scans)][-12:]


def _read_odometry(session: dict[str, Any], scan: dict[str, Any]) -> Any | None:
    source = session.get("odometry_source")
    if not isinstance(source, dict) or not source:
        return None
    reader = session["source_reader"]
    try:
        outputs = reader(dict(source))
    except Exception:
        return None
    if not isinstance(outputs, dict):
        return None
    status = outputs.get("status") if isinstance(outputs.get("status"), dict) else {}
    if not status.get("source_fresh"):
        return None
    options = session["options"]
    pose, _error = viewer_runtime._normalize_pose(
        outputs,
        source,
        int(scan.get("header_time_ns") or 0),
        float(options["pose_sync_tolerance_s"]),
        str(options["pose_parent_frame"]),
        str(options["pose_child_frame"]),
        str(scan.get("frame") or ""),
    )
    if not pose:
        return None
    x, y, yaw = viewer_runtime._combined_sensor_pose(options, pose)
    return np.asarray([x, y, yaw], dtype=np.float64)


def _map_points(session: dict[str, Any]) -> Any:
    keyframes = session.get("keyframes") or []
    if not keyframes:
        return np.empty((0, 3), dtype=np.float32)
    transformed = [
        _transform_points(keyframe["points"], keyframe["pose"])
        for keyframe in keyframes
    ]
    points = np.concatenate(transformed, axis=0) if transformed else np.empty((0, 3), dtype=np.float32)
    resolution = float(session["options"]["map_resolution_m"])
    unique: dict[tuple[int, int], Any] = {}
    for point in points:
        cell = (round(float(point[0]) / resolution), round(float(point[1]) / resolution))
        unique[cell] = point
    values = np.asarray(list(unique.values()), dtype=np.float32)
    maximum = int(session["options"]["max_map_points"])
    if len(values) > maximum:
        values = values[-maximum:]
    session["map_points"] = values
    return values


def _should_add_keyframe(session: dict[str, Any], pose: Any, scan_time_ns: int) -> bool:
    del scan_time_ns
    keyframes = session.get("keyframes") or []
    if not keyframes:
        return True
    previous = keyframes[-1]
    delta = _relative(previous["pose"], pose)
    options = session["options"]
    return (
        math.hypot(float(delta[0]), float(delta[1])) >= float(options["keyframe_translation_m"])
        or abs(float(delta[2])) >= float(options["keyframe_rotation_rad"])
    )


def _add_keyframe(session: dict[str, Any], local_points: Any, pose: Any, scan_time_ns: int, score: float) -> None:
    keyframes = session["keyframes"]
    if len(keyframes) >= int(session["options"]["max_keyframes"]):
        session["map_limited"] = True
        return
    keyframes.append({
        "pose": np.asarray(pose, dtype=np.float64),
        "points": np.asarray(local_points, dtype=np.float32),
        "source_time_ns": int(scan_time_ns),
        "score": float(score),
    })
    current_index = len(keyframes) - 1
    if current_index > 0:
        measurement = _relative(keyframes[current_index - 1]["pose"], keyframes[current_index]["pose"])
        confidence = max(0.1, float(score))
        session["edges"].append({
            "i": current_index - 1,
            "j": current_index,
            "measurement": measurement,
            "weight": [20.0 * confidence, 20.0 * confidence, 40.0 * confidence],
            "type": "scan",
        })

    options = session["options"]
    separation = int(options["loop_closure_min_separation"])
    if current_index < separation:
        _map_points(session)
        return
    current_pose = keyframes[current_index]["pose"]
    candidates = [
        index for index in range(0, current_index - separation + 1)
        if math.hypot(
            float(keyframes[index]["pose"][0] - current_pose[0]),
            float(keyframes[index]["pose"][1] - current_pose[1]),
        ) <= float(options["loop_closure_radius_m"])
    ]
    best: tuple[int, Any, float] | None = None
    for candidate_index in candidates[-12:]:
        reference = _transform_points(
            keyframes[candidate_index]["points"],
            keyframes[candidate_index]["pose"],
        )
        matched, loop_score = correlative_match(
            local_points,
            reference,
            current_pose,
            resolution=float(options["map_resolution_m"]),
            linear_window=float(options["loop_closure_radius_m"]),
            angular_window=float(options["match_angular_window_rad"]) * 2.0,
        )
        if best is None or loop_score > best[2]:
            best = (candidate_index, matched, loop_score)
    if best is not None and best[2] >= float(options["loop_closure_min_score"]):
        candidate_index, matched_pose, loop_score = best
        session["edges"].append({
            "i": candidate_index,
            "j": current_index,
            "measurement": _relative(keyframes[candidate_index]["pose"], matched_pose),
            "weight": [80.0 * loop_score, 80.0 * loop_score, 120.0 * loop_score],
            "type": "loop",
            "score": loop_score,
        })
        poses = optimize_pose_graph(
            np.asarray([keyframe["pose"] for keyframe in keyframes]),
            session["edges"],
        )
        for keyframe, optimized_pose in zip(keyframes, poses):
            keyframe["pose"] = optimized_pose
        session["loop_closures"] = int(session.get("loop_closures") or 0) + 1
        session["last_loop_score"] = float(loop_score)
    _map_points(session)


def _scene(session: dict[str, Any], scan: dict[str, Any], current_points: Any, kernel_ms: float) -> dict[str, Any]:
    map_points = np.asarray(session.get("map_points"), dtype=np.float32)
    display_stride = max(1, math.ceil(len(map_points) / _MAX_EDITOR_POINTS))
    display = map_points[::display_stride]
    pose_value = session.get("pose")
    current_pose = np.asarray(
        pose_value if pose_value is not None else [0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    current_world = _transform_points(current_points, current_pose)
    keyframes = session.get("keyframes") or []
    loop_edges = [edge for edge in session.get("edges") or [] if edge.get("type") == "loop"]
    return {
        "kind": "blacknode.viewer-scene",
        "schema_version": 1,
        "primitive": "point-cloud",
        "projection": "metric-xy",
        "frame": "map",
        "sequence": int(scan.get("source_time_ns") or 0),
        "points": display.astype(float).tolist(),
        "colors": [[0.04, 0.36, 0.48]] * len(display),
        "current_points": current_world.astype(float).tolist(),
        "current_colors": [[0.0, 0.78, 1.0]] * len(current_world),
        "point_count": len(map_points),
        "current_point_count": len(current_world),
        "accumulated_scan_count": len(keyframes),
        "display_count": len(display),
        "display_stride": display_stride,
        "history_registered": True,
        "history_paused": bool(session.get("mapping_paused")),
        "pose_source": "scan-matching",
        "sensor": {
            "x_m": float(current_pose[0]),
            "y_m": float(current_pose[1]),
            "yaw_rad": float(current_pose[2]),
        },
        "robot": {
            "length_m": float(session["options"].get("robot_length_m") or 0.25),
            "width_m": float(session["options"].get("robot_width_m") or 0.22),
            "height_m": float(session["options"].get("robot_height_m") or 0.08),
        },
        "scan": {
            "angle_min_rad": float(scan.get("angle_min") or 0.0),
            "angle_max_rad": float(scan.get("angle_max") or 0.0),
            "angle_increment_rad": float(scan.get("angle_increment") or 0.0),
            "range_min_m": float(scan.get("range_min") or 0.0),
            "range_max_m": float(scan.get("range_max") or 0.0),
        },
        "view": {
            "radius_m": max(1.0, float(session["options"]["filter_max_m"])),
            "units": "meters",
        },
        "trajectory": [
            [float(keyframe["pose"][0]), float(keyframe["pose"][1]), 0.0]
            for keyframe in keyframes
        ],
        "loop_closures": [
            {
                "from": [float(keyframes[edge["i"]]["pose"][0]), float(keyframes[edge["i"]]["pose"][1]), 0.0],
                "to": [float(keyframes[edge["j"]]["pose"][0]), float(keyframes[edge["j"]]["pose"][1]), 0.0],
                "score": float(edge.get("score") or 0.0),
            }
            for edge in loop_edges
        ],
        "registration": {
            "method": "correlative-scan-matching",
            "pose_graph_optimized": bool(loop_edges),
        },
        "slam": {
            "match_score": float(session.get("match_score") or 0.0),
            "keyframes": len(keyframes),
            "constraints": len(session.get("edges") or []),
            "loop_closures": int(session.get("loop_closures") or 0),
            "last_loop_score": float(session.get("last_loop_score") or 0.0),
            "map_resolution_m": float(session["options"]["map_resolution_m"]),
            "map_limited": bool(session.get("map_limited")),
        },
        "animation": {
            "enabled": bool(session["options"].get("animate_scan", True)),
            "show_rays": bool(session["options"].get("show_rays", True)),
            "ray_trail_count": 48,
            "pulse_hz": max(0.25, min(30.0, 1.0 / max(0.001, float(scan.get("scan_time") or 0.1)))),
            "accumulate_hits": not bool(session.get("mapping_paused")),
        },
        "device": str(session.get("device") or ""),
        "kernel_ms": float(kernel_ms),
    }


def _outputs(session: dict[str, Any]) -> dict[str, Any]:
    pose_value = session.get("pose")
    pose = np.asarray(
        pose_value if pose_value is not None else [0.0, 0.0, 0.0],
        dtype=np.float64,
    ) if np is not None else [0.0, 0.0, 0.0]
    map_points = session.get("map_points")
    point_count = len(map_points) if map_points is not None else 0
    return {
        "running": bool(session.get("running")),
        "live": bool(session.get("live")),
        "scene": dict(session.get("scene") or {}),
        "pose": {
            "kind": "blacknode.slam-pose",
            "schema_version": 1,
            "frame": "map",
            "child_frame": "laser",
            "x_m": float(pose[0]),
            "y_m": float(pose[1]),
            "yaw_rad": float(pose[2]),
            "source_time_ns": int(session.get("source_time_ns") or 0),
        },
        "map": {
            "kind": "blacknode.slam-map",
            "schema_version": 1,
            "frame": "map",
            "representation": "point-cloud",
            "resolution_m": float((session.get("options") or {}).get("map_resolution_m") or 0.05),
            "point_count": point_count,
            "keyframes": len(session.get("keyframes") or []),
            "loop_closures": int(session.get("loop_closures") or 0),
        },
        "status": dict(session.get("status") or {}),
        "viewer": {
            "kind": "blacknode.viewer",
            "schema_version": 1,
            "viewer_id": str(session.get("slam_id") or ""),
            "mode": str(session.get("mode") or "editor"),
            "state": "running" if session.get("running") else "stopped",
        },
        "report": str(session.get("report") or ""),
    }


def _update_native_viewer(session: dict[str, Any], scan: dict[str, Any], pose: Any) -> None:
    if session["mode"] != "device":
        return
    options = session["options"]
    native_scan = {
        **scan,
        "viewer_pose": {
            "x_m": float(pose[0]),
            "y_m": float(pose[1]),
            "yaw_rad": float(pose[2]),
        },
        "history_registered": True,
        "viewer_frame": "map",
    }
    native = session.get("native") if isinstance(session.get("native"), dict) else {}
    if not native.get("running"):
        native = warp_viewer_runtime.start_viewer(
            viewer_id=session["slam_id"],
            scan=native_scan,
            options={
                **options,
                "device": session["device"],
                "live": True,
                # Keep the native renderer on the direct CUDA/OpenGL interop
                # path. The editor overlays the animated ray sweep.
                "animate_scan": False,
                "show_rays": False,
                "accumulate_hits": not bool(session.get("mapping_paused")),
            },
        )
        session["native"] = native
    else:
        update = warp_viewer_runtime.update_viewer_scan(session["slam_id"], native_scan)
        if not update.get("ok"):
            session["native"] = update


def _process_scan(session: dict[str, Any], scan: dict[str, Any]) -> None:
    from .warp_points import process_laser_scan

    options = session["options"]
    processed = process_laser_scan(
        scan,
        device=session["device"],
        filter_min_m=float(options["filter_min_m"]),
        filter_max_m=float(options["filter_max_m"]),
        stride=int(options["stride"]),
        sensor_pose=(0.0, 0.0, 0.0),
        include_raw_points=False,
        compare_numpy=False,
    )
    if not processed.get("ok"):
        raise RuntimeError(str((processed.get("report") or {}).get("error") or "Warp scan processing failed"))
    local_points = np.asarray(processed.get("filtered_points") or [], dtype=np.float32)
    if len(local_points) == 0:
        raise RuntimeError("LaserScan has no valid returns inside the configured range")

    scan_time_ns = int(scan.get("source_time_ns") or time.time_ns())
    if len(local_points) < 8:
        pose_value = session.get("pose")
        pose = np.asarray(
            pose_value if pose_value is not None else [0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        scene = _scene(
            session,
            scan,
            local_points,
            float(processed.get("kernel_ms") or 0.0),
        )
        session.update(
            live=True,
            source_time_ns=scan_time_ns,
            scene=scene,
            status={
                "kind": "blacknode.slam-status",
                "schema_version": 1,
                "state": "waiting",
                "source_fresh": True,
                "localized": False,
                "mapping": not bool(session.get("mapping_paused")),
                "valid_returns": len(local_points),
                "error": (
                    f"SLAM needs at least 8 valid returns; displaying {len(local_points)} live"
                ),
            },
            report=(
                f"SLAM is displaying {len(local_points)} live return(s) and waiting for "
                "at least 8 valid returns to localize"
            ),
        )
        _update_native_viewer(session, scan, pose)
        return

    odometry = _read_odometry(session, scan)
    pose_value = session.get("pose")
    previous_pose = np.asarray(
        pose_value if pose_value is not None else [0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    odometry_stationary = False
    if session.get("source_time_ns"):
        if odometry is not None and session.get("last_odometry") is not None:
            odometry_delta = _relative(session["last_odometry"], odometry)
            odometry_stationary = (
                math.hypot(float(odometry_delta[0]), float(odometry_delta[1]))
                < _STATIONARY_TRANSLATION_M
                and abs(float(odometry_delta[2])) < _STATIONARY_ROTATION_RAD
            )
            initial = previous_pose.copy() if odometry_stationary else _compose(previous_pose, odometry_delta)
        elif session.get("previous_pose") is not None:
            initial = _compose(previous_pose, _relative(session["previous_pose"], previous_pose))
        else:
            initial = previous_pose.copy()
        reference = session.get("map_points")
        if reference is not None and len(reference) >= 8:
            pose, score = correlative_match(
                local_points,
                reference,
                initial,
                resolution=float(options["map_resolution_m"]),
                linear_window=float(options["match_linear_window_m"]),
                angular_window=float(options["match_angular_window_rad"]),
            )
            if odometry_stationary:
                pose = previous_pose.copy()
        else:
            pose, score = initial, 0.0
    else:
        pose, score = np.zeros(3, dtype=np.float64), 1.0
        if odometry is not None:
            session["odometry_origin"] = odometry.copy()

    session["previous_pose"] = previous_pose
    session["pose"] = pose
    session["match_score"] = float(score)
    if odometry is not None:
        session["last_odometry"] = odometry
    if not session.get("mapping_paused") and _should_add_keyframe(session, pose, scan_time_ns):
        _add_keyframe(session, local_points, pose, scan_time_ns, score)
        if session.get("keyframes"):
            pose = np.asarray(session["keyframes"][-1]["pose"], dtype=np.float64)
            session["pose"] = pose
    current_world = _transform_points(local_points, pose)
    scene = _scene(session, scan, local_points, float(processed.get("kernel_ms") or 0.0))
    session.update(
        live=True,
        source_time_ns=scan_time_ns,
        scene=scene,
        status={
            "kind": "blacknode.slam-status",
            "schema_version": 1,
            "state": "ready",
            "source_fresh": True,
            "localized": bool(session.get("keyframes")),
            "mapping": not bool(session.get("mapping_paused")),
            "match_score": float(score),
            "keyframes": len(session.get("keyframes") or []),
            "loop_closures": int(session.get("loop_closures") or 0),
            "error": "",
        },
        report=(
            f"SLAM localized at ({float(pose[0]):.2f}, {float(pose[1]):.2f}) m, "
            f"{math.degrees(float(pose[2])):.1f}°; map {len(session.get('map_points')) if session.get('map_points') is not None else 0:,} points; "
            f"score {float(score):.2f}; {int(session.get('loop_closures') or 0)} loop closure(s)"
        ),
    )

    _update_native_viewer(session, scan, pose)


def _update_session(session: dict[str, Any]) -> None:
    reader = session.get("source_reader")
    if not callable(reader):
        reader = viewer_runtime._local_stream_reader
    try:
        outputs = reader(dict(session["source"]))
    except Exception as exc:
        outputs = {"status": {"state": "unavailable", "source_fresh": False, "error": str(exc)}}
    if not isinstance(outputs, dict):
        outputs = {"status": {"state": "unavailable", "source_fresh": False}}
    source_status = outputs.get("status") if isinstance(outputs.get("status"), dict) else {}
    scans = _scan_values(outputs, session["source"])
    received = int(outputs.get("received") or source_status.get("received") or 0)
    previous_received = int(session.get("source_received") or 0)
    if received > 0:
        if received == previous_received:
            unseen = []
        elif received > previous_received:
            unseen = scans[-min(len(scans), max(1, received - previous_received)):]
        else:
            # A restarted subscriber resets its counter. The newest scan is new
            # even when its sensor clock is behind the previous process.
            unseen = scans[-1:]
    else:
        unseen = [
            scan for scan in scans
            if int(scan.get("source_time_ns") or 0) > int(session.get("source_time_ns") or 0)
        ]
    if not source_status.get("source_fresh") or not scans:
        error = str(source_status.get("error") or "SLAM is waiting for a fresh LaserScan message")
        session.update(
            live=False,
            status={
                "kind": "blacknode.slam-status",
                "schema_version": 1,
                "state": "waiting",
                "source_fresh": False,
                "error": error,
            },
            report=error,
        )
        return
    if not unseen:
        session["live"] = True
        return
    try:
        # SLAM is a live estimator, not a replay queue. If processing briefly
        # falls behind, use the newest complete scan and the newest odometry
        # prior instead of making the operator wait while stale frames cook.
        _process_scan(session, unseen[-1])
        if received > 0:
            session["source_received"] = received
            status = dict(session.get("status") or {})
            status["received"] = received
            session["status"] = status
    except Exception as exc:
        session.update(
            live=False,
            status={
                "kind": "blacknode.slam-status",
                "schema_version": 1,
                "state": "error",
                "source_fresh": True,
                "error": f"{type(exc).__name__}: {exc}",
            },
            report=f"SLAM processing failed ({type(exc).__name__}: {exc})",
        )


def _worker(slam_id: str, stop_event: threading.Event, interval: float) -> None:
    while not stop_event.wait(interval):
        with _LOCK:
            session = _SESSIONS.get(slam_id)
            if session is None or session.get("stop_event") is not stop_event:
                return
        session_lock = session["session_lock"]
        with session_lock:
            _update_session(session)
            session["snapshot"] = _outputs(session)


def start_slam(
    *,
    slam_id: str,
    node_id: str,
    source: dict[str, Any],
    odometry_source: dict[str, Any] | None,
    mode: str,
    device: str,
    options: dict[str, Any],
    source_reader: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    clean_id = _safe_id(slam_id)
    if not clean_id:
        return {"running": False, "live": False, "status": {"state": "error", "error": "slam_id is required"}, "report": "slam_id is required"}
    if np is None:
        return {"running": False, "live": False, "status": {"state": "unavailable", "error": "NumPy is unavailable"}, "report": "Install numpy>=1.24"}
    if source.get("kind") != "blacknode.message-stream":
        return {"running": False, "live": False, "status": {"state": "error", "error": "source must be a message stream"}, "report": "Connect ROS2.stream to SLAM.source"}
    if odometry_source and odometry_source.get("kind") != "blacknode.message-stream":
        return {"running": False, "live": False, "status": {"state": "error", "error": "odometry must be a message stream"}, "report": "Connect ROS2.stream to SLAM.odometry"}
    selected_mode = str(mode or "editor").strip().lower()
    if selected_mode not in {"editor", "device"}:
        selected_mode = "editor"
    with _LOCK:
        previous = _SESSIONS.pop(clean_id, None)
    if previous:
        event = previous.get("stop_event")
        if isinstance(event, threading.Event):
            event.set()
        if previous.get("mode") == "device":
            warp_viewer_runtime.stop_viewer(clean_id)
    stop_event = threading.Event()
    session = {
        "slam_id": clean_id,
        "node_id": str(node_id or ""),
        "source": dict(source),
        "odometry_source": dict(odometry_source or {}),
        "source_reader": source_reader if callable(source_reader) else viewer_runtime._local_stream_reader,
        "mode": selected_mode,
        "device": str(device or "cuda:0"),
        "options": dict(options),
        "running": True,
        "live": False,
        "mapping_paused": False,
        "keyframes": [],
        "edges": [],
        "map_points": np.empty((0, 3), dtype=np.float32),
        "pose": np.zeros(3, dtype=np.float64),
        "previous_pose": None,
        "last_odometry": None,
        "source_time_ns": 0,
        "source_received": 0,
        "loop_closures": 0,
        "scene": {},
        "native": {},
        "status": {"kind": "blacknode.slam-status", "schema_version": 1, "state": "waiting", "source_fresh": False, "error": ""},
        "report": "SLAM started; waiting for a fresh LaserScan message",
        "stop_event": stop_event,
        "session_lock": threading.RLock(),
        "snapshot": {},
    }
    with _LOCK:
        _SESSIONS[clean_id] = session
    with session["session_lock"]:
        _update_session(session)
        session["snapshot"] = _outputs(session)
        interval = max(0.02, 1.0 / max(1.0, min(120.0, float(options.get("fps") or 30.0))))
        worker = threading.Thread(target=_worker, args=(clean_id, stop_event, interval), name=f"blacknode-slam-{clean_id}", daemon=True)
        session["worker"] = worker
        worker.start()
        return _outputs(session)


def slam_status(slam_id: str) -> dict[str, Any]:
    clean_id = _safe_id(slam_id)
    with _LOCK:
        session = _SESSIONS.get(clean_id)
    if session is None:
        return {"running": False, "live": False, "scene": {}, "pose": {}, "map": {}, "status": {"kind": "blacknode.slam-status", "schema_version": 1, "state": "stopped"}, "viewer": {"viewer_id": clean_id, "state": "stopped"}, "report": "SLAM is stopped"}
    with session["session_lock"]:
        _update_session(session)
        outputs = _outputs(session)
        session["snapshot"] = outputs
        return outputs


def clear_slam(slam_id: str) -> dict[str, Any]:
    clean_id = _safe_id(slam_id)
    with _LOCK:
        session = _SESSIONS.get(clean_id)
    if session is None:
        return slam_status(clean_id)
    with session["session_lock"]:
        session.update(
            mapping_paused=True,
            keyframes=[], edges=[], map_points=np.empty((0, 3), dtype=np.float32),
            pose=np.zeros(3, dtype=np.float64), previous_pose=None, last_odometry=None,
            source_time_ns=0, loop_closures=0, last_loop_score=0.0, scene={}, native={},
            report="SLAM map cleared; mapping is off",
        )
        if session.get("mode") == "device":
            warp_viewer_runtime.stop_viewer(clean_id)
        status = dict(session.get("status") or {})
        status.update(mapping=False, keyframes=0, loop_closures=0)
        session["status"] = status
        outputs = _outputs(session)
        session["snapshot"] = outputs
        return outputs


def set_mapping(slam_id: str, enabled: bool) -> dict[str, Any]:
    clean_id = _safe_id(slam_id)
    with _LOCK:
        session = _SESSIONS.get(clean_id)
    if session is None:
        return slam_status(clean_id)
    with session["session_lock"]:
        session["mapping_paused"] = not enabled
        if session.get("mode") == "device":
            warp_viewer_runtime.stop_viewer(clean_id)
            session["native"] = {}
        scene = dict(session.get("scene") or {})
        scene["history_paused"] = not enabled
        animation = dict(scene.get("animation") or {})
        animation["accumulate_hits"] = enabled
        scene["animation"] = animation
        session["scene"] = scene
        session["report"] = "SLAM mapping enabled" if enabled else "SLAM mapping disabled; localization remains active"
        outputs = _outputs(session)
        session["snapshot"] = outputs
        return outputs


def stop_slam(slam_id: str = "") -> dict[str, Any]:
    clean_id = _safe_id(slam_id)
    with _LOCK:
        ids = [clean_id] if clean_id else list(_SESSIONS)
        sessions = [
            (session_id, session)
            for session_id in ids
            if (session := _SESSIONS.pop(session_id, None)) is not None
        ]
    for session_id, session in sessions:
        event = session.get("stop_event")
        if isinstance(event, threading.Event):
            event.set()
        if session.get("mode") == "device":
            warp_viewer_runtime.stop_viewer(session_id)
    stopped = len(sessions)
    return {"ok": True, "stopped": stopped}


def runtime_status() -> dict[str, Any]:
    with _LOCK:
        sessions = list(_SESSIONS.items())
    node_outputs = []
    for slam_id, session in sessions:
        outputs = session.get("snapshot")
        if not isinstance(outputs, dict) or not outputs:
            outputs = _outputs(session)
        node_outputs.append({
            "node_type": "SLAM",
            "node_id": session.get("node_id", ""),
            "run_id": slam_id,
            "outputs": outputs,
        })
    return {
        "ok": all(item["outputs"].get("status", {}).get("state") != "error" for item in node_outputs),
        "active": bool(node_outputs),
        "streams": [],
        "managed_runs": [],
        "node_outputs": node_outputs,
        "detached_count": 0,
        "report": f"{len(node_outputs)} SLAM session(s) active" if node_outputs else "no SLAM sessions active",
    }


def stop_runtime_services() -> dict[str, Any]:
    stopped = int(stop_slam().get("stopped") or 0)
    return {"ok": True, "stopped": {"streams": stopped, "managed_runs": 0, "detached": 0}, "report": f"stopped {stopped} SLAM session(s)"}
