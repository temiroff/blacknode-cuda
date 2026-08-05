"""Warp HashGrid alignment and confidence coloring for LiDAR/RGB-D fusion."""
from __future__ import annotations

import math
import time
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - package diagnostics own this path
    np = None

try:
    import warp as wp
except Exception:  # pragma: no cover - discovery must work without Warp
    wp = None


if wp is not None:

    @wp.kernel
    def _score_calibration_kernel(
        grid: wp.uint64,
        lidar_points: wp.array(dtype=wp.vec3),
        depth_points: wp.array(dtype=wp.vec3),
        anchor: wp.vec3,
        candidate_x: wp.array(dtype=wp.float32),
        candidate_y: wp.array(dtype=wp.float32),
        candidate_yaw: wp.array(dtype=wp.float32),
        depth_count: wp.int32,
        maximum_distance: wp.float32,
        residual_sums: wp.array(dtype=wp.float32),
        matched_counts: wp.array(dtype=wp.int32),
    ):
        work_index = wp.tid()
        hypothesis_index = work_index // depth_count
        depth_index = work_index - hypothesis_index * depth_count
        point = depth_points[depth_index]
        relative = point - anchor
        yaw = candidate_yaw[hypothesis_index]
        cosine = wp.cos(yaw)
        sine = wp.sin(yaw)
        corrected = wp.vec3(
            anchor[0] + cosine * relative[0] - sine * relative[1] + candidate_x[hypothesis_index],
            anchor[1] + sine * relative[0] + cosine * relative[1] + candidate_y[hypothesis_index],
            point[2],
        )
        maximum_distance_sq = maximum_distance * maximum_distance
        nearest_distance_sq = maximum_distance_sq + wp.float32(1.0)
        for lidar_index in wp.hash_grid_query(grid, corrected, maximum_distance):
            delta = corrected - lidar_points[lidar_index]
            distance_sq = wp.dot(delta, delta)
            if distance_sq < nearest_distance_sq:
                nearest_distance_sq = distance_sq
        if nearest_distance_sq <= maximum_distance_sq:
            wp.atomic_add(residual_sums, hypothesis_index, wp.sqrt(nearest_distance_sq))
            wp.atomic_add(matched_counts, hypothesis_index, 1)


    @wp.kernel
    def _align_depth_kernel(
        grid: wp.uint64,
        lidar_points: wp.array(dtype=wp.vec3),
        depth_points: wp.array(dtype=wp.vec3),
        depth_colors: wp.array(dtype=wp.vec3),
        anchor: wp.vec3,
        correction_x: wp.float32,
        correction_y: wp.float32,
        correction_yaw: wp.float32,
        maximum_distance: wp.float32,
        corrected_points: wp.array(dtype=wp.vec3),
        display_colors: wp.array(dtype=wp.vec3),
        residuals: wp.array(dtype=wp.float32),
        confidence: wp.array(dtype=wp.float32),
        matched: wp.array(dtype=wp.int32),
    ):
        depth_index = wp.tid()
        point = depth_points[depth_index]
        relative = point - anchor
        cosine = wp.cos(correction_yaw)
        sine = wp.sin(correction_yaw)
        corrected = wp.vec3(
            anchor[0] + cosine * relative[0] - sine * relative[1] + correction_x,
            anchor[1] + sine * relative[0] + cosine * relative[1] + correction_y,
            point[2],
        )
        corrected_points[depth_index] = corrected
        maximum_distance_sq = maximum_distance * maximum_distance
        nearest_distance_sq = maximum_distance_sq + wp.float32(1.0)
        for lidar_index in wp.hash_grid_query(grid, corrected, maximum_distance):
            delta = corrected - lidar_points[lidar_index]
            distance_sq = wp.dot(delta, delta)
            if distance_sq < nearest_distance_sq:
                nearest_distance_sq = distance_sq
        if nearest_distance_sq <= maximum_distance_sq:
            residual = wp.sqrt(nearest_distance_sq)
            score = wp.clamp(1.0 - residual / maximum_distance, 0.0, 1.0)
            ratio = wp.clamp(residual / maximum_distance, 0.0, 1.0)
            heat = wp.vec3(ratio, 1.0 - ratio, 0.12)
            display_colors[depth_index] = depth_colors[depth_index] * 0.68 + heat * 0.32
            residuals[depth_index] = residual
            confidence[depth_index] = score
            matched[depth_index] = 1
        else:
            display_colors[depth_index] = wp.vec3(1.0, 0.18, 0.12)
            residuals[depth_index] = maximum_distance
            confidence[depth_index] = 0.0
            matched[depth_index] = 0


def _error(message: str, *, device: str = "") -> dict[str, Any]:
    return {
        "ok": False,
        "fused_points": [],
        "fused_colors": [],
        "report": {"state": "error", "device": device, "error": message},
    }


def _bounded(values: Any, maximum: int) -> Any:
    array = np.asarray(values or [], dtype=np.float32).reshape((-1, 3))
    if len(array) <= maximum:
        return np.ascontiguousarray(array)
    indices = np.linspace(0, len(array) - 1, maximum, dtype=np.int64)
    return np.ascontiguousarray(array[indices])


def _world_depth(point_cloud: dict[str, Any], pose: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    points = np.asarray(point_cloud.get("points_xyz") or [], dtype=np.float32).reshape((-1, 3))
    colors = np.asarray(point_cloud.get("colors_rgb") or [], dtype=np.float32).reshape((-1, 3))
    confidence = np.asarray(point_cloud.get("confidence") or [], dtype=np.float32).reshape((-1,))
    if len(colors) != len(points):
        colors = np.tile(np.asarray([[0.08, 0.78, 0.92]], dtype=np.float32), (len(points), 1))
    if len(confidence) != len(points):
        confidence = np.ones(len(points), dtype=np.float32)
    yaw = float(pose.get("yaw_rad") or 0.0)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    world = points.copy()
    world[:, 0] = float(pose.get("x_m") or 0.0) + cosine * points[:, 0] - sine * points[:, 1]
    world[:, 1] = float(pose.get("y_m") or 0.0) + sine * points[:, 0] + cosine * points[:, 1]
    world[:, 2] = float(pose.get("z_m") or 0.0) + points[:, 2]
    extrinsics = point_cloud.get("processing", {}).get("sensor_extrinsics", {})
    anchor = np.asarray([
        float(pose.get("x_m") or 0.0) + cosine * float(extrinsics.get("x_m") or 0.0) - sine * float(extrinsics.get("y_m") or 0.0),
        float(pose.get("y_m") or 0.0) + sine * float(extrinsics.get("x_m") or 0.0) + cosine * float(extrinsics.get("y_m") or 0.0),
        float(pose.get("z_m") or 0.0) + float(extrinsics.get("z_m") or 0.0),
    ], dtype=np.float32)
    return np.ascontiguousarray(world), np.ascontiguousarray(colors), confidence, anchor


def _candidates(stage: dict[str, Any]) -> tuple[Any, Any, Any]:
    steps = max(1, min(5, int(stage.get("calibration_steps") or 3)))
    if steps % 2 == 0:
        steps += 1
    translation = max(0.0, min(1.0, float(stage.get("calibration_translation_m") or 0.1)))
    yaw = math.radians(max(0.0, min(20.0, float(stage.get("calibration_yaw_deg") or 3.0))))
    if not stage.get("calibration_search", True):
        steps = 1
        translation = 0.0
        yaw = 0.0
    translations = np.linspace(-translation, translation, steps, dtype=np.float32)
    yaws = np.linspace(-yaw, yaw, steps, dtype=np.float32)
    candidates = [(x, y, angle) for angle in yaws for y in translations for x in translations]
    return tuple(np.asarray([item[index] for item in candidates], dtype=np.float32) for index in range(3))


def _cpu_hypotheses(
    lidar: Any,
    depth: Any,
    anchor: Any,
    candidate_x: Any,
    candidate_y: Any,
    candidate_yaw: Any,
    maximum_distance: float,
) -> tuple[int, Any, Any]:
    sums = np.zeros(len(candidate_x), dtype=np.float64)
    counts = np.zeros(len(candidate_x), dtype=np.int32)
    for hypothesis in range(len(candidate_x)):
        cosine, sine = math.cos(float(candidate_yaw[hypothesis])), math.sin(float(candidate_yaw[hypothesis]))
        relative = depth - anchor
        corrected = depth.copy()
        corrected[:, 0] = anchor[0] + cosine * relative[:, 0] - sine * relative[:, 1] + candidate_x[hypothesis]
        corrected[:, 1] = anchor[1] + sine * relative[:, 0] + cosine * relative[:, 1] + candidate_y[hypothesis]
        distances = np.linalg.norm(corrected[:, None, :] - lidar[None, :, :], axis=2)
        nearest = distances.min(axis=1)
        valid = nearest <= maximum_distance
        sums[hypothesis] = float(nearest[valid].sum())
        counts[hypothesis] = int(np.count_nonzero(valid))
    scores = sums / np.maximum(counts, 1) + (1.0 - counts / max(1, len(depth))) * maximum_distance
    return int(np.argmin(scores)), sums, counts


def process_sensor_fusion(
    lidar_points: Any,
    depth_point_cloud: dict[str, Any],
    *,
    pose: dict[str, Any],
    stage: dict[str, Any],
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Align and color one fresh LiDAR/RGB-D pair through a Warp HashGrid."""
    if wp is None:
        return _error("NVIDIA Warp is not installed; install warp-lang>=1.15", device=device)
    if np is None:
        return _error("NumPy is not installed; install numpy>=1.24", device=device)
    if stage.get("kind") != "blacknode.warp-sensor-fusion":
        return _error("connect WarpSensorFusion.stage to Viewer.sensor_fusion", device=device)
    try:
        selected_device = wp.get_device(device)
        maximum_points = max(64, min(250_000, int(stage.get("maximum_points") or 60_000)))
        per_sensor_limit = max(32, maximum_points // 2)
        lidar = _bounded(lidar_points, per_sensor_limit)
        world_depth, colors, depth_confidence, anchor = _world_depth(depth_point_cloud, pose)
        valid = depth_confidence >= max(0.0, min(1.0, float(stage.get("minimum_depth_confidence") or 0.1)))
        world_depth = world_depth[valid]
        colors = colors[valid]
        if len(world_depth) > per_sensor_limit:
            indices = np.linspace(0, len(world_depth) - 1, per_sensor_limit, dtype=np.int64)
            world_depth = np.ascontiguousarray(world_depth[indices])
            colors = np.ascontiguousarray(colors[indices])
        if not len(lidar) or not len(world_depth):
            return _error("sensor fusion requires nonempty LiDAR and depth points", device=device)

        maximum_distance = max(0.01, min(5.0, float(stage.get("maximum_alignment_distance_m") or 0.35)))
        candidate_x, candidate_y, candidate_yaw = _candidates(stage)
        lidar_wp = wp.array(lidar, dtype=wp.vec3, device=selected_device)
        depth_wp = wp.array(world_depth, dtype=wp.vec3, device=selected_device)
        colors_wp = wp.array(colors, dtype=wp.vec3, device=selected_device)
        candidate_x_wp = wp.array(candidate_x, dtype=wp.float32, device=selected_device)
        candidate_y_wp = wp.array(candidate_y, dtype=wp.float32, device=selected_device)
        candidate_yaw_wp = wp.array(candidate_yaw, dtype=wp.float32, device=selected_device)
        residual_sums_wp = wp.zeros(len(candidate_x), dtype=wp.float32, device=selected_device)
        matched_counts_wp = wp.zeros(len(candidate_x), dtype=wp.int32, device=selected_device)
        corrected_wp = wp.zeros(len(world_depth), dtype=wp.vec3, device=selected_device)
        display_colors_wp = wp.zeros(len(world_depth), dtype=wp.vec3, device=selected_device)
        residuals_wp = wp.zeros(len(world_depth), dtype=wp.float32, device=selected_device)
        confidence_wp = wp.zeros(len(world_depth), dtype=wp.float32, device=selected_device)
        matched_wp = wp.zeros(len(world_depth), dtype=wp.int32, device=selected_device)
        side = max(8, min(256, int(math.ceil(len(lidar) ** (1.0 / 3.0))) * 4))
        grid = wp.HashGrid(side, side, side, device=selected_device)
        started = time.perf_counter()
        grid.build(lidar_wp, maximum_distance)
        wp.launch(
            _score_calibration_kernel,
            dim=len(candidate_x) * len(world_depth),
            inputs=[
                grid.id, lidar_wp, depth_wp, wp.vec3(*anchor.tolist()),
                candidate_x_wp, candidate_y_wp, candidate_yaw_wp,
                len(world_depth), maximum_distance,
            ],
            outputs=[residual_sums_wp, matched_counts_wp],
            device=selected_device,
        )
        wp.synchronize_device(selected_device)
        residual_sums = residual_sums_wp.numpy()
        matched_counts = matched_counts_wp.numpy()
        hypothesis_scores = (
            residual_sums / np.maximum(matched_counts, 1)
            + (1.0 - matched_counts / max(1, len(world_depth))) * maximum_distance
        )
        best_index = int(np.argmin(hypothesis_scores))
        correction = {
            "x_m": float(candidate_x[best_index]),
            "y_m": float(candidate_y[best_index]),
            "yaw_rad": float(candidate_yaw[best_index]),
            "yaw_deg": math.degrees(float(candidate_yaw[best_index])),
        }
        wp.launch(
            _align_depth_kernel,
            dim=len(world_depth),
            inputs=[
                grid.id, lidar_wp, depth_wp, colors_wp, wp.vec3(*anchor.tolist()),
                correction["x_m"], correction["y_m"], correction["yaw_rad"], maximum_distance,
            ],
            outputs=[corrected_wp, display_colors_wp, residuals_wp, confidence_wp, matched_wp],
            device=selected_device,
        )
        wp.synchronize_device(selected_device)
        corrected = corrected_wp.numpy()
        display_colors = display_colors_wp.numpy()
        residuals = residuals_wp.numpy()
        confidence = confidence_wp.numpy()
        matched = matched_wp.numpy().astype(bool, copy=False)
        pipeline_ms = (time.perf_counter() - started) * 1000.0
    except Exception as exc:
        return _error(f"Warp sensor fusion failed ({type(exc).__name__}: {exc})", device=device)

    matched_residuals = residuals[matched]
    matched_count = int(np.count_nonzero(matched))
    mean_residual = float(np.mean(matched_residuals)) if matched_count else maximum_distance
    rms_residual = float(np.sqrt(np.mean(matched_residuals ** 2))) if matched_count else maximum_distance
    p95_residual = float(np.percentile(matched_residuals, 95)) if matched_count else maximum_distance
    mean_confidence = float(np.mean(confidence)) if len(confidence) else 0.0
    lidar_colors = np.tile(np.asarray([[0.05, 0.88, 1.0]], dtype=np.float32), (len(lidar), 1))
    fused_points = np.concatenate((lidar, corrected), axis=0)
    fused_colors = np.concatenate((lidar_colors, display_colors), axis=0)

    cpu_ms = 0.0
    cpu_best_index = -1
    correction_error = 0.0
    comparison_limited = False
    if stage.get("compare_cpu"):
        cpu_lidar = lidar[:1_024]
        cpu_depth = world_depth[:1_024]
        comparison_limited = len(lidar) > len(cpu_lidar) or len(world_depth) > len(cpu_depth)
        cpu_started = time.perf_counter()
        cpu_best_index, _, _ = _cpu_hypotheses(
            cpu_lidar, cpu_depth, anchor, candidate_x, candidate_y, candidate_yaw, maximum_distance,
        )
        cpu_ms = (time.perf_counter() - cpu_started) * 1000.0
        correction_error = float(math.sqrt(
            (float(candidate_x[cpu_best_index]) - correction["x_m"]) ** 2
            + (float(candidate_y[cpu_best_index]) - correction["y_m"]) ** 2
            + (float(candidate_yaw[cpu_best_index]) - correction["yaw_rad"]) ** 2
        ))

    report = {
        "state": "ready",
        "backend": "warp-hash-grid",
        "device": str(selected_device),
        "lidar_points": int(len(lidar)),
        "depth_points": int(len(world_depth)),
        "fused_points": int(len(fused_points)),
        "matched_points": matched_count,
        "unmatched_points": int(len(world_depth) - matched_count),
        "matched_ratio": float(matched_count / max(1, len(world_depth))),
        "mean_residual_m": mean_residual,
        "rms_residual_m": rms_residual,
        "p95_residual_m": p95_residual,
        "maximum_alignment_distance_m": maximum_distance,
        "mean_confidence": mean_confidence,
        "calibration_hypotheses": int(len(candidate_x)),
        "calibration_work_items": int(len(candidate_x) * len(world_depth)),
        "best_hypothesis": best_index,
        "best_score": float(hypothesis_scores[best_index]),
        "correction": correction,
        "pipeline_ms": float(pipeline_ms),
        "cpu_ms": float(cpu_ms),
        "speedup": float(cpu_ms / pipeline_ms) if cpu_ms > 0.0 and pipeline_ms > 0.0 else 0.0,
        "cpu_best_hypothesis": cpu_best_index,
        "correction_error": correction_error,
        "comparison_limited": comparison_limited,
    }
    return {
        "ok": True,
        "fused_points": fused_points.astype(float).tolist(),
        "fused_colors": fused_colors.astype(float).tolist(),
        "lidar_points": lidar.astype(float).tolist(),
        "lidar_colors": lidar_colors.astype(float).tolist(),
        "depth_points": corrected.astype(float).tolist(),
        "depth_colors": display_colors.astype(float).tolist(),
        "alignment_residuals": residuals.astype(float).tolist(),
        "alignment_confidence": confidence.astype(float).tolist(),
        "alignment_matched": matched.astype(bool).tolist(),
        "sensor_origin": anchor.astype(float).tolist(),
        "device": str(selected_device),
        "kernel_ms": float(pipeline_ms),
        "report": report,
    }
