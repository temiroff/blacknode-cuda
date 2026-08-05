"""Parallel bounded trajectory evaluation backed by NVIDIA Warp."""
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
except Exception:  # pragma: no cover - package discovery must remain available
    wp = None


if wp is not None:

    @wp.kernel
    def _score_trajectory_candidates_kernel(
        candidates: wp.array(dtype=wp.vec2),
        static_grid: wp.uint64,
        static_points: wp.array(dtype=wp.vec3),
        static_count: wp.int32,
        dynamic_grid: wp.uint64,
        dynamic_points: wp.array(dtype=wp.vec3),
        dynamic_velocities: wp.array(dtype=wp.vec3),
        dynamic_count: wp.int32,
        start_x: wp.float32,
        start_y: wp.float32,
        start_yaw: wp.float32,
        goal_x: wp.float32,
        goal_y: wp.float32,
        time_step_s: wp.float32,
        time_steps: wp.int32,
        robot_radius: wp.float32,
        clearance_radius: wp.float32,
        dynamic_query_radius: wp.float32,
        maximum_linear_speed: wp.float32,
        maximum_angular_speed: wp.float32,
        start_goal_distance: wp.float32,
        scores: wp.array(dtype=wp.float32),
        unsafe_flags: wp.array(dtype=wp.int32),
        terminal_distances: wp.array(dtype=wp.float32),
        minimum_clearances: wp.array(dtype=wp.float32),
    ):
        candidate_index = wp.tid()
        candidate = candidates[candidate_index]
        linear_speed = candidate[0]
        angular_speed = candidate[1]
        x = start_x
        y = start_y
        yaw = start_yaw
        minimum_clearance = clearance_radius
        unsafe = wp.int32(0)

        for step_index in range(time_steps):
            yaw = yaw + angular_speed * time_step_s
            x = x + wp.cos(yaw) * linear_speed * time_step_s
            y = y + wp.sin(yaw) * linear_speed * time_step_s
            elapsed = wp.float32(step_index + 1) * time_step_s
            query_point = wp.vec3(x, y, 0.0)

            if static_count > 0:
                for obstacle_index in wp.hash_grid_query(
                    static_grid,
                    query_point,
                    clearance_radius,
                ):
                    if obstacle_index < static_count:
                        distance = wp.length(static_points[obstacle_index] - query_point)
                        minimum_clearance = wp.min(minimum_clearance, distance)
                        if distance <= robot_radius:
                            unsafe = wp.int32(1)

            if dynamic_count > 0:
                for obstacle_index in wp.hash_grid_query(
                    dynamic_grid,
                    query_point,
                    dynamic_query_radius,
                ):
                    if obstacle_index < dynamic_count:
                        predicted = (
                            dynamic_points[obstacle_index]
                            + dynamic_velocities[obstacle_index] * elapsed
                        )
                        distance = wp.length(predicted - query_point)
                        minimum_clearance = wp.min(minimum_clearance, distance)
                        if distance <= robot_radius:
                            unsafe = wp.int32(1)

        goal_dx = goal_x - x
        goal_dy = goal_y - y
        terminal_distance = wp.sqrt(goal_dx * goal_dx + goal_dy * goal_dy)
        safe_clearance = wp.clamp(
            (minimum_clearance - robot_radius)
            / wp.max(wp.float32(1.0e-5), clearance_radius - robot_radius),
            0.0,
            1.0,
        )
        progress = wp.clamp(
            (start_goal_distance - terminal_distance)
            / wp.max(start_goal_distance, wp.float32(1.0e-5)),
            -1.0,
            1.0,
        )
        goal_score = wp.float32(1.0) / (wp.float32(1.0) + terminal_distance)
        speed_score = linear_speed / wp.max(maximum_linear_speed, wp.float32(1.0e-5))
        turn_cost = wp.abs(angular_speed) / wp.max(maximum_angular_speed, wp.float32(1.0e-5))
        score = (
            progress * wp.float32(3.0)
            + goal_score * wp.float32(2.0)
            + safe_clearance * wp.float32(1.5)
            + speed_score * wp.float32(0.25)
            - turn_cost * wp.float32(0.25)
            - wp.float32(unsafe) * wp.float32(8.0)
        )
        scores[candidate_index] = score
        unsafe_flags[candidate_index] = unsafe
        terminal_distances[candidate_index] = terminal_distance
        minimum_clearances[candidate_index] = minimum_clearance


def trajectory_candidates(
    count: int,
    maximum_linear_speed_mps: float,
    maximum_angular_speed_rps: float,
) -> Any:
    """Generate a deterministic low-discrepancy set of forward unicycle controls."""
    if np is None:
        raise RuntimeError("NumPy is required for trajectory evaluation")
    candidate_count = max(1, int(count))
    indices = np.arange(candidate_count, dtype=np.float64)
    linear_fraction = (indices + 0.5) / float(candidate_count)
    angular_fraction = np.mod(indices * 0.6180339887498949, 1.0)
    values = np.column_stack((
        linear_fraction * float(maximum_linear_speed_mps),
        (angular_fraction * 2.0 - 1.0) * float(maximum_angular_speed_rps),
    )).astype(np.float32)
    values[0] = [0.0, 0.0]
    if candidate_count > 1:
        values[1] = [float(maximum_linear_speed_mps), 0.0]
    return values


def integrate_candidate_path(
    start_pose: Any,
    candidate: Any,
    *,
    horizon_s: float,
    time_steps: int,
) -> Any:
    """Integrate one differential-drive arc into map-frame XYZ samples."""
    if np is None:
        raise RuntimeError("NumPy is required for trajectory evaluation")
    pose = np.asarray(start_pose, dtype=np.float64)
    control = np.asarray(candidate, dtype=np.float64)
    step_count = max(1, int(time_steps))
    dt_s = float(horizon_s) / step_count
    path = np.zeros((step_count + 1, 3), dtype=np.float32)
    path[0, :2] = pose[:2]
    x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    for index in range(1, step_count + 1):
        yaw += float(control[1]) * dt_s
        x += math.cos(yaw) * float(control[0]) * dt_s
        y += math.sin(yaw) * float(control[0]) * dt_s
        path[index, 0] = x
        path[index, 1] = y
    return path


def evaluate_trajectories_cpu(
    candidates: Any,
    static_points: Any,
    dynamic_points: Any,
    dynamic_velocities: Any,
    start_pose: Any,
    goal_xy: Any,
    *,
    horizon_s: float,
    time_steps: int,
    robot_radius_m: float,
    clearance_margin_m: float,
    maximum_linear_speed_mps: float,
    maximum_angular_speed_rps: float,
) -> tuple[Any, Any, Any, Any]:
    """NumPy reference for correctness tests and opt-in equal-work comparison."""
    if np is None:
        raise RuntimeError("NumPy is required for trajectory evaluation")
    controls = np.asarray(candidates, dtype=np.float32)
    fixed = np.asarray(static_points, dtype=np.float32).reshape((-1, 3))
    moving = np.asarray(dynamic_points, dtype=np.float32).reshape((-1, 3))
    velocities = np.asarray(dynamic_velocities, dtype=np.float32).reshape((-1, 3))
    pose = np.asarray(start_pose, dtype=np.float32)
    goal = np.asarray(goal_xy, dtype=np.float32)
    step_count = max(1, int(time_steps))
    dt_s = float(horizon_s) / step_count
    clearance_radius = float(robot_radius_m) + float(clearance_margin_m)
    start_goal_distance = max(1.0e-5, float(np.linalg.norm(goal[:2] - pose[:2])))
    scores = np.zeros(len(controls), dtype=np.float32)
    unsafe = np.zeros(len(controls), dtype=np.int32)
    terminal = np.zeros(len(controls), dtype=np.float32)
    clearances = np.full(len(controls), clearance_radius, dtype=np.float32)
    for candidate_index, control in enumerate(controls):
        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        minimum_clearance = clearance_radius
        collision = False
        for step_index in range(step_count):
            yaw += float(control[1]) * dt_s
            x += math.cos(yaw) * float(control[0]) * dt_s
            y += math.sin(yaw) * float(control[0]) * dt_s
            query = np.asarray([x, y, 0.0], dtype=np.float32)
            if len(fixed):
                distance = float(np.min(np.linalg.norm(fixed - query, axis=1)))
                minimum_clearance = min(minimum_clearance, distance)
                collision = collision or distance <= robot_radius_m
            if len(moving):
                predicted = moving + velocities * ((step_index + 1) * dt_s)
                distance = float(np.min(np.linalg.norm(predicted - query, axis=1)))
                minimum_clearance = min(minimum_clearance, distance)
                collision = collision or distance <= robot_radius_m
        terminal_distance = math.hypot(float(goal[0]) - x, float(goal[1]) - y)
        safe_clearance = max(0.0, min(1.0, (
            minimum_clearance - robot_radius_m
        ) / max(1.0e-5, clearance_radius - robot_radius_m)))
        progress = max(-1.0, min(1.0, (
            start_goal_distance - terminal_distance
        ) / start_goal_distance))
        goal_score = 1.0 / (1.0 + terminal_distance)
        speed_score = float(control[0]) / max(1.0e-5, maximum_linear_speed_mps)
        turn_cost = abs(float(control[1])) / max(1.0e-5, maximum_angular_speed_rps)
        scores[candidate_index] = (
            progress * 3.0 + goal_score * 2.0 + safe_clearance * 1.5
            + speed_score * 0.25 - turn_cost * 0.25 - (8.0 if collision else 0.0)
        )
        unsafe[candidate_index] = int(collision)
        terminal[candidate_index] = terminal_distance
        clearances[candidate_index] = minimum_clearance
    return scores, unsafe, terminal, clearances


class WarpTrajectoryEvaluator:
    """Score bounded trajectory controls in parallel against static and moving obstacles."""

    def __init__(self, *, device: str) -> None:
        if wp is None:
            raise RuntimeError("Warp is unavailable; install warp-lang>=1.15")
        if np is None:
            raise RuntimeError("NumPy is unavailable; install numpy>=1.24")
        self.device = wp.get_device(device)
        self.revision = 0

    @staticmethod
    def _grid(points_wp: Any, point_count: int, device: Any, radius: float) -> Any:
        side = max(16, min(256, int(math.ceil(math.sqrt(max(1, point_count)))) * 2))
        grid = wp.HashGrid(side, side, 1, device=device)
        grid.build(points_wp, max(0.01, float(radius)))
        return grid

    def evaluate(
        self,
        static_points: Any,
        dynamic_points: Any,
        dynamic_velocities: Any,
        start_pose: Any,
        goal_xy: Any,
        *,
        trajectory_count: int,
        time_steps: int,
        horizon_s: float,
        maximum_linear_speed_mps: float,
        maximum_angular_speed_rps: float,
        robot_radius_m: float,
        clearance_margin_m: float,
        display_trajectories: int,
        compare_cpu: bool,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        controls = trajectory_candidates(
            trajectory_count,
            maximum_linear_speed_mps,
            maximum_angular_speed_rps,
        )
        fixed = np.asarray(static_points, dtype=np.float32).reshape((-1, 3))
        moving = np.asarray(dynamic_points, dtype=np.float32).reshape((-1, 3))
        velocities = np.asarray(dynamic_velocities, dtype=np.float32).reshape((-1, 3))
        if len(velocities) != len(moving):
            velocities = np.zeros_like(moving)
        fixed_upload = fixed if len(fixed) else np.asarray([[1.0e6, 1.0e6, 0.0]], dtype=np.float32)
        moving_upload = moving if len(moving) else np.asarray([[1.0e6, 1.0e6, 0.0]], dtype=np.float32)
        velocity_upload = velocities if len(moving) else np.zeros((1, 3), dtype=np.float32)
        controls_wp = wp.array(controls, dtype=wp.vec2, device=self.device)
        fixed_wp = wp.array(fixed_upload, dtype=wp.vec3, device=self.device)
        moving_wp = wp.array(moving_upload, dtype=wp.vec3, device=self.device)
        velocities_wp = wp.array(velocity_upload, dtype=wp.vec3, device=self.device)
        clearance_radius = float(robot_radius_m) + float(clearance_margin_m)
        static_grid = self._grid(fixed_wp, len(fixed), self.device, clearance_radius)
        maximum_dynamic_speed = (
            float(np.max(np.linalg.norm(velocities, axis=1))) if len(velocities) else 0.0
        )
        dynamic_query_radius = clearance_radius + maximum_dynamic_speed * float(horizon_s)
        dynamic_grid = self._grid(moving_wp, len(moving), self.device, dynamic_query_radius)
        upload_ms = (time.perf_counter() - started) * 1000.0

        scores_wp = wp.zeros(len(controls), dtype=wp.float32, device=self.device)
        unsafe_wp = wp.zeros(len(controls), dtype=wp.int32, device=self.device)
        terminal_wp = wp.zeros(len(controls), dtype=wp.float32, device=self.device)
        clearance_wp = wp.zeros(len(controls), dtype=wp.float32, device=self.device)
        pose = np.asarray(start_pose, dtype=np.float32)
        goal = np.asarray(goal_xy, dtype=np.float32)
        start_goal_distance = max(1.0e-5, float(np.linalg.norm(goal[:2] - pose[:2])))
        kernel_started = time.perf_counter()
        wp.launch(
            _score_trajectory_candidates_kernel,
            dim=len(controls),
            inputs=[
                controls_wp,
                static_grid.id, fixed_wp, len(fixed),
                dynamic_grid.id, moving_wp, velocities_wp, len(moving),
                float(pose[0]), float(pose[1]), float(pose[2]),
                float(goal[0]), float(goal[1]),
                float(horizon_s) / max(1, int(time_steps)), int(time_steps),
                float(robot_radius_m), clearance_radius, dynamic_query_radius,
                float(maximum_linear_speed_mps), float(maximum_angular_speed_rps),
                start_goal_distance,
                scores_wp, unsafe_wp, terminal_wp, clearance_wp,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        scores = scores_wp.numpy()
        unsafe = unsafe_wp.numpy().astype(bool, copy=False)
        terminal = terminal_wp.numpy()
        clearances = clearance_wp.numpy()
        kernel_ms = (time.perf_counter() - kernel_started) * 1000.0
        pipeline_ms = (time.perf_counter() - started) * 1000.0

        cpu_ms = 0.0
        maximum_error = 0.0
        if compare_cpu:
            cpu_started = time.perf_counter()
            cpu_scores, cpu_unsafe, cpu_terminal, cpu_clearances = evaluate_trajectories_cpu(
                controls, fixed, moving, velocities, pose, goal,
                horizon_s=horizon_s,
                time_steps=time_steps,
                robot_radius_m=robot_radius_m,
                clearance_margin_m=clearance_margin_m,
                maximum_linear_speed_mps=maximum_linear_speed_mps,
                maximum_angular_speed_rps=maximum_angular_speed_rps,
            )
            cpu_ms = (time.perf_counter() - cpu_started) * 1000.0
            maximum_error = max(
                float(np.max(np.abs(scores - cpu_scores))) if len(scores) else 0.0,
                float(np.max(np.abs(terminal - cpu_terminal))) if len(terminal) else 0.0,
                float(np.max(np.abs(clearances - cpu_clearances))) if len(clearances) else 0.0,
                float(np.max(np.abs(unsafe.astype(np.int32) - cpu_unsafe))) if len(unsafe) else 0.0,
            )

        safe_indices = np.flatnonzero(~unsafe)
        best_index = int(safe_indices[np.argmax(scores[safe_indices])]) if len(safe_indices) else int(np.argmax(scores))
        display_count = max(1, min(len(controls), int(display_trajectories)))
        safe_ranked = safe_indices[np.argsort(scores[safe_indices])[::-1]] if len(safe_indices) else safe_indices
        unsafe_indices = np.flatnonzero(unsafe)
        safe_capacity = min(len(safe_ranked), max(1, display_count * 2 // 3))
        selected = list(safe_ranked[:safe_capacity].astype(int))
        remaining = display_count - len(selected)
        if remaining > 0 and len(unsafe_indices):
            positions = np.linspace(0, len(unsafe_indices) - 1, min(remaining, len(unsafe_indices)), dtype=np.int32)
            selected.extend(unsafe_indices[positions].astype(int).tolist())
        if len(selected) < display_count:
            for candidate_index in np.argsort(scores)[::-1].astype(int):
                if candidate_index not in selected:
                    selected.append(candidate_index)
                if len(selected) >= display_count:
                    break
        if best_index not in selected:
            if len(selected) >= display_count:
                selected[-1] = best_index
            else:
                selected.append(best_index)
        selected = selected[:display_count]
        score_min = float(np.min(scores))
        score_span = max(1.0e-6, float(np.max(scores)) - score_min)
        paths = [
            integrate_candidate_path(
                pose,
                controls[index],
                horizon_s=horizon_s,
                time_steps=time_steps,
            ).astype(float).tolist()
            for index in selected
        ]
        display_best_index = selected.index(best_index)
        self.revision += 1
        return {
            "state": "ready",
            "backend": "warp",
            "device": str(self.device),
            "dtype": "float32",
            "warmup": self.revision == 1,
            "revision": int(self.revision),
            "trajectory_count": int(len(controls)),
            "time_steps": int(time_steps),
            "work_items": int(len(controls) * int(time_steps)),
            "safe_trajectories": int(np.count_nonzero(~unsafe)),
            "unsafe_trajectories": int(np.count_nonzero(unsafe)),
            "display_trajectories": int(len(selected)),
            "pipeline_ms": float(pipeline_ms),
            "upload_ms": float(upload_ms),
            "kernel_ms": float(kernel_ms),
            "cpu_ms": float(cpu_ms),
            "speedup": float(cpu_ms / pipeline_ms) if cpu_ms > 0.0 and pipeline_ms > 0.0 else 0.0,
            "max_error": float(maximum_error),
            "static_obstacles": int(len(fixed)),
            "dynamic_obstacles": int(len(moving)),
            "goal": [float(goal[0]), float(goal[1]), 0.0],
            "best_score": float(scores[best_index]),
            "best_terminal_distance_m": float(terminal[best_index]),
            "best_minimum_clearance_m": float(clearances[best_index]),
            "best_candidate": {
                "linear_speed_mps": float(controls[best_index, 0]),
                "angular_speed_rps": float(controls[best_index, 1]),
                "safe": bool(not unsafe[best_index]),
                "commands_motion": False,
            },
            "best_display_index": int(display_best_index),
            "paths": paths,
            "path_scores": [float((scores[index] - score_min) / score_span) for index in selected],
            "path_safe": [bool(not unsafe[index]) for index in selected],
        }
