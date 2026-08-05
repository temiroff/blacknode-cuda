"""Warp hash-grid motion classification for pose-registered sensor returns."""
from __future__ import annotations

import time
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - package dependency diagnostics own this path
    np = None

try:
    import warp as wp
except Exception:  # pragma: no cover - package dependency diagnostics own this path
    wp = None


if wp is not None:

    @wp.kernel
    def _classify_motion_kernel(
        grid: wp.uint64,
        previous: wp.array(dtype=wp.vec3),
        current: wp.array(dtype=wp.vec3),
        stable_radius: wp.float32,
        tracking_radius: wp.float32,
        inverse_dt: wp.float32,
        minimum_speed: wp.float32,
        velocities: wp.array(dtype=wp.vec3),
        motion_scores: wp.array(dtype=wp.float32),
        dynamic_flags: wp.array(dtype=wp.int32),
        matched_flags: wp.array(dtype=wp.int32),
    ):
        point_index = wp.tid()
        point = current[point_index]
        tracking_radius_sq = tracking_radius * tracking_radius
        nearest_distance_sq = tracking_radius_sq + wp.float32(1.0)
        nearest_index = wp.int32(-1)

        for candidate_index in wp.hash_grid_query(grid, point, tracking_radius):
            delta = point - previous[candidate_index]
            distance_sq = wp.dot(delta, delta)
            if distance_sq < nearest_distance_sq:
                nearest_distance_sq = distance_sq
                nearest_index = candidate_index

        velocity = wp.vec3(0.0, 0.0, 0.0)
        score = wp.float32(0.0)
        moving = wp.int32(0)
        matched = wp.int32(0)
        if nearest_index >= 0:
            matched = wp.int32(1)
            delta = point - previous[nearest_index]
            distance = wp.sqrt(nearest_distance_sq)
            velocity = delta * inverse_dt
            speed = wp.length(velocity)
            radius_span = wp.max(wp.float32(1.0e-5), tracking_radius - stable_radius)
            residual_score = wp.clamp((distance - stable_radius) / radius_span, 0.0, 1.0)
            maximum_speed = wp.max(minimum_speed + wp.float32(1.0e-5), tracking_radius * inverse_dt)
            speed_score = wp.clamp(
                (speed - minimum_speed) / (maximum_speed - minimum_speed),
                0.0,
                1.0,
            )
            score = residual_score * wp.float32(0.65) + speed_score * wp.float32(0.35)
            motion_distance = wp.max(wp.float32(0.01), stable_radius * wp.float32(0.35))
            if distance > stable_radius or (distance > motion_distance and speed >= minimum_speed):
                moving = wp.int32(1)
        else:
            # A return with no spatial predecessor is new or newly revealed.
            # Keep it out of the persistent map until a later scan confirms it.
            score = wp.float32(0.8)
            moving = wp.int32(1)

        velocities[point_index] = velocity
        motion_scores[point_index] = score
        dynamic_flags[point_index] = moving
        matched_flags[point_index] = matched


def classify_motion_cpu(
    previous: Any,
    current: Any,
    *,
    stable_radius_m: float,
    tracking_radius_m: float,
    dt_s: float,
    minimum_speed_mps: float,
) -> tuple[Any, Any, Any]:
    """Equivalent NumPy reference used by tests and opt-in comparisons."""
    if np is None:
        raise RuntimeError("NumPy is required for dynamic occupancy")
    previous_values = np.asarray(previous, dtype=np.float32)[:, :3]
    current_values = np.asarray(current, dtype=np.float32)[:, :3]
    velocities = np.zeros_like(current_values)
    scores = np.zeros(len(current_values), dtype=np.float32)
    flags = np.zeros(len(current_values), dtype=np.int32)
    inverse_dt = 1.0 / max(1.0e-4, float(dt_s))
    tracking_radius_sq = float(tracking_radius_m) ** 2
    radius_span = max(1.0e-5, float(tracking_radius_m) - float(stable_radius_m))
    maximum_speed = max(float(minimum_speed_mps) + 1.0e-5, float(tracking_radius_m) * inverse_dt)
    for start in range(0, len(current_values), 256):
        batch = current_values[start:start + 256]
        distance_sq = np.sum(
            (batch[:, None, :] - previous_values[None, :, :]) ** 2,
            axis=2,
        )
        nearest_indices = np.argmin(distance_sq, axis=1)
        nearest_sq = distance_sq[np.arange(len(batch)), nearest_indices]
        matched = nearest_sq <= tracking_radius_sq
        delta = batch - previous_values[nearest_indices]
        batch_velocity = delta * inverse_dt
        speed = np.linalg.norm(batch_velocity, axis=1)
        distance = np.sqrt(nearest_sq)
        residual_score = np.clip((distance - stable_radius_m) / radius_span, 0.0, 1.0)
        speed_score = np.clip(
            (speed - minimum_speed_mps) / (maximum_speed - minimum_speed_mps),
            0.0,
            1.0,
        )
        stop = start + len(batch)
        velocities[start:stop] = np.where(matched[:, None], batch_velocity, 0.0)
        scores[start:stop] = np.where(matched, residual_score * 0.65 + speed_score * 0.35, 0.0)
        motion_distance = max(0.01, float(stable_radius_m) * 0.35)
        flags[start:stop] = (
            (~matched)
            | (
                matched
                & (
                    (distance > stable_radius_m)
                    | ((distance > motion_distance) & (speed >= minimum_speed_mps))
                )
            )
        ).astype(np.int32)
        scores[start:stop] = np.where(matched, scores[start:stop], 0.8)
    return velocities, scores, flags


class WarpDynamicOccupancyTracker:
    """Keep one registered scan and classify motion with a device hash grid."""

    def __init__(self, *, device: str, maximum_points: int = 65_536) -> None:
        if wp is None:
            raise RuntimeError("Warp is unavailable; install warp-lang>=1.15")
        if np is None:
            raise RuntimeError("NumPy is unavailable; install numpy>=1.24")
        self.device = wp.get_device(device)
        self.maximum_points = max(64, min(65_536, int(maximum_points)))
        self.previous_points: Any | None = None
        self.previous_time_ns = 0
        self.revision = 0

    def clear(self) -> None:
        self.previous_points = None
        self.previous_time_ns = 0
        self.revision += 1

    def update(
        self,
        world_points: Any,
        time_ns: int,
        *,
        stable_radius_m: float,
        tracking_radius_m: float,
        minimum_speed_mps: float,
        maximum_age_s: float,
        display_points: int,
        compare_cpu: bool = False,
    ) -> dict[str, Any]:
        values = np.asarray(world_points, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] < 3 or len(values) == 0:
            return self._waiting("waiting")
        values = np.ascontiguousarray(values[:self.maximum_points, :3])
        previous = self.previous_points
        previous_time_ns = self.previous_time_ns
        self.revision += 1
        if previous is None or len(previous) == 0 or int(time_ns) <= previous_time_ns:
            self.previous_points = values.copy()
            self.previous_time_ns = int(time_ns)
            return self._waiting("warming", point_count=len(values))

        dt_s = (int(time_ns) - previous_time_ns) / 1_000_000_000.0
        if dt_s <= 0.0 or dt_s > max(0.05, float(maximum_age_s)):
            self.previous_points = values.copy()
            self.previous_time_ns = int(time_ns)
            return self._waiting("warming", point_count=len(values), dt_s=dt_s)

        stable_radius = max(0.01, float(stable_radius_m))
        tracking_radius = max(stable_radius + 0.01, float(tracking_radius_m))
        minimum_speed = max(0.0, float(minimum_speed_mps))
        reference_values = values
        reference_interval_s = max(0.12, min(0.4, float(maximum_age_s) * 0.75))
        comparison_limited = bool(compare_cpu and (len(previous) > 1_024 or len(values) > 1_024))
        if compare_cpu:
            previous = np.ascontiguousarray(previous[:1_024])
            values = np.ascontiguousarray(values[:1_024])

        previous_wp = wp.array(previous, dtype=wp.vec3, device=self.device)
        current_wp = wp.array(values, dtype=wp.vec3, device=self.device)
        grid = wp.HashGrid(128, 128, 1, device=self.device)
        velocities_wp = wp.zeros(len(values), dtype=wp.vec3, device=self.device)
        scores_wp = wp.zeros(len(values), dtype=wp.float32, device=self.device)
        flags_wp = wp.zeros(len(values), dtype=wp.int32, device=self.device)
        matched_wp = wp.zeros(len(values), dtype=wp.int32, device=self.device)
        started = time.perf_counter()
        grid.build(previous_wp, tracking_radius)
        wp.launch(
            _classify_motion_kernel,
            dim=len(values),
            inputs=[
                grid.id,
                previous_wp,
                current_wp,
                stable_radius,
                tracking_radius,
                1.0 / dt_s,
                minimum_speed,
                velocities_wp,
                scores_wp,
                flags_wp,
                matched_wp,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        velocities = velocities_wp.numpy()
        scores = scores_wp.numpy()
        flags = flags_wp.numpy().astype(bool, copy=False)
        matched = matched_wp.numpy().astype(bool, copy=False)
        pipeline_ms = (time.perf_counter() - started) * 1000.0

        if dt_s >= reference_interval_s:
            self.previous_points = reference_values.copy()
            self.previous_time_ns = int(time_ns)

        cpu_ms = 0.0
        maximum_error = 0.0
        if compare_cpu:
            cpu_started = time.perf_counter()
            cpu_velocities, cpu_scores, cpu_flags = classify_motion_cpu(
                previous,
                values,
                stable_radius_m=stable_radius,
                tracking_radius_m=tracking_radius,
                dt_s=dt_s,
                minimum_speed_mps=minimum_speed,
            )
            cpu_ms = (time.perf_counter() - cpu_started) * 1000.0
            maximum_error = max(
                float(np.max(np.abs(velocities - cpu_velocities))) if len(values) else 0.0,
                float(np.max(np.abs(scores - cpu_scores))) if len(values) else 0.0,
                float(np.max(np.abs(flags.astype(np.int32) - cpu_flags))) if len(values) else 0.0,
            )

        dynamic_indices = np.flatnonzero(flags)
        static_flags = matched & ~flags
        display_capacity = max(16, min(4_000, int(display_points)))
        if len(dynamic_indices) > display_capacity:
            selection = np.linspace(0, len(dynamic_indices) - 1, display_capacity, dtype=np.int32)
            display_indices = dynamic_indices[selection]
        else:
            display_indices = dynamic_indices
        speed = np.linalg.norm(velocities[display_indices], axis=1) if len(display_indices) else np.empty(0)
        return {
            "state": "ready",
            "backend": "warp-hash-grid",
            "device": str(self.device),
            "input_points": int(len(values)),
            "reference_points": int(len(previous)),
            "dynamic_points": int(len(dynamic_indices)),
            "display_points": int(len(display_indices)),
            "pipeline_ms": float(pipeline_ms),
            "cpu_ms": float(cpu_ms),
            "speedup": float(cpu_ms / pipeline_ms) if cpu_ms > 0.0 and pipeline_ms > 0.0 else 0.0,
            "max_error": float(maximum_error),
            "comparison_limited": comparison_limited,
            "dt_s": float(dt_s),
            "reference_interval_s": float(reference_interval_s),
            "stable_radius_m": stable_radius,
            "tracking_radius_m": tracking_radius,
            "minimum_speed_mps": minimum_speed,
            "mean_speed_mps": float(np.mean(speed)) if len(speed) else 0.0,
            "max_speed_mps": float(np.max(speed)) if len(speed) else 0.0,
            "trail_distance_limit_m": float(min(0.3, max(stable_radius * 2.0, tracking_radius * 0.65))),
            "revision": int(self.revision),
            "_dynamic_mask": flags,
            "_static_mask": static_flags,
            "points": values[display_indices].astype(float).tolist(),
            "velocities": velocities[display_indices].astype(float).tolist(),
            "scores": scores[display_indices].astype(float).tolist(),
        }

    def _waiting(
        self,
        state: str,
        *,
        point_count: int = 0,
        dt_s: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "state": state,
            "backend": "warp-hash-grid",
            "device": str(self.device),
            "input_points": int(point_count),
            "reference_points": 0,
            "dynamic_points": 0,
            "display_points": 0,
            "pipeline_ms": 0.0,
            "cpu_ms": 0.0,
            "speedup": 0.0,
            "max_error": 0.0,
            "comparison_limited": False,
            "dt_s": float(dt_s),
            "mean_speed_mps": 0.0,
            "max_speed_mps": 0.0,
            "revision": int(self.revision),
            "_dynamic_mask": np.zeros(point_count, dtype=bool),
            "_static_mask": np.zeros(point_count, dtype=bool),
            "points": [],
            "velocities": [],
            "scores": [],
        }
