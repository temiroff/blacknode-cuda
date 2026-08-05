"""GPU-resident correlative LaserScan matcher backed by Warp."""
from __future__ import annotations

import math
import time
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - package dependency diagnostics own this path
    np = None

try:
    import warp as wp
except Exception:  # pragma: no cover - package must still discover without Warp
    wp = None


if wp is not None:
    @wp.kernel
    def _score_pose_candidates_kernel(
        local_points: wp.array(dtype=wp.vec3),
        candidates: wp.array(dtype=wp.vec3),
        occupied: wp.array(dtype=wp.uint8),
        origin_cell_x: wp.int32,
        origin_cell_y: wp.int32,
        grid_width: wp.int32,
        grid_height: wp.int32,
        inverse_resolution: wp.float32,
        scores: wp.array(dtype=wp.float32),
    ):
        candidate_index = wp.tid()
        pose = candidates[candidate_index]
        cosine = wp.cos(pose[2])
        sine = wp.sin(pose[2])
        hits = wp.int32(0)
        point_count = local_points.shape[0]
        for point_index in range(point_count):
            point = local_points[point_index]
            world_x = cosine * point[0] - sine * point[1] + pose[0]
            world_y = sine * point[0] + cosine * point[1] + pose[1]
            cell_x = wp.int32(wp.floor(world_x * inverse_resolution + 0.5)) - origin_cell_x
            cell_y = wp.int32(wp.floor(world_y * inverse_resolution + 0.5)) - origin_cell_y
            if cell_x >= 0 and cell_x < grid_width and cell_y >= 0 and cell_y < grid_height:
                if occupied[cell_y * grid_width + cell_x] != wp.uint8(0):
                    hits += 1
        scores[candidate_index] = wp.float32(hits) / wp.float32(wp.max(point_count, 1))


def available(device: str) -> bool:
    if wp is None or np is None or not str(device).startswith("cuda"):
        return False
    try:
        return bool(wp.get_device(device).is_cuda)
    except Exception:
        return False


class WarpCorrelativeMatcher:
    """Score all pose candidates in parallel against one cached reference grid."""

    def __init__(
        self,
        reference_points: Any,
        *,
        resolution: float,
        linear_window: float,
        device: str,
        maximum_grid_cells: int = 4_000_000,
    ) -> None:
        if not available(device):
            raise RuntimeError(f"Warp CUDA matcher is unavailable on {device}")
        reference = np.asarray(reference_points, dtype=np.float32)
        if reference.ndim != 2 or reference.shape[1] < 2 or len(reference) == 0:
            raise ValueError("Warp matcher requires non-empty XY reference points")
        self.device = wp.get_device(device)
        self.resolution = max(1.0e-4, float(resolution))
        reference_cells = np.rint(reference[:, :2] / self.resolution).astype(np.int32)
        padding = max(2, int(math.ceil(float(linear_window) / self.resolution)) + 3)
        minimum = reference_cells.min(axis=0) - padding
        maximum = reference_cells.max(axis=0) + padding
        self.origin_cell_x = int(minimum[0])
        self.origin_cell_y = int(minimum[1])
        self.grid_width = int(maximum[0] - minimum[0] + 1)
        self.grid_height = int(maximum[1] - minimum[1] + 1)
        grid_cells = self.grid_width * self.grid_height
        if grid_cells > int(maximum_grid_cells):
            raise ValueError(f"Warp matcher grid would require {grid_cells:,} cells")
        occupied = np.zeros(grid_cells, dtype=np.uint8)
        local_x = reference_cells[:, 0] - self.origin_cell_x
        local_y = reference_cells[:, 1] - self.origin_cell_y
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                x = local_x + offset_x
                y = local_y + offset_y
                valid = (x >= 0) & (x < self.grid_width) & (y >= 0) & (y < self.grid_height)
                occupied[y[valid] * self.grid_width + x[valid]] = 1
        self.occupied = wp.array(occupied, dtype=wp.uint8, device=self.device)
        self.last_kernel_ms = 0.0
        self.last_particle_pipeline_ms = 0.0

    @staticmethod
    def _candidates(center: Any, linear_step: float, angular_step: float, radius: int) -> Any:
        values = []
        for x_index in range(-radius, radius + 1):
            for y_index in range(-radius, radius + 1):
                for yaw_index in range(-radius, radius + 1):
                    yaw = float(center[2]) + yaw_index * angular_step
                    values.append([
                        float(center[0]) + x_index * linear_step,
                        float(center[1]) + y_index * linear_step,
                        math.atan2(math.sin(yaw), math.cos(yaw)),
                    ])
        return np.asarray(values, dtype=np.float32)

    def _score(self, local_points: Any, candidates: Any) -> Any:
        local = np.asarray(local_points, dtype=np.float32)
        candidate_values = np.asarray(candidates, dtype=np.float32)
        local_wp = wp.array(local[:, :3], dtype=wp.vec3, device=self.device)
        candidates_wp = wp.array(candidate_values[:, :3], dtype=wp.vec3, device=self.device)
        scores_wp = wp.zeros(len(candidate_values), dtype=wp.float32, device=self.device)
        wp.launch(
            _score_pose_candidates_kernel,
            dim=len(candidate_values),
            inputs=[
                local_wp, candidates_wp, self.occupied,
                self.origin_cell_x, self.origin_cell_y,
                self.grid_width, self.grid_height,
                1.0 / self.resolution,
                scores_wp,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        return scores_wp.numpy()

    def match(
        self,
        local_points: Any,
        initial_pose: Any,
        *,
        linear_window: float,
        angular_window: float,
        minimum_score_gain: float,
    ) -> tuple[Any, float]:
        local = np.asarray(local_points, dtype=np.float32)
        local = local[::max(1, math.ceil(len(local) / 1440))]
        best = np.asarray(initial_pose, dtype=np.float64).copy()
        started = time.perf_counter()

        def search(center: Any, linear_step: float, angular_step: float, radius: int) -> tuple[Any, float]:
            candidates = self._candidates(center, linear_step, angular_step, radius)
            scores = self._score(local, candidates)
            center_score = float(scores[len(scores) // 2])
            best_index = int(np.argmax(scores))
            best_score = float(scores[best_index])
            if best_score <= center_score + float(minimum_score_gain):
                return np.asarray(center, dtype=np.float64).copy(), center_score
            return candidates[best_index].astype(np.float64), best_score

        coarse_linear = max(self.resolution * 2.0, float(linear_window) / 4.0)
        coarse_angular = max(math.radians(0.5), float(angular_window) / 4.0)
        coarse_radius = max(1, min(4, math.ceil(float(linear_window) / coarse_linear)))
        best, best_score = search(best, coarse_linear, coarse_angular, coarse_radius)
        best, best_score = search(best, max(self.resolution * 0.5, coarse_linear / 2.0), coarse_angular / 2.0, 2)
        best, best_score = search(
            best,
            max(self.resolution * 0.25, coarse_linear / 4.0),
            max(math.radians(0.1), coarse_angular / 5.0),
            2,
        )
        self.last_kernel_ms = (time.perf_counter() - started) * 1000.0
        return best, float(best_score)

    def score_pose(self, local_points: Any, pose: Any) -> float:
        local = np.asarray(local_points, dtype=np.float32)
        local = local[::max(1, math.ceil(len(local) / 1440))]
        return float(self._score(local, np.asarray([pose], dtype=np.float32))[0])

    def score_candidates(self, local_points: Any, candidates: Any) -> Any:
        """Score an explicit pose population and report synchronized pipeline time."""
        local = np.asarray(local_points, dtype=np.float32)
        local = local[::max(1, math.ceil(len(local) / 1440))]
        candidate_values = np.asarray(candidates, dtype=np.float32)
        if candidate_values.ndim != 2 or candidate_values.shape[1] < 3:
            raise ValueError("Warp particle scoring requires Nx3 pose candidates")
        started = time.perf_counter()
        scores = self._score(local, candidate_values)
        self.last_particle_pipeline_ms = (time.perf_counter() - started) * 1000.0
        return scores
