"""Persistent Warp occupancy/free-space grid for real LaserScan returns."""
from __future__ import annotations

import base64
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
    def _trace_free_space_kernel(
        hits: wp.array(dtype=wp.vec3),
        ray_count: wp.int32,
        origin_x: wp.float32,
        origin_y: wp.float32,
        beam_half_width_tangent: wp.float32,
        neighbor_min_cosine: wp.float32,
        range_jump_min: wp.float32,
        range_jump_ratio: wp.float32,
        world_min_x: wp.float32,
        world_min_y: wp.float32,
        resolution: wp.float32,
        grid_width: wp.int32,
        grid_height: wp.int32,
        free_evidence: wp.array(dtype=wp.int32),
        hit_evidence: wp.array(dtype=wp.int32),
    ):
        """Trace one real beam per thread into a persistent fixed grid."""
        ray = wp.tid()
        hit = hits[ray]
        dx = hit[0] - origin_x
        dy = hit[1] - origin_y
        distance = wp.sqrt(dx * dx + dy * dy)
        if distance < resolution * 0.5:
            return

        unit_x = dx / distance
        unit_y = dy / distance
        perpendicular_x = -unit_y
        perpendicular_y = unit_x
        negative_width_scale = wp.float32(1.0)
        positive_width_scale = wp.float32(1.0)

        # Widen only toward neighbors that belong to the same continuous
        # surface. A sudden range or angular jump is an object boundary, so the
        # beam footprint is narrowed on that side instead of clearing through
        # the gap between two unrelated returns.
        if ray > 0:
            neighbor = hits[ray - 1]
            neighbor_dx = neighbor[0] - origin_x
            neighbor_dy = neighbor[1] - origin_y
            neighbor_distance = wp.sqrt(neighbor_dx * neighbor_dx + neighbor_dy * neighbor_dy)
            if neighbor_distance > resolution * 0.5:
                neighbor_unit_x = neighbor_dx / neighbor_distance
                neighbor_unit_y = neighbor_dy / neighbor_distance
                neighbor_dot = unit_x * neighbor_unit_x + unit_y * neighbor_unit_y
                allowed_jump = wp.max(range_jump_min, range_jump_ratio * wp.min(distance, neighbor_distance))
                continuous = neighbor_dot >= neighbor_min_cosine and wp.abs(neighbor_distance - distance) <= allowed_jump
                cross = unit_x * neighbor_unit_y - unit_y * neighbor_unit_x
                if not continuous:
                    if cross >= 0.0:
                        positive_width_scale = wp.float32(0.35)
                    else:
                        negative_width_scale = wp.float32(0.35)
        if ray + 1 < ray_count:
            neighbor = hits[ray + 1]
            neighbor_dx = neighbor[0] - origin_x
            neighbor_dy = neighbor[1] - origin_y
            neighbor_distance = wp.sqrt(neighbor_dx * neighbor_dx + neighbor_dy * neighbor_dy)
            if neighbor_distance > resolution * 0.5:
                neighbor_unit_x = neighbor_dx / neighbor_distance
                neighbor_unit_y = neighbor_dy / neighbor_distance
                neighbor_dot = unit_x * neighbor_unit_x + unit_y * neighbor_unit_y
                allowed_jump = wp.max(range_jump_min, range_jump_ratio * wp.min(distance, neighbor_distance))
                continuous = neighbor_dot >= neighbor_min_cosine and wp.abs(neighbor_distance - distance) <= allowed_jump
                cross = unit_x * neighbor_unit_y - unit_y * neighbor_unit_x
                if not continuous:
                    if cross >= 0.0:
                        positive_width_scale = wp.float32(0.35)
                    else:
                        negative_width_scale = wp.float32(0.35)

        # Half-cell longitudinal samples plus a distance-dependent lateral
        # footprint approximate the physical angular beam instead of drawing a
        # one-cell line. The four-cell cap bounds work and avoids overstating
        # free space for unusually coarse scans.
        step_length = resolution * 0.5
        step_count = wp.int32(wp.ceil(distance / step_length))
        maximum_steps = grid_width + grid_height
        step_count = wp.min(step_count, maximum_steps)
        for step in range(step_count):
            fraction = wp.float32(step) / wp.float32(step_count)
            sample_distance = distance * fraction
            sample_x = origin_x + unit_x * sample_distance
            sample_y = origin_y + unit_y * sample_distance
            half_width = wp.max(resolution * 0.35, sample_distance * beam_half_width_tangent)
            half_width_cells = wp.min(wp.int32(4), wp.int32(wp.ceil(half_width / resolution)))
            for lateral_index in range(9):
                lateral_cell = lateral_index - 4
                if wp.abs(lateral_cell) <= half_width_cells:
                    width_scale = negative_width_scale
                    if lateral_cell >= 0:
                        width_scale = positive_width_scale
                    lateral_distance = wp.float32(lateral_cell) * resolution
                    if wp.abs(lateral_distance) <= half_width * width_scale + resolution * 0.35:
                        free_x = sample_x + perpendicular_x * lateral_distance
                        free_y = sample_y + perpendicular_y * lateral_distance
                        column = wp.int32(wp.floor((free_x - world_min_x) / resolution))
                        row = wp.int32(wp.floor((free_y - world_min_y) / resolution))
                        if column >= 0 and column < grid_width and row >= 0 and row < grid_height:
                            wp.atomic_add(free_evidence, row * grid_width + column, 1)

        # Endpoints use the same angular footprint, capped at three lateral
        # cells. This makes a continuous wall band while keeping its range
        # thickness at one cell and respecting discontinuity side gates.
        wall_half_width = wp.max(resolution * 0.6, distance * beam_half_width_tangent)
        wall_half_width_cells = wp.min(wp.int32(3), wp.int32(wp.ceil(wall_half_width / resolution)))
        for lateral_index in range(7):
            lateral_cell = lateral_index - 3
            if wp.abs(lateral_cell) <= wall_half_width_cells:
                width_scale = negative_width_scale
                if lateral_cell >= 0:
                    width_scale = positive_width_scale
                lateral_distance = wp.float32(lateral_cell) * resolution
                if wp.abs(lateral_distance) <= wall_half_width * width_scale + resolution * 0.5:
                    wall_x = hit[0] + perpendicular_x * lateral_distance
                    wall_y = hit[1] + perpendicular_y * lateral_distance
                    hit_column = wp.int32(wp.floor((wall_x - world_min_x) / resolution))
                    hit_row = wp.int32(wp.floor((wall_y - world_min_y) / resolution))
                    if hit_column >= 0 and hit_column < grid_width and hit_row >= 0 and hit_row < grid_height:
                        wp.atomic_add(hit_evidence, hit_row * grid_width + hit_column, 1)


    @wp.kernel
    def _classify_cells_kernel(
        free_evidence: wp.array(dtype=wp.int32),
        hit_evidence: wp.array(dtype=wp.int32),
        cell_states: wp.array(dtype=wp.uint8),
        free_output_count: wp.array(dtype=wp.int32),
        occupied_output_count: wp.array(dtype=wp.int32),
    ):
        cell = wp.tid()
        free_count = free_evidence[cell]
        hit_count = hit_evidence[cell]
        if hit_count >= 3 and free_count <= hit_count * 2:
            cell_states[cell] = wp.uint8(2)
            wp.atomic_add(occupied_output_count, 0, 1)
        elif free_count > 0:
            cell_states[cell] = wp.uint8(1)
            wp.atomic_add(free_output_count, 0, 1)
        else:
            cell_states[cell] = wp.uint8(0)


    @wp.kernel
    def _pack_cell_states_kernel(
        cell_states: wp.array(dtype=wp.uint8),
        cell_count: wp.int32,
        packed_states: wp.array(dtype=wp.uint8),
    ):
        slot = wp.tid()
        first = slot * 4
        packed = wp.int32(0)
        if first < cell_count:
            packed = packed | wp.int32(cell_states[first])
        if first + 1 < cell_count:
            packed = packed | (wp.int32(cell_states[first + 1]) << 2)
        if first + 2 < cell_count:
            packed = packed | (wp.int32(cell_states[first + 2]) << 4)
        if first + 3 < cell_count:
            packed = packed | (wp.int32(cell_states[first + 3]) << 6)
        packed_states[slot] = wp.uint8(packed)


class WarpOccupancyGrid:
    """Fixed-origin occupancy evidence with Warp-only map calculations."""

    def __init__(
        self,
        *,
        device: str,
        resolution_m: float,
        radius_m: float,
        center_xy: tuple[float, float],
        display_capacity: int = 40_000,
    ) -> None:
        if wp is None:
            raise RuntimeError("Warp is unavailable; install warp-lang>=1.6")
        if np is None:
            raise RuntimeError("NumPy is unavailable; install numpy>=1.24")
        self.device = wp.get_device(device)
        self.resolution_m = max(0.01, float(resolution_m))
        self.radius_m = max(self.resolution_m, float(radius_m))
        diameter_cells = max(1, int(math.ceil(self.radius_m * 2.0 / self.resolution_m)))
        # A bounded grid prevents a bad configuration from allocating an
        # unbounded device buffer. Its world origin never rolls with the robot.
        self.grid_width = min(2048, diameter_cells)
        self.grid_height = self.grid_width
        covered_radius = self.grid_width * self.resolution_m * 0.5
        self.world_min_x = float(center_xy[0]) - covered_radius
        self.world_min_y = float(center_xy[1]) - covered_radius
        self.capacity = max(1, int(display_capacity))
        cell_count = self.grid_width * self.grid_height
        self.packed_count = (cell_count + 3) // 4
        self.free_evidence = wp.zeros(cell_count, dtype=wp.int32, device=self.device)
        self.hit_evidence = wp.zeros(cell_count, dtype=wp.int32, device=self.device)
        self.cell_states = wp.zeros(cell_count, dtype=wp.uint8, device=self.device)
        self.packed_states = wp.zeros(self.packed_count, dtype=wp.uint8, device=self.device)
        self.output_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.occupied_output_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.packed_bytes = np.zeros(self.packed_count, dtype=np.uint8)
        self._state_bytes_cache: Any | None = np.zeros(cell_count, dtype=np.uint8)
        self.encoded_states = ""
        self._free_points_cache: Any | None = None
        self._occupied_points_cache: Any | None = None
        self.kernel_ms = 0.0
        self.encode_ms = 0.0
        self.rays = 0
        self.discovered_cells = 0
        self.occupied_cells = 0
        self.angular_increment_rad = 0.0
        self.beam_half_angle_rad = 0.0
        self.display_limited = False
        self.revision = 0

    @property
    def cell_count(self) -> int:
        return self.grid_width * self.grid_height

    def _materialize_points(self) -> None:
        if self._free_points_cache is not None and self._occupied_points_cache is not None:
            return
        if self._state_bytes_cache is None:
            states = np.empty(self.cell_count, dtype=np.uint8)
            states[0::4] = self.packed_bytes & np.uint8(3)
            states[1::4] = (self.packed_bytes[:len(states[1::4])] >> np.uint8(2)) & np.uint8(3)
            states[2::4] = (self.packed_bytes[:len(states[2::4])] >> np.uint8(4)) & np.uint8(3)
            states[3::4] = (self.packed_bytes[:len(states[3::4])] >> np.uint8(6)) & np.uint8(3)
            self._state_bytes_cache = states
        values = []
        for state, z_value in ((1, -0.015), (2, 0.0)):
            indices = np.flatnonzero(self._state_bytes_cache == state)
            rows = indices // self.grid_width
            columns = indices % self.grid_width
            points = np.column_stack((
                self.world_min_x + (columns.astype(np.float32) + 0.5) * self.resolution_m,
                self.world_min_y + (rows.astype(np.float32) + 0.5) * self.resolution_m,
                np.full(len(indices), z_value, dtype=np.float32),
            )).astype(np.float32, copy=False)
            values.append(points)
        self._free_points_cache, self._occupied_points_cache = values

    @property
    def points(self) -> Any:
        """Materialize free cell centers only for compatibility/debug callers."""
        self._materialize_points()
        return self._free_points_cache

    @property
    def occupied_points(self) -> Any:
        """Materialize occupied cell centers only for compatibility/debug callers."""
        self._materialize_points()
        return self._occupied_points_cache

    def _encode_states(self) -> None:
        started = time.perf_counter()
        self.encoded_states = base64.b64encode(self.packed_bytes.tobytes()).decode("ascii")
        self.encode_ms = (time.perf_counter() - started) * 1000.0

    def update(
        self,
        world_hits: Any,
        origin_xy: tuple[float, float],
        *,
        angular_increment_rad: float = math.radians(1.0),
    ) -> dict[str, Any]:
        hits_np = np.asarray(world_hits, dtype=np.float32)
        if hits_np.ndim != 2 or hits_np.shape[1] < 3 or len(hits_np) == 0:
            return self.snapshot()
        angular_increment = abs(float(angular_increment_rad))
        if not math.isfinite(angular_increment) or angular_increment <= 0.0:
            angular_increment = math.radians(1.0)
        beam_half_angle = min(math.radians(2.5), max(math.radians(0.1), angular_increment * 0.55))
        neighbor_angle = min(math.radians(8.0), max(math.radians(0.5), angular_increment * 1.75))
        hits_wp = wp.array(hits_np[:, :3], dtype=wp.vec3, device=self.device)
        self.output_count.zero_()
        self.occupied_output_count.zero_()
        started = time.perf_counter()
        wp.launch(
            _trace_free_space_kernel,
            dim=len(hits_np),
            inputs=[
                hits_wp,
                len(hits_np),
                float(origin_xy[0]), float(origin_xy[1]),
                math.tan(beam_half_angle), math.cos(neighbor_angle),
                max(self.resolution_m * 2.0, 0.15), 0.15,
                self.world_min_x, self.world_min_y, self.resolution_m,
                self.grid_width, self.grid_height,
                self.free_evidence, self.hit_evidence,
            ],
            device=self.device,
        )
        wp.launch(
            _classify_cells_kernel,
            dim=self.cell_count,
            inputs=[
                self.free_evidence, self.hit_evidence,
                self.cell_states,
                self.output_count, self.occupied_output_count,
            ],
            device=self.device,
        )
        wp.launch(
            _pack_cell_states_kernel,
            dim=self.packed_count,
            inputs=[self.cell_states, self.cell_count, self.packed_states],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        discovered_count = int(self.output_count.numpy()[0])
        occupied_count = int(self.occupied_output_count.numpy()[0])
        self.packed_bytes = self.packed_states.numpy().astype(np.uint8, copy=False)
        self.kernel_ms = (time.perf_counter() - started) * 1000.0
        self._encode_states()
        self._state_bytes_cache = None
        self._free_points_cache = None
        self._occupied_points_cache = None
        self.rays = len(hits_np)
        self.discovered_cells = discovered_count
        self.occupied_cells = occupied_count
        self.angular_increment_rad = angular_increment
        self.beam_half_angle_rad = beam_half_angle
        self.display_limited = False
        self.revision += 1
        return self.snapshot()

    def clear(self) -> None:
        self.free_evidence.zero_()
        self.hit_evidence.zero_()
        self.cell_states.zero_()
        self.packed_states.zero_()
        self.output_count.zero_()
        self.occupied_output_count.zero_()
        self.packed_bytes = np.zeros(self.packed_count, dtype=np.uint8)
        self._state_bytes_cache = np.zeros(self.cell_count, dtype=np.uint8)
        self.encoded_states = ""
        self._free_points_cache = None
        self._occupied_points_cache = None
        self.kernel_ms = 0.0
        self.encode_ms = 0.0
        self.rays = 0
        self.discovered_cells = 0
        self.occupied_cells = 0
        self.angular_increment_rad = 0.0
        self.beam_half_angle_rad = 0.0
        self.display_limited = False
        self.revision += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "backend": "warp",
            "device": str(self.device),
            "kernel_ms": float(self.kernel_ms),
            "encode_ms": float(self.encode_ms),
            "rays": int(self.rays),
            "grid_cells": int(self.cell_count),
            "grid_width": int(self.grid_width),
            "grid_height": int(self.grid_height),
            "free_cells": int(self.discovered_cells),
            "display_cells": int(self.discovered_cells),
            "occupied_cells": int(self.occupied_cells),
            "occupied_display_cells": int(self.occupied_cells),
            "beam_model": "angular-footprint",
            "angular_increment_rad": float(self.angular_increment_rad),
            "beam_half_angle_rad": float(self.beam_half_angle_rad),
            "free_half_width_limit_cells": 4,
            "wall_half_width_limit_cells": 3,
            "discontinuity_gating": True,
            "display_limited": bool(self.display_limited),
            "resolution_m": float(self.resolution_m),
            "world_min_x": float(self.world_min_x),
            "world_min_y": float(self.world_min_y),
            "fixed_origin": True,
            "encoding": "u2-base64",
            "data": self.encoded_states,
            "revision": int(self.revision),
        }
