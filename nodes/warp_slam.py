"""Managed Warp-powered synthetic rover SLAM discovery viewer."""
from __future__ import annotations

import ctypes
import math
import time
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency contract
    np = None

try:
    import warp as wp
except Exception:  # pragma: no cover - package must load without Warp
    wp = None

from blacknode.node import Bool, Dict, Enum, Float, Int, Text, node

from . import warp_viewer_runtime as viewer_rt


_CATEGORY = "NVIDIA CUDA"
_WORLD_BOUNDS = (-9.0, -6.5, 9.0, 6.5)
_OBSTACLES = [
    (-6.2, -3.2, -4.2, -1.8),
    (-6.0, 1.9, -4.4, 4.2),
    (-1.2, -0.8, 1.2, 0.8),
    (-1.5, 3.5, 1.8, 4.4),
    (2.6, -4.2, 5.0, -2.8),
    (4.2, 1.1, 6.1, 3.8),
    (1.8, 1.8, 3.0, 2.8),
]
_ROUTE = [
    (-7.5, -5.2),
    (-7.4, 0.0),
    (-7.0, 5.2),
    (-2.5, 5.2),
    (-2.3, 2.6),
    (0.0, 2.6),
    (2.0, 3.2),
    (2.2, 5.1),
    (3.0, 5.2),
    (7.4, 4.8),
    (7.3, 0.0),
    (6.8, -5.2),
    (1.0, -5.2),
    (-2.0, -3.1),
    (-5.7, -4.7),
    (-7.5, -5.2),
]

_SLAM_GL_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 position;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec3 source_position;
void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
    source_position = position;
}
"""

_SLAM_GL_FRAGMENT_SHADER = """
#version 330 core
in vec3 source_position;
uniform float point_mode;
uniform float alpha;
uniform float white_mode;
out vec4 fragment_color;
void main() {
    if (point_mode > 0.5) {
        vec2 centered = gl_PointCoord - vec2(0.5);
        float radius = length(centered);
        if (radius > 0.5) discard;
    }
    float x_mix = smoothstep(-8.5, 8.5, source_position.x);
    float y_mix = smoothstep(-6.0, 6.0, source_position.y);
    vec3 electric_blue = vec3(0.015, 0.20, 1.0);
    vec3 laser_cyan = vec3(0.0, 0.95, 1.0);
    vec3 signal_green = vec3(0.15, 1.0, 0.38);
    vec3 color = mix(electric_blue, laser_cyan, x_mix);
    color = mix(color, signal_green, y_mix * 0.72);
    float glow = 0.88 + 0.12 * sin(source_position.x * 1.7 + source_position.y * 1.3);
    if (white_mode > 0.5) {
        color = vec3(0.88, 0.94, 1.0);
        glow = 1.0;
    }
    fragment_color = vec4(color * glow, alpha);
}
"""


if wp is not None:
    @wp.kernel
    def _raycast_aabb_kernel(
        obstacles: wp.array(dtype=wp.vec4),
        obstacle_count: wp.int32,
        origin_x: wp.float32,
        origin_y: wp.float32,
        origin_z: wp.float32,
        yaw: wp.float32,
        angle_min: wp.float32,
        angle_increment: wp.float32,
        elevation_min: wp.float32,
        elevation_increment: wp.float32,
        vertical_layers: wp.int32,
        range_max: wp.float32,
        world_min_x: wp.float32,
        world_min_y: wp.float32,
        world_max_x: wp.float32,
        world_max_y: wp.float32,
        world_wall_height: wp.float32,
        hits: wp.array(dtype=wp.vec3),
        ranges: wp.array(dtype=wp.float32),
    ):
        ray = wp.tid()
        vertical_index = ray % vertical_layers
        horizontal_index = ray // vertical_layers
        angle = yaw + angle_min + wp.float32(horizontal_index) * angle_increment
        elevation = elevation_min + wp.float32(vertical_index) * elevation_increment
        horizontal_scale = wp.cos(elevation)
        dx = horizontal_scale * wp.cos(angle)
        dy = horizontal_scale * wp.sin(angle)
        dz = wp.sin(elevation)
        best = range_max
        found = wp.int32(0)
        epsilon = 1.0e-6

        if wp.abs(dx) > epsilon:
            wall_x = world_max_x if dx > 0.0 else world_min_x
            distance = (wall_x - origin_x) / dx
            wall_y = origin_y + distance * dy
            wall_z = origin_z + distance * dz
            if distance > 0.0 and distance < best and wall_y >= world_min_y and wall_y <= world_max_y and wall_z >= 0.0 and wall_z <= world_wall_height:
                best = distance
                found = 1
        if wp.abs(dy) > epsilon:
            wall_y = world_max_y if dy > 0.0 else world_min_y
            distance = (wall_y - origin_y) / dy
            wall_x = origin_x + distance * dx
            wall_z = origin_z + distance * dz
            if distance > 0.0 and distance < best and wall_x >= world_min_x and wall_x <= world_max_x and wall_z >= 0.0 and wall_z <= world_wall_height:
                best = distance
                found = 1

        for obstacle_index in range(obstacle_count):
            box = obstacles[obstacle_index]
            box_height = 0.85 + 0.24 * wp.float32(obstacle_index % 3)
            near_distance = 0.0
            far_distance = range_max
            valid = wp.int32(1)
            if wp.abs(dx) > epsilon:
                distance0 = (box[0] - origin_x) / dx
                distance1 = (box[2] - origin_x) / dx
                axis_near = wp.min(distance0, distance1)
                axis_far = wp.max(distance0, distance1)
                near_distance = wp.max(near_distance, axis_near)
                far_distance = wp.min(far_distance, axis_far)
            elif origin_x < box[0] or origin_x > box[2]:
                valid = 0
            if wp.abs(dy) > epsilon and valid == 1:
                distance0 = (box[1] - origin_y) / dy
                distance1 = (box[3] - origin_y) / dy
                axis_near = wp.min(distance0, distance1)
                axis_far = wp.max(distance0, distance1)
                near_distance = wp.max(near_distance, axis_near)
                far_distance = wp.min(far_distance, axis_far)
            elif wp.abs(dy) <= epsilon and (origin_y < box[1] or origin_y > box[3]):
                valid = 0
            if wp.abs(dz) > epsilon and valid == 1:
                distance0 = (0.0 - origin_z) / dz
                distance1 = (box_height - origin_z) / dz
                axis_near = wp.min(distance0, distance1)
                axis_far = wp.max(distance0, distance1)
                near_distance = wp.max(near_distance, axis_near)
                far_distance = wp.min(far_distance, axis_far)
            elif wp.abs(dz) <= epsilon and (origin_z < 0.0 or origin_z > box_height):
                valid = 0
            if valid == 1 and far_distance >= near_distance and near_distance > 0.0 and near_distance < best:
                best = near_distance
                found = 1

        ranges[ray] = best
        if found == 1:
            hits[ray] = wp.vec3(
                origin_x + best * dx,
                origin_y + best * dy,
                origin_z + best * dz,
            )
        else:
            hits[ray] = wp.vec3(0.0, 0.0, -1000.0)


    @wp.kernel
    def _apply_lidar_noise_kernel(
        hits: wp.array(dtype=wp.vec3),
        ranges: wp.array(dtype=wp.float32),
        origin_x: wp.float32,
        origin_y: wp.float32,
        origin_z: wp.float32,
        range_max: wp.float32,
        seed: wp.int32,
    ):
        ray = wp.tid()
        hit = hits[ray]
        if hit[2] < -1.0:
            return

        offset_x = hit[0] - origin_x
        offset_y = hit[1] - origin_y
        offset_z = hit[2] - origin_z
        true_range = wp.sqrt(
            offset_x * offset_x + offset_y * offset_y + offset_z * offset_z
        )
        if true_range < 1.0e-6:
            return

        rng = wp.rand_init(seed, ray)
        dropout_probability = 0.002 + 0.0004 * true_range
        if wp.randf(rng) < dropout_probability:
            ranges[ray] = range_max
            hits[ray] = wp.vec3(0.0, 0.0, -1000.0)
            return

        yaw = wp.atan2(offset_y, offset_x)
        elevation = wp.asin(wp.clamp(offset_z / true_range, -1.0, 1.0))
        noisy_yaw = yaw + 0.00044 * wp.randn(rng)
        noisy_elevation = elevation + 0.00070 * wp.randn(rng)
        range_sigma = 0.004 + 0.0015 * true_range
        noisy_range = wp.clamp(
            true_range + range_sigma * wp.randn(rng),
            0.05,
            range_max,
        )
        horizontal_scale = wp.cos(noisy_elevation)
        hits[ray] = wp.vec3(
            origin_x + noisy_range * horizontal_scale * wp.cos(noisy_yaw),
            origin_y + noisy_range * horizontal_scale * wp.sin(noisy_yaw),
            origin_z + noisy_range * wp.sin(noisy_elevation),
        )
        ranges[ray] = noisy_range


    @wp.kernel
    def _clear_points_kernel(points: wp.array(dtype=wp.vec3)):
        point = wp.tid()
        points[point] = wp.vec3(0.0, 0.0, -1000.0)


    @wp.kernel
    def _clear_lines_kernel(lines: wp.array(dtype=wp.vec3, ndim=2)):
        line = wp.tid()
        hidden = wp.vec3(0.0, 0.0, -1000.0)
        lines[line, 0] = hidden
        lines[line, 1] = hidden


    @wp.kernel
    def _accumulate_occupancy_kernel(
        hits: wp.array(dtype=wp.vec3),
        world_min_x: wp.float32,
        world_min_y: wp.float32,
        resolution: wp.float32,
        grid_width: wp.int32,
        grid_height: wp.int32,
        grid_depth: wp.int32,
        map_capacity: wp.int32,
        occupancy: wp.array(dtype=wp.int32),
        map_count: wp.array(dtype=wp.int32),
        map_points: wp.array(dtype=wp.vec3),
    ):
        ray = wp.tid()
        hit = hits[ray]
        if hit[2] < 0.0:
            return
        column = wp.int32(wp.floor((hit[0] - world_min_x) / resolution))
        row = wp.int32(wp.floor((hit[1] - world_min_y) / resolution))
        layer = wp.int32(wp.floor(hit[2] / resolution))
        if column < 0 or column >= grid_width or row < 0 or row >= grid_height or layer < 0 or layer >= grid_depth:
            return
        cell = (layer * grid_height + row) * grid_width + column
        previous = wp.atomic_cas(occupancy, cell, 0, 1)
        if previous == 0:
            slot = wp.atomic_add(map_count, 0, 1)
            if slot < map_capacity:
                map_points[slot] = wp.vec3(
                    world_min_x + (wp.float32(column) + 0.5) * resolution,
                    world_min_y + (wp.float32(row) + 0.5) * resolution,
                    (wp.float32(layer) + 0.5) * resolution,
                )


    @wp.kernel
    def _accumulate_pointcloud_kernel(
        hits: wp.array(dtype=wp.vec3),
        world_min_x: wp.float32,
        world_min_y: wp.float32,
        resolution: wp.float32,
        grid_width: wp.int32,
        grid_height: wp.int32,
        grid_depth: wp.int32,
        map_capacity: wp.int32,
        occupancy: wp.array(dtype=wp.int32),
        map_count: wp.array(dtype=wp.int32),
        map_points: wp.array(dtype=wp.vec3),
    ):
        ray = wp.tid()
        hit = hits[ray]
        if hit[2] < 0.0:
            return
        column = wp.int32(wp.floor((hit[0] - world_min_x) / resolution))
        row = wp.int32(wp.floor((hit[1] - world_min_y) / resolution))
        layer = wp.int32(wp.floor(hit[2] / resolution))
        if column < 0 or column >= grid_width or row < 0 or row >= grid_height or layer < 0 or layer >= grid_depth:
            return
        cell = (layer * grid_height + row) * grid_width + column
        previous = wp.atomic_cas(occupancy, cell, 0, 1)
        if previous == 0:
            slot = wp.atomic_add(map_count, 0, 1)
            if slot < map_capacity:
                map_points[slot] = hit


    @wp.kernel
    def _accumulate_floor_discovery_kernel(
        obstacles: wp.array(dtype=wp.vec4),
        obstacle_count: wp.int32,
        origin_x: wp.float32,
        origin_y: wp.float32,
        range_max: wp.float32,
        world_min_x: wp.float32,
        world_min_y: wp.float32,
        resolution: wp.float32,
        grid_width: wp.int32,
        discovered: wp.array(dtype=wp.int32),
        observation_count: wp.array(dtype=wp.int32),
        floor_points: wp.array(dtype=wp.vec3),
    ):
        """Permanently reveal each floor cell visible from the current pose."""
        cell = wp.tid()
        column = cell % grid_width
        row = cell // grid_width
        target_x = world_min_x + (wp.float32(column) + 0.5) * resolution
        target_y = world_min_y + (wp.float32(row) + 0.5) * resolution
        offset_x = target_x - origin_x
        offset_y = target_y - origin_y
        target_distance = wp.sqrt(offset_x * offset_x + offset_y * offset_y)
        if target_distance > range_max:
            return
        if target_distance < 1.0e-6:
            if discovered[cell] == 0:
                discovered[cell] = 1
                floor_points[cell] = wp.vec3(target_x, target_y, 0.012)
            observation_count[cell] = observation_count[cell] + 1
            return

        dx = offset_x / target_distance
        dy = offset_y / target_distance
        nearest_blocker = range_max
        epsilon = 1.0e-6
        for obstacle_index in range(obstacle_count):
            box = obstacles[obstacle_index]
            if wp.abs(dx) > epsilon:
                distance = (box[0] - origin_x) / dx
                hit_y = origin_y + distance * dy
                if distance > 0.0 and distance < nearest_blocker and hit_y >= box[1] and hit_y <= box[3]:
                    nearest_blocker = distance
                distance = (box[2] - origin_x) / dx
                hit_y = origin_y + distance * dy
                if distance > 0.0 and distance < nearest_blocker and hit_y >= box[1] and hit_y <= box[3]:
                    nearest_blocker = distance
            if wp.abs(dy) > epsilon:
                distance = (box[1] - origin_y) / dy
                hit_x = origin_x + distance * dx
                if distance > 0.0 and distance < nearest_blocker and hit_x >= box[0] and hit_x <= box[2]:
                    nearest_blocker = distance
                distance = (box[3] - origin_y) / dy
                hit_x = origin_x + distance * dx
                if distance > 0.0 and distance < nearest_blocker and hit_x >= box[0] and hit_x <= box[2]:
                    nearest_blocker = distance

        if target_distance <= nearest_blocker + resolution * 0.35:
            if discovered[cell] == 0:
                discovered[cell] = 1
                floor_points[cell] = wp.vec3(target_x, target_y, 0.012)
            observation_count[cell] = observation_count[cell] + 1


    @wp.kernel
    def _jitter_new_floor_points_kernel(
        discovered: wp.array(dtype=wp.int32),
        floor_points: wp.array(dtype=wp.vec3),
        resolution: wp.float32,
        noise_enabled: wp.int32,
        seed: wp.int32,
    ):
        cell = wp.tid()
        previous = wp.atomic_cas(discovered, cell, 1, 2)
        if previous != 1 or noise_enabled == 0:
            return
        point = floor_points[cell]
        rng = wp.rand_init(seed, cell)
        limit = resolution * 0.42
        jitter_x = wp.clamp(resolution * 0.18 * wp.randn(rng), -limit, limit)
        jitter_y = wp.clamp(resolution * 0.18 * wp.randn(rng), -limit, limit)
        jitter_z = wp.clamp(0.005 * wp.randn(rng), -0.009, 0.014)
        floor_points[cell] = wp.vec3(
            point[0] + jitter_x,
            point[1] + jitter_y,
            wp.max(0.002, point[2] + jitter_z),
        )


    @wp.kernel
    def _build_stable_floor_surfels_kernel(
        discovered: wp.array(dtype=wp.int32),
        observation_count: wp.array(dtype=wp.int32),
        surfels_built: wp.array(dtype=wp.int32),
        floor_points: wp.array(dtype=wp.vec3),
        samples_per_cell: wp.int32,
        samples_per_observation: wp.int32,
        resolution: wp.float32,
        noise_enabled: wp.int32,
        seed: wp.int32,
        surfel_points: wp.array(dtype=wp.vec3),
    ):
        cell = wp.tid()
        if discovered[cell] == 0:
            return
        built_count = surfels_built[cell]
        desired_count = wp.min(
            samples_per_cell,
            observation_count[cell] * samples_per_observation,
        )
        if desired_count <= built_count:
            return
        point = floor_points[cell]
        jitter_scale = resolution * 0.24 if noise_enabled == 1 else 0.0
        jitter_z_scale = 0.006 if noise_enabled == 1 else 0.0
        for sample_in_cell in range(samples_per_cell):
            if sample_in_cell >= built_count and sample_in_cell < desired_count:
                sample = cell * samples_per_cell + sample_in_cell
                rng = wp.rand_init(seed, sample)
                surfel_points[sample] = wp.vec3(
                    point[0] + jitter_scale * wp.randn(rng),
                    point[1] + jitter_scale * wp.randn(rng),
                    wp.max(0.002, point[2] + jitter_z_scale * wp.randn(rng)),
                )
        surfels_built[cell] = desired_count


    @wp.kernel
    def _append_persistent_wall_samples_kernel(
        hits: wp.array(dtype=wp.vec3),
        ray_count: wp.int32,
        sample_capacity: wp.int32,
        sample_count: wp.array(dtype=wp.int32),
        seed: wp.int32,
        samples: wp.array(dtype=wp.vec3),
    ):
        sample = wp.tid()
        rng = wp.rand_init(seed, sample)
        ray = wp.randi(rng, 0, ray_count)
        hit = hits[ray]
        if hit[2] < 0.0:
            return
        slot = wp.atomic_add(sample_count, 0, 1)
        if slot < sample_capacity:
            samples[slot] = hit


    @wp.kernel
    def _append_persistent_floor_samples_kernel(
        discovered: wp.array(dtype=wp.int32),
        floor_points: wp.array(dtype=wp.vec3),
        cell_sequence: wp.array(dtype=wp.int32),
        floor_cell_count: wp.int32,
        resolution: wp.float32,
        noise_enabled: wp.int32,
        sample_capacity: wp.int32,
        sample_count: wp.array(dtype=wp.int32),
        seed: wp.int32,
        samples: wp.array(dtype=wp.vec3),
    ):
        sample = wp.tid()
        rng = wp.rand_init(seed, sample)
        cell = wp.randi(rng, 0, floor_cell_count)
        if discovered[cell] == 0:
            return
        slot = wp.atomic_add(sample_count, 0, 1)
        if slot >= sample_capacity:
            return
        sequence = wp.atomic_add(cell_sequence, cell, 1)
        point = floor_points[cell]
        # Randomized stratification keeps the visual distribution irregular,
        # while visiting every subregion before clustering in one part of a cell.
        strata = wp.int32(11)
        stratum_count = wp.int32(121)
        cycle = sequence // stratum_count
        stratum = (sequence * 37 + cell * 53 + cycle * 29) % stratum_count
        stratum_x = stratum % strata
        stratum_y = stratum // strata
        u = (wp.float32(stratum_x) + wp.randf(rng, 0.0, 1.0)) / wp.float32(strata)
        v = (wp.float32(stratum_y) + wp.randf(rng, 0.0, 1.0)) / wp.float32(strata)
        span = resolution * 0.48
        xy_sigma = resolution * 0.015 if noise_enabled == 1 else 0.0
        height_sigma = 0.006 if noise_enabled == 1 else 0.0
        samples[slot] = wp.vec3(
            point[0] + (2.0 * u - 1.0) * span + xy_sigma * wp.randn(rng),
            point[1] + (2.0 * v - 1.0) * span + xy_sigma * wp.randn(rng),
            wp.max(0.002, point[2] + height_sigma * wp.randn(rng)),
        )


    @wp.kernel
    def _build_current_lines_kernel(
        hits: wp.array(dtype=wp.vec3),
        ray_count: wp.int32,
        origin_x: wp.float32,
        origin_y: wp.float32,
        lines: wp.array(dtype=wp.vec3, ndim=2),
    ):
        line = wp.tid()
        line_count = lines.shape[0]
        hit_index = wp.int32(0)
        if line_count > 1:
            hit_index = wp.int32(
                wp.float32(line) * wp.float32(ray_count - 1) / wp.float32(line_count - 1)
            )
        hit = hits[hit_index]
        if hit[2] > -1.0:
            lines[line, 0] = wp.vec3(origin_x, origin_y, 0.03)
            lines[line, 1] = hit
        else:
            hidden = wp.vec3(0.0, 0.0, -1000.0)
            lines[line, 0] = hidden
            lines[line, 1] = hidden


    @wp.kernel
    def _build_slam_ray_lines_kernel(
        hits: wp.array(dtype=wp.vec3),
        ray_count: wp.int32,
        origin_x: wp.float32,
        origin_y: wp.float32,
        origin_z: wp.float32,
        lines: wp.array(dtype=wp.vec3, ndim=2),
    ):
        line = wp.tid()
        line_count = lines.shape[0]
        hit_index = wp.int32(0)
        if line_count > 1:
            hit_index = wp.int32(
                wp.float32(line) * wp.float32(ray_count - 1) / wp.float32(line_count - 1)
            )
        hit = hits[hit_index]
        if hit[2] > -1.0:
            lines[line, 0] = wp.vec3(origin_x, origin_y, origin_z)
            lines[line, 1] = hit
        else:
            hidden = wp.vec3(0.0, 0.0, -1000.0)
            lines[line, 0] = hidden
            lines[line, 1] = hidden


    @wp.kernel
    def _append_history_lines_kernel(
        current_lines: wp.array(dtype=wp.vec3, ndim=2),
        history_capacity: wp.int32,
        history_count: wp.array(dtype=wp.int32),
        history_lines: wp.array(dtype=wp.vec3, ndim=2),
    ):
        line = wp.tid()
        hit = current_lines[line, 1]
        if hit[2] < -1.0:
            return
        absolute_slot = wp.atomic_add(history_count, 0, 1)
        slot = absolute_slot % history_capacity
        history_lines[slot, 0] = current_lines[line, 0]
        history_lines[slot, 1] = hit


    @wp.kernel
    def _build_radial_rings_kernel(
        origin_x: wp.float32,
        origin_y: wp.float32,
        elapsed: wp.float32,
        range_max: wp.float32,
        ring_count: wp.int32,
        segments: wp.int32,
        lines: wp.array(dtype=wp.vec3, ndim=2),
    ):
        line = wp.tid()
        ring = line // segments
        segment = line - ring * segments
        cycle = elapsed * 0.46 + wp.float32(ring) / wp.float32(ring_count)
        phase = cycle - wp.floor(cycle)
        radius = 0.18 + phase * (range_max - 0.18)
        angle0 = 2.0 * wp.pi * wp.float32(segment) / wp.float32(segments)
        angle1 = 2.0 * wp.pi * wp.float32(segment + 1) / wp.float32(segments)
        height = 0.10 + 0.025 * phase
        lines[line, 0] = wp.vec3(
            origin_x + radius * wp.cos(angle0),
            origin_y + radius * wp.sin(angle0),
            height,
        )
        lines[line, 1] = wp.vec3(
            origin_x + radius * wp.cos(angle1),
            origin_y + radius * wp.sin(angle1),
            height,
        )


def slam_demo_scenario(ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the deterministic public scenario passed to the managed worker."""
    values = ctx or {}
    scan_visualization = str(values.get("scan_visualization") or "rays").strip().lower()
    if scan_visualization not in {"rays", "rings"}:
        scan_visualization = "rays"
    return {
        "kind": "blacknode.warp-slam-demo",
        "schema_version": 1,
        "world_bounds": list(_WORLD_BOUNDS),
        "obstacles": [list(box) for box in _OBSTACLES],
        "route": [list(point) for point in _ROUTE],
        "rays_per_scan": max(64, min(2_000_000, int(values.get("rays_per_scan") or 1_000_000))),
        "scan_hz": max(0.5, min(60.0, float(values.get("scan_hz") or 12.0))),
        "rover_speed_m_s": max(0.05, min(5.0, float(values.get("rover_speed_m_s") or 1.2))),
        "range_max_m": max(1.0, min(50.0, float(values.get("range_max_m") or 10.0))),
        "map_resolution_m": max(0.02, min(0.5, float(values.get("map_resolution_m") or 0.08))),
        "map_capacity": max(1_000, min(250_000, int(values.get("map_capacity") or 250_000))),
        "ray_history_capacity": max(
            256, min(250_000, int(values.get("ray_history_capacity") or 100_000))
        ),
        "fps": max(10, min(120, int(values.get("fps") or 30))),
        "show_ground_truth": bool(values.get("show_ground_truth", False)),
        "show_paths": bool(values.get("show_paths", False)),
        "accumulate_rays": bool(values.get("accumulate_rays", True)),
        "scan_visualization": scan_visualization,
    }


def route_pose(route: Any, distance_m: float) -> tuple[float, float, float, float]:
    """Interpolate a constant-speed pose along a closed polyline."""
    if np is None:
        raise RuntimeError("NumPy is required")
    points = np.asarray(route, dtype=np.float32)
    deltas = points[1:] - points[:-1]
    lengths = np.linalg.norm(deltas, axis=1)
    total = float(np.sum(lengths))
    if total <= 0.0:
        return float(points[0, 0]), float(points[0, 1]), 0.0, 0.0
    wrapped = float(distance_m) % total
    cumulative = np.cumsum(lengths)
    segment = int(np.searchsorted(cumulative, wrapped, side="right"))
    segment = min(segment, len(lengths) - 1)
    before = 0.0 if segment == 0 else float(cumulative[segment - 1])
    fraction = (wrapped - before) / max(float(lengths[segment]), 1.0e-6)
    point = points[segment] + fraction * deltas[segment]
    yaw = math.atan2(float(deltas[segment, 1]), float(deltas[segment, 0]))
    return float(point[0]), float(point[1]), yaw, wrapped / total


def _rectangle_lines(boxes: Any) -> tuple[list[list[float]], list[int]]:
    vertices: list[list[float]] = []
    indices: list[int] = []
    for xmin, ymin, xmax, ymax in boxes:
        start = len(vertices)
        vertices.extend([
            [xmin, ymin, 0.0], [xmax, ymin, 0.0],
            [xmax, ymax, 0.0], [xmin, ymax, 0.0],
        ])
        indices.extend([
            start, start + 1, start + 1, start + 2,
            start + 2, start + 3, start + 3, start,
        ])
    return vertices, indices


def _ghost_obstacle_geometry(boxes: Any) -> tuple[Any, Any]:
    """Build translucent box faces and luminous edge lines for obstacles."""
    if np is None:
        raise RuntimeError("NumPy is required")
    face_vertices: list[list[float]] = []
    edge_vertices: list[list[float]] = []
    triangle_indices = (
        0, 2, 1, 0, 3, 2,  # bottom
        4, 5, 6, 4, 6, 7,  # top
        0, 1, 5, 0, 5, 4,
        1, 2, 6, 1, 6, 5,
        2, 3, 7, 2, 7, 6,
        3, 0, 4, 3, 4, 7,
    )
    edge_indices = (
        0, 1, 1, 2, 2, 3, 3, 0,
        4, 5, 5, 6, 6, 7, 7, 4,
        0, 4, 1, 5, 2, 6, 3, 7,
    )
    for obstacle_index, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        height = 0.85 + 0.24 * float(obstacle_index % 3)
        corners = (
            [xmin, ymin, 0.0], [xmax, ymin, 0.0],
            [xmax, ymax, 0.0], [xmin, ymax, 0.0],
            [xmin, ymin, height], [xmax, ymin, height],
            [xmax, ymax, height], [xmin, ymax, height],
        )
        face_vertices.extend(corners[index] for index in triangle_indices)
        edge_vertices.extend(corners[index] for index in edge_indices)
    return (
        np.asarray(face_vertices, dtype=np.float32),
        np.asarray(edge_vertices, dtype=np.float32),
    )


def run_slam_discovery_viewer(*, config: dict[str, Any], device: str) -> None:
    """Animate a rover and accumulate its synthetic LiDAR hits into a map."""
    if wp is None or np is None:
        raise RuntimeError("NVIDIA Warp and NumPy are required for the SLAM discovery viewer")
    import pyglet
    import warp.render
    from pyglet.graphics.shader import Shader, ShaderProgram

    selected_device = wp.get_device(device)
    bounds = tuple(float(value) for value in config["world_bounds"])
    obstacles_np = np.asarray(config["obstacles"], dtype=np.float32)
    route = np.asarray(config["route"], dtype=np.float32)
    ray_count = int(config["rays_per_scan"])
    scan_hz = float(config["scan_hz"])
    speed = float(config["rover_speed_m_s"])
    range_max = float(config["range_max_m"])
    resolution = float(config["map_resolution_m"])
    capacity = int(config["map_capacity"])
    ray_history_capacity = int(config["ray_history_capacity"])
    fps = int(config["fps"])
    scan_visual_mode = str(config.get("scan_visualization") or "rays")
    world_center_x = 0.5 * (bounds[0] + bounds[2])
    world_center_y = 0.5 * (bounds[1] + bounds[3])

    renderer = warp.render.OpenGLRenderer(
        title="Blacknode SLAM — rover discovery",
        fps=fps,
        up_axis="Z",
        screen_width=1200,
        screen_height=860,
        near_plane=0.01,
        far_plane=100.0,
        camera_fov=43.0,
        camera_pos=(-19.0 + world_center_y, 17.0, 15.0 + world_center_x),
        camera_front=(19.0, -17.0, -15.0),
        camera_up=(0.0, 1.0, 0.0),
        background_color=(0.003, 0.007, 0.018),
        draw_grid=False,
        draw_sky=False,
        draw_axis=False,
        show_info=False,
        axis_scale=0.6,
        enable_mouse_interaction=True,
        device=selected_device,
    )

    grid_width = max(1, int(math.ceil((bounds[2] - bounds[0]) / resolution)))
    grid_height = max(1, int(math.ceil((bounds[3] - bounds[1]) / resolution)))
    floor_cell_count = grid_width * grid_height
    floor_samples_per_scan = 8192
    floor_history_capacity = 8_000_000
    wall_samples_per_scan = 4096
    wall_sample_capacity = 4_000_000
    map_height = 1.6
    grid_depth = max(1, int(math.ceil(map_height / resolution)))
    voxel_count = grid_width * grid_height * grid_depth
    vertical_layers = 64
    horizontal_rays = max(1, int(math.ceil(ray_count / vertical_layers)))
    sensor_height = 0.45
    elevation_min = math.radians(-7.0)
    elevation_max = math.radians(16.0)
    elevation_increment = (elevation_max - elevation_min) / max(1, vertical_layers - 1)
    world_wall_height = 1.45
    ring_count = 7
    ring_segments = 192
    ring_line_count = ring_count * ring_segments
    ray_line_count = 2048
    scan_line_capacity = max(ring_line_count, ray_line_count)
    obstacles_wp = wp.array(obstacles_np, dtype=wp.vec4, device=selected_device)
    ranges_wp = wp.zeros(ray_count, dtype=wp.float32, device=selected_device)
    occupancy_wp = wp.zeros(voxel_count, dtype=wp.int32, device=selected_device)
    floor_discovered_wp = wp.zeros(floor_cell_count, dtype=wp.int32, device=selected_device)
    floor_observation_count_wp = wp.zeros(
        floor_cell_count, dtype=wp.int32, device=selected_device
    )
    floor_sample_sequence_wp = wp.zeros(
        floor_cell_count, dtype=wp.int32, device=selected_device
    )
    floor_history_count_wp = wp.zeros(1, dtype=wp.int32, device=selected_device)
    wall_sample_count_wp = wp.zeros(1, dtype=wp.int32, device=selected_device)
    map_count_wp = wp.zeros(1, dtype=wp.int32, device=selected_device)

    renderer._switch_context()
    gl = pyglet.gl
    shared_program = ShaderProgram(
        Shader(_SLAM_GL_VERTEX_SHADER, "vertex"),
        Shader(_SLAM_GL_FRAGMENT_SHADER, "fragment"),
    )
    uniform_model = gl.glGetUniformLocation(shared_program.id, b"model")
    uniform_view = gl.glGetUniformLocation(shared_program.id, b"view")
    uniform_projection = gl.glGetUniformLocation(shared_program.id, b"projection")
    uniform_point_mode = gl.glGetUniformLocation(shared_program.id, b"point_mode")
    uniform_alpha = gl.glGetUniformLocation(shared_program.id, b"alpha")
    uniform_white_mode = gl.glGetUniformLocation(shared_program.id, b"white_mode")

    def create_shared_buffer(vertex_count: int):
        vao = gl.GLuint()
        vbo = gl.GLuint()
        gl.glGenVertexArrays(1, vao)
        gl.glGenBuffers(1, vbo)
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            vertex_count * 3 * np.dtype(np.float32).itemsize,
            None,
            gl.GL_DYNAMIC_DRAW,
        )
        gl.glVertexAttribPointer(
            0, 3, gl.GL_FLOAT, gl.GL_FALSE,
            3 * np.dtype(np.float32).itemsize,
            ctypes.c_void_p(0),
        )
        gl.glEnableVertexAttribArray(0)
        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        registered = wp.RegisteredGLBuffer(
            int(vbo.value),
            selected_device,
            fallback_to_copy=not selected_device.is_cuda,
        )
        return vao, vbo, registered

    def create_static_buffer(vertices: Any):
        contiguous = np.ascontiguousarray(vertices, dtype=np.float32)
        vao = gl.GLuint()
        vbo = gl.GLuint()
        gl.glGenVertexArrays(1, vao)
        gl.glGenBuffers(1, vbo)
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            contiguous.nbytes,
            contiguous.ctypes.data_as(ctypes.c_void_p),
            gl.GL_STATIC_DRAW,
        )
        gl.glVertexAttribPointer(
            0, 3, gl.GL_FLOAT, gl.GL_FALSE,
            3 * np.dtype(np.float32).itemsize,
            ctypes.c_void_p(0),
        )
        gl.glEnableVertexAttribArray(0)
        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        return vao, vbo, len(contiguous)

    map_vao, map_vbo, map_buffer = create_shared_buffer(capacity)
    scan_vao, scan_vbo, scan_buffer = create_shared_buffer(ray_count)
    floor_vao, floor_vbo, floor_buffer = create_shared_buffer(floor_cell_count)
    floor_history_vao, floor_history_vbo, floor_history_buffer = create_shared_buffer(
        floor_history_capacity
    )
    wall_samples_vao, wall_samples_vbo, wall_samples_buffer = create_shared_buffer(
        wall_sample_capacity
    )
    scan_lines_vao, scan_lines_vbo, scan_lines_buffer = create_shared_buffer(
        scan_line_capacity * 2
    )
    ghost_faces, ghost_edges = _ghost_obstacle_geometry(obstacles_np)
    ghost_faces_vao, ghost_faces_vbo, ghost_face_count = create_static_buffer(ghost_faces)
    ghost_edges_vao, ghost_edges_vbo, ghost_edge_count = create_static_buffer(ghost_edges)

    def matrix_pointer(matrix: Any):
        return np.asarray(matrix, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def draw_shared_slam_geometry() -> None:
        gl.glUseProgram(shared_program.id)
        gl.glUniformMatrix4fv(uniform_model, 1, gl.GL_FALSE, matrix_pointer(renderer._model_matrix))
        gl.glUniformMatrix4fv(uniform_view, 1, gl.GL_FALSE, matrix_pointer(renderer._view_matrix))
        gl.glUniformMatrix4fv(uniform_projection, 1, gl.GL_FALSE, matrix_pointer(renderer._projection_matrix))
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glUniform1f(uniform_point_mode, 0.0)
        gl.glUniform1f(uniform_white_mode, 0.0)
        if config.get("show_ground_truth"):
            culling_was_enabled = bool(gl.glIsEnabled(gl.GL_CULL_FACE))
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glDepthMask(gl.GL_FALSE)
            gl.glUniform1f(uniform_alpha, 0.13)
            gl.glBindVertexArray(ghost_faces_vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, ghost_face_count)
            gl.glUniform1f(uniform_alpha, 0.68)
            gl.glLineWidth(1.55)
            gl.glBindVertexArray(ghost_edges_vao)
            gl.glDrawArrays(gl.GL_LINES, 0, ghost_edge_count)
            gl.glDepthMask(gl.GL_TRUE)
            if culling_was_enabled:
                gl.glEnable(gl.GL_CULL_FACE)
        gl.glUniform1f(uniform_point_mode, 1.0)
        gl.glUniform1f(uniform_white_mode, 1.0)
        gl.glDepthMask(gl.GL_FALSE)
        # The regular floor-cell buffer is only the discovery index. Rendering it
        # would expose the fixed cell lattice and hide the samples accumulating.
        gl.glUniform1f(uniform_alpha, 0.055)
        gl.glPointSize(max(1.6, resolution * 22.0))
        gl.glBindVertexArray(floor_history_vao)
        gl.glDrawArrays(gl.GL_POINTS, 0, floor_history_capacity)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glUniform1f(uniform_point_mode, 0.0)
        gl.glUniform1f(uniform_white_mode, 0.0)
        if scan_visual_mode == "rays":
            gl.glUniform1f(uniform_alpha, 0.20)
            gl.glLineWidth(0.85)
            scan_line_count = ray_line_count
        else:
            gl.glUniform1f(uniform_alpha, 0.42)
            gl.glLineWidth(1.35)
            scan_line_count = ring_line_count
        gl.glBindVertexArray(scan_lines_vao)
        gl.glDrawArrays(gl.GL_LINES, 0, scan_line_count * 2)
        gl.glUniform1f(uniform_point_mode, 1.0)
        gl.glUniform1f(uniform_alpha, 0.55)
        gl.glPointSize(max(3.0, resolution * 48.0))
        gl.glBindVertexArray(map_vao)
        gl.glDrawArrays(gl.GL_POINTS, 0, capacity)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glUniform1f(uniform_alpha, 0.025)
        gl.glPointSize(2.2)
        gl.glBindVertexArray(wall_samples_vao)
        gl.glDrawArrays(gl.GL_POINTS, 0, wall_sample_capacity)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glBindVertexArray(0)
        gl.glDisable(gl.GL_BLEND)

    renderer.render_3d_callbacks.append(draw_shared_slam_geometry)

    def clear_shared_buffers() -> None:
        map_points = map_buffer.map(dtype=wp.vec3, shape=(capacity,))
        scan_points = scan_buffer.map(dtype=wp.vec3, shape=(ray_count,))
        floor_points = floor_buffer.map(dtype=wp.vec3, shape=(floor_cell_count,))
        floor_history_points = floor_history_buffer.map(
            dtype=wp.vec3, shape=(floor_history_capacity,)
        )
        wall_sample_points = wall_samples_buffer.map(
            dtype=wp.vec3, shape=(wall_sample_capacity,)
        )
        scan_lines = scan_lines_buffer.map(dtype=wp.vec3, shape=(scan_line_capacity, 2))
        try:
            wp.launch(_clear_points_kernel, dim=capacity, inputs=[map_points], device=selected_device)
            wp.launch(_clear_points_kernel, dim=ray_count, inputs=[scan_points], device=selected_device)
            wp.launch(
                _clear_points_kernel,
                dim=floor_cell_count,
                inputs=[floor_points],
                device=selected_device,
            )
            wp.launch(
                _clear_points_kernel,
                dim=floor_history_capacity,
                inputs=[floor_history_points],
                device=selected_device,
            )
            wp.launch(
                _clear_points_kernel,
                dim=wall_sample_capacity,
                inputs=[wall_sample_points],
                device=selected_device,
            )
            wp.launch(
                _clear_lines_kernel,
                dim=scan_line_capacity,
                inputs=[scan_lines],
                device=selected_device,
            )
            wp.synchronize_device(selected_device)
        finally:
            scan_lines_buffer.unmap()
            wall_samples_buffer.unmap()
            floor_history_buffer.unmap()
            floor_buffer.unmap()
            scan_buffer.unmap()
            map_buffer.unmap()

    clear_shared_buffers()

    scan_count = 0
    latest_pipeline_ms = 0.0
    started = time.perf_counter()
    last_scan = -1.0
    paused = False
    noise_enabled = True
    pause_started = 0.0
    paused_total = 0.0

    def reset() -> None:
        nonlocal last_scan, paused_total, scan_count, started
        occupancy_wp.zero_()
        floor_discovered_wp.zero_()
        floor_observation_count_wp.zero_()
        floor_sample_sequence_wp.zero_()
        floor_history_count_wp.zero_()
        wall_sample_count_wp.zero_()
        map_count_wp.zero_()
        clear_shared_buffers()
        scan_count = 0
        started = time.perf_counter()
        paused_total = 0.0
        last_scan = -1.0

    def on_key(symbol: int, modifiers: int):
        del modifiers
        nonlocal noise_enabled, pause_started, paused, paused_total, scan_visual_mode, started
        if symbol == pyglet.window.key.SPACE:
            scan_visual_mode = "rings" if scan_visual_mode == "rays" else "rays"
            return pyglet.event.EVENT_HANDLED
        if symbol == pyglet.window.key.N:
            noise_enabled = not noise_enabled
            paused = False
            reset()
            return pyglet.event.EVENT_HANDLED
        if symbol == pyglet.window.key.P:
            now = time.perf_counter()
            if paused:
                paused_total += now - pause_started
                paused = False
            else:
                pause_started = now
                paused = True
            return pyglet.event.EVENT_HANDLED
        if symbol == pyglet.window.key.R:
            paused = False
            reset()
            return pyglet.event.EVENT_HANDLED
        return None

    renderer.register_key_press_callback(on_key)
    route_vertices = [[float(x), float(y), 0.0] for x, y in route]
    world_vertices, world_indices = _rectangle_lines([[bounds[0], bounds[1], bounds[2], bounds[3]]])
    angle_min = -math.pi
    angle_increment = 2.0 * math.pi / horizontal_rays
    frame = 0

    while renderer.is_running():
        now = time.perf_counter()
        effective_now = pause_started if paused else now
        elapsed = max(0.0, effective_now - started - paused_total)
        rover_x, rover_y, rover_yaw, route_progress = route_pose(route, elapsed * speed)
        scan_slot = math.floor(elapsed * scan_hz)
        if not paused and scan_slot != last_scan:
            launch_started = time.perf_counter()
            map_points = map_buffer.map(dtype=wp.vec3, shape=(capacity,))
            hits = scan_buffer.map(dtype=wp.vec3, shape=(ray_count,))
            floor_points = floor_buffer.map(dtype=wp.vec3, shape=(floor_cell_count,))
            floor_history_points = floor_history_buffer.map(
                dtype=wp.vec3, shape=(floor_history_capacity,)
            )
            wall_sample_points = wall_samples_buffer.map(
                dtype=wp.vec3, shape=(wall_sample_capacity,)
            )
            ray_lines = None
            if config.get("accumulate_rays") and scan_visual_mode == "rays":
                ray_lines = scan_lines_buffer.map(
                    dtype=wp.vec3, shape=(scan_line_capacity, 2)
                )
            try:
                wp.launch(
                    _raycast_aabb_kernel,
                    dim=ray_count,
                    inputs=[
                        obstacles_wp, len(obstacles_np), rover_x, rover_y, sensor_height,
                        rover_yaw, angle_min, angle_increment,
                        elevation_min, elevation_increment, vertical_layers, range_max,
                        bounds[0], bounds[1], bounds[2], bounds[3],
                        world_wall_height,
                    ],
                    outputs=[hits, ranges_wp],
                    device=selected_device,
                )
                if noise_enabled:
                    wp.launch(
                        _apply_lidar_noise_kernel,
                        dim=ray_count,
                        inputs=[
                            hits, ranges_wp, rover_x, rover_y, sensor_height,
                            range_max, 1337 + scan_count * 7919,
                        ],
                        device=selected_device,
                    )
                wp.launch(
                    _accumulate_pointcloud_kernel,
                    dim=ray_count,
                    inputs=[
                        hits, bounds[0], bounds[1], resolution,
                        grid_width, grid_height, grid_depth, capacity,
                        occupancy_wp, map_count_wp, map_points,
                    ],
                    device=selected_device,
                )
                wp.launch(
                    _accumulate_floor_discovery_kernel,
                    dim=floor_cell_count,
                    inputs=[
                        obstacles_wp, len(obstacles_np), rover_x, rover_y, range_max,
                        bounds[0], bounds[1], resolution, grid_width,
                        floor_discovered_wp, floor_observation_count_wp, floor_points,
                    ],
                    device=selected_device,
                )
                wp.launch(
                    _jitter_new_floor_points_kernel,
                    dim=floor_cell_count,
                    inputs=[
                        floor_discovered_wp, floor_points, resolution,
                        1 if noise_enabled else 0, 9001 + scan_count * 3571,
                    ],
                    device=selected_device,
                )
                wp.launch(
                    _append_persistent_floor_samples_kernel,
                    dim=floor_samples_per_scan,
                    inputs=[
                        floor_discovered_wp, floor_points, floor_sample_sequence_wp,
                        floor_cell_count,
                        resolution, 1 if noise_enabled else 0,
                        floor_history_capacity, floor_history_count_wp,
                        12011 + scan_count * 4567, floor_history_points,
                    ],
                    device=selected_device,
                )
                wp.launch(
                    _append_persistent_wall_samples_kernel,
                    dim=wall_samples_per_scan,
                    inputs=[
                        hits, ray_count, wall_sample_capacity, wall_sample_count_wp,
                        15013 + scan_count * 6521, wall_sample_points,
                    ],
                    device=selected_device,
                )
                if ray_lines is not None:
                    wp.launch(
                        _build_slam_ray_lines_kernel,
                        dim=ray_line_count,
                        inputs=[
                            hits, ray_count, rover_x, rover_y, sensor_height, ray_lines,
                        ],
                        device=selected_device,
                    )
                wp.synchronize_device(selected_device)
            finally:
                if ray_lines is not None:
                    scan_lines_buffer.unmap()
                wall_samples_buffer.unmap()
                floor_history_buffer.unmap()
                floor_buffer.unmap()
                scan_buffer.unmap()
                map_buffer.unmap()
            latest_pipeline_ms = (time.perf_counter() - launch_started) * 1000.0
            scan_count += 1
            last_scan = scan_slot

        if config.get("accumulate_rays") and not paused and scan_visual_mode == "rings":
            scan_lines = scan_lines_buffer.map(
                dtype=wp.vec3, shape=(scan_line_capacity, 2)
            )
            try:
                wp.launch(
                    _build_radial_rings_kernel,
                    dim=ring_line_count,
                    inputs=[
                        rover_x, rover_y, elapsed, range_max,
                        ring_count, ring_segments, scan_lines,
                    ],
                    device=selected_device,
                )
                wp.synchronize_device(selected_device)
            finally:
                scan_lines_buffer.unmap()

        renderer.begin_frame(frame / max(1, fps))
        renderer.render_line_list(
            "known_world_boundary", world_vertices, world_indices,
            color=(0.38, 0.42, 0.48), radius=0.018,
        )
        if config.get("show_paths"):
            renderer.render_line_strip(
                "planned_route", route_vertices,
                color=(0.12, 0.48, 0.18), radius=0.022,
            )
        rover_rotation = (0.0, 0.0, math.sin(rover_yaw * 0.5), math.cos(rover_yaw * 0.5))
        renderer.render_box(
            "rover", pos=(rover_x, rover_y, 0.08), rot=rover_rotation,
            extents=(0.38, 0.24, 0.10), color=(1.0, 0.27, 0.06),
        )
        heading = [
            [rover_x, rover_y, 0.18],
            [rover_x + 0.75 * math.cos(rover_yaw), rover_y + 0.75 * math.sin(rover_yaw), 0.18],
        ]
        renderer.render_line_strip(
            "rover_heading", heading, color=(1.0, 0.92, 0.18), radius=0.025,
        )
        renderer.end_frame()
        renderer.window.set_caption(
            "Blacknode SLAM — rover discovery | "
            f"route {route_progress * 100:5.1f}% | scans {scan_count:,} | "
            f"rays {ray_count:,} ({vertical_layers} layers) @ {scan_hz:g} Hz | "
            f"GPU map {voxel_count:,} voxels + {floor_cell_count:,} floor cells | "
            f"floor +{floor_samples_per_scan:,}/scan permanent | "
            f"walls +{wall_samples_per_scan:,}/scan | "
            f"{('rays ' + format(ray_line_count, ',')) if scan_visual_mode == 'rays' else ('rings ' + str(ring_count) + 'x' + str(ring_segments))} | "
            f"noise {'ON' if noise_enabled else 'OFF'} | "
            f"pipeline {latest_pipeline_ms:.3f} ms | "
            f"{'PAUSED | ' if paused else ''}P: pause  R: reset"
        )
        frame += 1
    renderer.close()


@node(
    name="WarpSLAMDiscoveryViewer",
    hidden=True,
    component="spatial-processing",
    category=_CATEGORY,
    description=(
        "Start or stop a synthetic rover run where multi-layer Warp LiDAR scans "
        "reveal unknown obstacles into an accumulated 3D voxel point cloud."
    ),
    inputs={
        "action": Enum(["stop", "start"], default="stop"),
        "viewer_id": Text(default="warp_slam_discovery"),
        "device": Enum(["cuda:0", "cpu"], default="cuda:0"),
        "rays_per_scan": Int(default=1_000_000),
        "scan_hz": Float(default=12.0),
        "rover_speed_m_s": Float(default=1.2),
        "range_max_m": Float(default=10.0),
        "map_resolution_m": Float(default=0.08),
        "map_capacity": Int(default=250_000),
        "ray_history_capacity": Int(default=100_000),
        "fps": Int(default=30),
        "show_ground_truth": Bool(default=False),
        "show_paths": Bool(default=False),
        "accumulate_rays": Bool(default=True),
        "scan_visualization": Enum(["rays", "rings"], default="rays"),
    },
    outputs={"running": Bool, "viewer": Dict, "scenario": Dict, "report": Text},
)
def warp_slam_discovery_viewer(ctx: dict) -> dict:
    viewer_id = str(ctx.get("viewer_id") or "warp_slam_discovery").strip()
    scenario = slam_demo_scenario(ctx)
    if str(ctx.get("action") or "stop") == "stop":
        stopped = viewer_rt.stop_viewer(viewer_id)
        return {
            "running": False,
            "viewer": {"viewer_id": viewer_id, "state": "stopped"},
            "scenario": scenario,
            "report": f"Warp SLAM viewer stopped {int(stopped.get('stopped') or 0)} process(es)",
        }
    started = viewer_rt.start_slam_viewer(
        viewer_id=viewer_id,
        config=scenario,
        device=str(ctx.get("device") or "cuda:0"),
    )
    if not started.get("ok"):
        reason = str(started.get("error") or "could not start Warp SLAM viewer")
        return {
            "running": False,
            "viewer": {"viewer_id": viewer_id, "state": "failed", "error": reason},
            "scenario": scenario,
            "report": reason,
        }
    return {
        "running": True,
        "viewer": {
            "kind": "blacknode.slam-discovery-viewer",
            "schema_version": 1,
            "viewer_id": viewer_id,
            "state": "running",
            "device": str(ctx.get("device") or "cuda:0"),
            "memory_path": (
                "cuda-opengl-registered-buffers"
                if str(ctx.get("device") or "cuda:0").startswith("cuda")
                else "cpu-registered-buffer-fallback"
            ),
            "controls": {
                "left_drag": "rotate camera",
                "wasd_or_arrows": "move camera",
                "mouse_wheel": "zoom camera",
                "space": "toggle ray beams or radar rings",
                "n": "toggle LiDAR noise and reset map",
                "p": "pause or resume",
                "r": "clear the map and restart",
                "escape": "close",
            },
        },
        "scenario": scenario,
        "report": (
            "Warp SLAM discovery running "
            f"{'GPU-to-GPU' if str(ctx.get('device') or 'cuda:0').startswith('cuda') else 'with CPU fallback'}: "
            f"{scenario['rays_per_scan']:,} rays at "
            f"{scenario['scan_hz']:g} Hz across {len(scenario['obstacles'])} unknown obstacles"
        ),
    }
