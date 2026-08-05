"""Persistent bounded TSDF integration and surface extraction with NVIDIA Warp."""
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
    def _integrate_tsdf_kernel(
        points: wp.array(dtype=wp.vec3),
        colors: wp.array(dtype=wp.vec3),
        sensor_origin: wp.vec3,
        grid_origin: wp.vec3,
        voxel_size: wp.float32,
        dim_x: wp.int32,
        dim_y: wp.int32,
        dim_z: wp.int32,
        truncation: wp.float32,
        samples_per_ray: wp.int32,
        integrate_color: wp.int32,
        sdf_sum: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        color_r: wp.array(dtype=wp.float32),
        color_g: wp.array(dtype=wp.float32),
        color_b: wp.array(dtype=wp.float32),
        color_weights: wp.array(dtype=wp.float32),
    ):
        thread_index = wp.tid()
        point_index = thread_index // samples_per_ray
        sample_index = thread_index - point_index * samples_per_ray
        surface = points[point_index]
        ray = surface - sensor_origin
        ray_length = wp.length(ray)
        if ray_length <= 1.0e-6:
            return
        direction = ray / ray_length
        fraction = wp.float32(sample_index) / wp.float32(wp.max(samples_per_ray - 1, 1))
        signed_distance = -truncation + fraction * truncation * 2.0
        sample = surface - direction * signed_distance
        ix = wp.int32(wp.floor((sample[0] - grid_origin[0]) / voxel_size))
        iy = wp.int32(wp.floor((sample[1] - grid_origin[1]) / voxel_size))
        iz = wp.int32(wp.floor((sample[2] - grid_origin[2]) / voxel_size))
        if ix < 0 or iy < 0 or iz < 0 or ix >= dim_x or iy >= dim_y or iz >= dim_z:
            return
        index = ix + dim_x * (iy + dim_y * iz)
        wp.atomic_add(sdf_sum, index, wp.clamp(signed_distance / truncation, -1.0, 1.0))
        wp.atomic_add(weights, index, 1.0)
        if integrate_color != 0 and wp.abs(signed_distance) <= voxel_size:
            color = colors[point_index]
            wp.atomic_add(color_r, index, color[0])
            wp.atomic_add(color_g, index, color[1])
            wp.atomic_add(color_b, index, color[2])
            wp.atomic_add(color_weights, index, 1.0)


    @wp.kernel
    def _extract_surface_kernel(
        sdf_sum: wp.array(dtype=wp.float32),
        weights: wp.array(dtype=wp.float32),
        color_r: wp.array(dtype=wp.float32),
        color_g: wp.array(dtype=wp.float32),
        color_b: wp.array(dtype=wp.float32),
        color_weights: wp.array(dtype=wp.float32),
        grid_origin: wp.vec3,
        voxel_size: wp.float32,
        dim_x: wp.int32,
        dim_y: wp.int32,
        dim_z: wp.int32,
        iso_level: wp.float32,
        surface_band: wp.float32,
        minimum_weight: wp.float32,
        maximum_points: wp.int32,
        positions: wp.array(dtype=wp.vec3),
        normals: wp.array(dtype=wp.vec3),
        colors: wp.array(dtype=wp.vec3),
        confidence: wp.array(dtype=wp.float32),
        surface_count: wp.array(dtype=wp.int32),
        observed_count: wp.array(dtype=wp.int32),
    ):
        index = wp.tid()
        weight = weights[index]
        if weight > 0.0:
            wp.atomic_add(observed_count, 0, 1)
        if weight < minimum_weight:
            return
        sdf = sdf_sum[index] / weight
        if wp.abs(sdf - iso_level) > surface_band:
            return
        output_index = wp.atomic_add(surface_count, 0, 1)
        if output_index >= maximum_points:
            return
        plane = dim_x * dim_y
        iz = index // plane
        remainder = index - iz * plane
        iy = remainder // dim_x
        ix = remainder - iy * dim_x
        positions[output_index] = grid_origin + wp.vec3(
            (wp.float32(ix) + 0.5) * voxel_size,
            (wp.float32(iy) + 0.5) * voxel_size,
            (wp.float32(iz) + 0.5) * voxel_size,
        )
        gradient = wp.vec3(0.0, 0.0, 0.0)
        if ix > 0 and ix + 1 < dim_x:
            left = index - 1
            right = index + 1
            if weights[left] > 0.0 and weights[right] > 0.0:
                gradient[0] = sdf_sum[right] / weights[right] - sdf_sum[left] / weights[left]
        if iy > 0 and iy + 1 < dim_y:
            down = index - dim_x
            up = index + dim_x
            if weights[down] > 0.0 and weights[up] > 0.0:
                gradient[1] = sdf_sum[up] / weights[up] - sdf_sum[down] / weights[down]
        if iz > 0 and iz + 1 < dim_z:
            behind = index - plane
            ahead = index + plane
            if weights[behind] > 0.0 and weights[ahead] > 0.0:
                gradient[2] = sdf_sum[ahead] / weights[ahead] - sdf_sum[behind] / weights[behind]
        gradient_length = wp.length(gradient)
        normals[output_index] = gradient / gradient_length if gradient_length > 1.0e-6 else wp.vec3(-1.0, 0.0, 0.0)
        color_weight = color_weights[index]
        colors[output_index] = (
            wp.vec3(color_r[index], color_g[index], color_b[index]) / color_weight
            if color_weight > 0.0
            else wp.vec3(0.08, 0.78, 0.92)
        )
        confidence[output_index] = wp.min(1.0, weight / wp.max(minimum_weight * 4.0, 1.0))


def _error(message: str, *, device: str = "") -> dict[str, Any]:
    return {
        "ok": False,
        "points": [],
        "colors": [],
        "normals": [],
        "confidence": [],
        "report": {"state": "error", "device": device, "error": message},
    }


class WarpTSDFVolume:
    """One managed, device-resident TSDF volume with bounded memory."""

    def __init__(self, stage: dict[str, Any], *, device: str) -> None:
        if wp is None:
            raise RuntimeError("NVIDIA Warp is not installed; install warp-lang>=1.15")
        if np is None:
            raise RuntimeError("NumPy is not installed; install numpy>=1.24")
        self.device = wp.get_device(device)
        self.radius_m = max(0.25, float(stage.get("volume_radius_m") or 3.0))
        self.voxel_size_m = max(0.01, float(stage.get("voxel_size_m") or 0.08))
        self.origin = np.asarray([
            float(stage.get("volume_origin_x_m") or 0.0) - self.radius_m,
            float(stage.get("volume_origin_y_m") or 0.0) - self.radius_m,
            float(stage.get("volume_origin_z_m") if stage.get("volume_origin_z_m") is not None else -1.0) - self.radius_m,
        ], dtype=np.float32)
        dimension = max(2, int(math.ceil((self.radius_m * 2.0) / self.voxel_size_m)))
        self.dimensions = (dimension, dimension, dimension)
        self.voxel_count = dimension ** 3
        maximum_voxels = max(8_000, min(8_000_000, int(stage.get("maximum_voxels") or 2_000_000)))
        if self.voxel_count > maximum_voxels:
            raise ValueError(
                f"TSDF volume needs {self.voxel_count:,} voxels; increase voxel_size_m "
                f"or maximum_voxels (current limit {maximum_voxels:,})"
            )
        self.sdf_sum = wp.zeros(self.voxel_count, dtype=wp.float32, device=self.device)
        self.weights = wp.zeros(self.voxel_count, dtype=wp.float32, device=self.device)
        self.color_r = wp.zeros(self.voxel_count, dtype=wp.float32, device=self.device)
        self.color_g = wp.zeros(self.voxel_count, dtype=wp.float32, device=self.device)
        self.color_b = wp.zeros(self.voxel_count, dtype=wp.float32, device=self.device)
        self.color_weights = wp.zeros(self.voxel_count, dtype=wp.float32, device=self.device)
        self.output_capacity = 0
        self.positions = None
        self.normals = None
        self.colors = None
        self.confidence = None
        self.surface_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.observed_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.frames_integrated = 0
        self.samples_integrated = 0
        self.integration_ms = 0.0

    def compatible(self, stage: dict[str, Any], device: str) -> bool:
        requested_origin_z = float(
            stage.get("volume_origin_z_m") if stage.get("volume_origin_z_m") is not None else -1.0
        )
        expected_origin = np.asarray([
            float(stage.get("volume_origin_x_m") or 0.0) - max(0.25, float(stage.get("volume_radius_m") or 3.0)),
            float(stage.get("volume_origin_y_m") or 0.0) - max(0.25, float(stage.get("volume_radius_m") or 3.0)),
            requested_origin_z - max(0.25, float(stage.get("volume_radius_m") or 3.0)),
        ], dtype=np.float32)
        return (
            str(self.device) == str(wp.get_device(device))
            and abs(self.radius_m - max(0.25, float(stage.get("volume_radius_m") or 3.0))) < 1.0e-9
            and abs(self.voxel_size_m - max(0.01, float(stage.get("voxel_size_m") or 0.08))) < 1.0e-9
            and bool(np.allclose(self.origin, expected_origin))
        )

    def integrate(
        self,
        point_cloud: dict[str, Any],
        *,
        pose: dict[str, Any],
        stage: dict[str, Any],
    ) -> dict[str, Any]:
        points = np.asarray(point_cloud.get("points_xyz") or [], dtype=np.float32).reshape((-1, 3))
        colors = np.asarray(point_cloud.get("colors_rgb") or [], dtype=np.float32).reshape((-1, 3))
        if not len(points):
            return _error("TSDF integration received no valid depth points", device=str(self.device))
        if len(colors) != len(points):
            colors = np.tile(np.asarray([[0.08, 0.78, 0.92]], dtype=np.float32), (len(points), 1))
        yaw = float(pose.get("yaw_rad") or 0.0)
        cosine, sine = math.cos(yaw), math.sin(yaw)
        world_points = points.copy()
        world_points[:, 0] = float(pose.get("x_m") or 0.0) + cosine * points[:, 0] - sine * points[:, 1]
        world_points[:, 1] = float(pose.get("y_m") or 0.0) + sine * points[:, 0] + cosine * points[:, 1]
        world_points[:, 2] = float(pose.get("z_m") or 0.0) + points[:, 2]
        extrinsics = point_cloud.get("processing", {}).get("sensor_extrinsics", {})
        sensor_origin = np.asarray([
            float(pose.get("x_m") or 0.0) + cosine * float(extrinsics.get("x_m") or 0.0) - sine * float(extrinsics.get("y_m") or 0.0),
            float(pose.get("y_m") or 0.0) + sine * float(extrinsics.get("x_m") or 0.0) + cosine * float(extrinsics.get("y_m") or 0.0),
            float(pose.get("z_m") or 0.0) + float(extrinsics.get("z_m") or 0.0),
        ], dtype=np.float32)
        samples_per_ray = max(3, min(15, int(stage.get("samples_per_ray") or 7)))
        if samples_per_ray % 2 == 0:
            samples_per_ray += 1
        truncation = max(self.voxel_size_m, float(stage.get("truncation_m") or self.voxel_size_m * 3.0))
        points_wp = wp.array(np.ascontiguousarray(world_points), dtype=wp.vec3, device=self.device)
        colors_wp = wp.array(np.ascontiguousarray(colors), dtype=wp.vec3, device=self.device)
        started = time.perf_counter()
        wp.launch(
            _integrate_tsdf_kernel,
            dim=len(world_points) * samples_per_ray,
            inputs=[
                points_wp,
                colors_wp,
                wp.vec3(*sensor_origin.tolist()),
                wp.vec3(*self.origin.tolist()),
                self.voxel_size_m,
                *self.dimensions,
                truncation,
                samples_per_ray,
                1 if stage.get("integrate_color", True) else 0,
            ],
            outputs=[
                self.sdf_sum,
                self.weights,
                self.color_r,
                self.color_g,
                self.color_b,
                self.color_weights,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.frames_integrated += 1
        self.samples_integrated += len(world_points) * samples_per_ray
        self.integration_ms = elapsed_ms
        return {
            "ok": True,
            "world_points": world_points.astype(float).tolist(),
            "colors": colors.astype(float).tolist(),
            "sensor_origin": sensor_origin.astype(float).tolist(),
            "report": {
                "state": "ready",
                "backend": "warp",
                "device": str(self.device),
                "input_points": int(len(world_points)),
                "samples_per_ray": samples_per_ray,
                "work_items": int(len(world_points) * samples_per_ray),
                "integration_ms": float(elapsed_ms),
                "frames_integrated": self.frames_integrated,
                "samples_integrated": self.samples_integrated,
                "voxel_size_m": self.voxel_size_m,
                "truncation_m": truncation,
                "voxel_count": self.voxel_count,
                "dimensions": list(self.dimensions),
                "volume_origin_m": self.origin.astype(float).tolist(),
                "volume_radius_m": self.radius_m,
                "color_integrated": bool(stage.get("integrate_color", True)),
            },
        }

    def extract(self, stage: dict[str, Any]) -> dict[str, Any]:
        iso_level = max(-1.0, min(1.0, float(stage.get("iso_level") or 0.0)))
        surface_band = max(0.01, min(1.0, float(stage.get("surface_band") or 0.2)))
        minimum_weight = max(1.0, float(stage.get("minimum_weight") or 1.0))
        maximum_points = max(64, min(250_000, int(stage.get("maximum_points") or 60_000)))
        if self.output_capacity != maximum_points:
            self.output_capacity = maximum_points
            self.positions = wp.zeros(maximum_points, dtype=wp.vec3, device=self.device)
            self.normals = wp.zeros(maximum_points, dtype=wp.vec3, device=self.device)
            self.colors = wp.zeros(maximum_points, dtype=wp.vec3, device=self.device)
            self.confidence = wp.zeros(maximum_points, dtype=wp.float32, device=self.device)
        self.surface_count.zero_()
        self.observed_count.zero_()
        started = time.perf_counter()
        wp.launch(
            _extract_surface_kernel,
            dim=self.voxel_count,
            inputs=[
                self.sdf_sum,
                self.weights,
                self.color_r,
                self.color_g,
                self.color_b,
                self.color_weights,
                wp.vec3(*self.origin.tolist()),
                self.voxel_size_m,
                *self.dimensions,
                iso_level,
                surface_band,
                minimum_weight,
                maximum_points,
            ],
            outputs=[
                self.positions, self.normals, self.colors, self.confidence,
                self.surface_count, self.observed_count,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        surface_voxels = int(self.surface_count.numpy()[0])
        observed_voxels = int(self.observed_count.numpy()[0])
        display_count = min(surface_voxels, maximum_points)
        display_limited = surface_voxels > maximum_points
        if display_count:
            positions_wp = wp.empty(display_count, dtype=wp.vec3, device=self.device)
            normals_wp = wp.empty(display_count, dtype=wp.vec3, device=self.device)
            colors_wp = wp.empty(display_count, dtype=wp.vec3, device=self.device)
            confidence_wp = wp.empty(display_count, dtype=wp.float32, device=self.device)
            wp.copy(positions_wp, self.positions, count=display_count)
            wp.copy(normals_wp, self.normals, count=display_count)
            wp.copy(colors_wp, self.colors, count=display_count)
            wp.copy(confidence_wp, self.confidence, count=display_count)
            positions = positions_wp.numpy()
            normals = normals_wp.numpy()
            colors = colors_wp.numpy()
            confidence = confidence_wp.numpy()
        else:
            positions = np.empty((0, 3), dtype=np.float32)
            normals = np.empty((0, 3), dtype=np.float32)
            colors = np.empty((0, 3), dtype=np.float32)
            confidence = np.empty((0,), dtype=np.float32)
        extraction_ms = (time.perf_counter() - started) * 1000.0
        return {
            "ok": True,
            "points": positions.astype(float).tolist(),
            "normals": normals.astype(float).tolist(),
            "colors": colors.astype(float).tolist(),
            "confidence": confidence.astype(float).tolist(),
            "report": {
                "state": "ready",
                "backend": "warp",
                "device": str(self.device),
                "surface_voxels": surface_voxels,
                "display_points": display_count,
                "display_limited": display_limited,
                "observed_voxels": observed_voxels,
                "allocated_voxels": self.voxel_count,
                "iso_level": iso_level,
                "surface_band": surface_band,
                "minimum_weight": minimum_weight,
                "extraction_ms": float(extraction_ms),
                "frames_integrated": self.frames_integrated,
                "integration_ms": self.integration_ms,
                "voxel_size_m": self.voxel_size_m,
                "dimensions": list(self.dimensions),
            },
        }
