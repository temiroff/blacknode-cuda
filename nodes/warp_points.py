"""NVIDIA Warp point filtering, sensor transforms, colors, and debug viewer."""
from __future__ import annotations

import copy
import math
import time
from typing import Any, Callable

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency contract
    np = None

try:
    import warp as wp
except Exception:  # pragma: no cover - package must still load without Warp
    wp = None

from blacknode.node import Bool, Dict, Enum, Float, Int, List, Text, node

from . import warp_viewer_runtime as viewer_rt


_CATEGORY = "NVIDIA CUDA"


if wp is not None:
    @wp.kernel
    def _laser_scan_points_kernel(
        ranges: wp.array(dtype=wp.float32),
        angle_min: wp.float32,
        angle_increment: wp.float32,
        filter_min: wp.float32,
        filter_max: wp.float32,
        sensor_x: wp.float32,
        sensor_y: wp.float32,
        sensor_yaw: wp.float32,
        stride: wp.int32,
        raw_points: wp.array(dtype=wp.vec3),
        filtered_points: wp.array(dtype=wp.vec3),
        filtered_colors: wp.array(dtype=wp.vec3),
        raw_valid: wp.array(dtype=wp.int32),
        filtered_valid: wp.array(dtype=wp.int32),
    ):
        index = wp.tid()
        distance = ranges[index]
        valid_raw = distance > 0.0 and distance < 1.0e20
        if valid_raw:
            angle = angle_min + wp.float32(index) * angle_increment + sensor_yaw
            point = wp.vec3(
                sensor_x + distance * wp.cos(angle),
                sensor_y + distance * wp.sin(angle),
                0.0,
            )
            raw_points[index] = point
            raw_valid[index] = 1
            valid_filtered = distance >= filter_min and distance <= filter_max and index % stride == 0
            if valid_filtered:
                filtered_points[index] = point
                span = wp.max(filter_max - filter_min, 1.0e-6)
                distance_fraction = wp.clamp((distance - filter_min) / span, 0.0, 1.0)
                filtered_colors[index] = wp.vec3(0.0, 1.0 - 0.55 * distance_fraction, 1.0)
                filtered_valid[index] = 1

    @wp.kernel
    def _reveal_points_kernel(
        source: wp.array(dtype=wp.vec3),
        reveal_count: wp.int32,
        hidden_z: wp.float32,
        visible: wp.array(dtype=wp.vec3),
    ):
        index = wp.tid()
        if index < reveal_count:
            visible[index] = source[index]
        else:
            visible[index] = wp.vec3(0.0, 0.0, hidden_z)


def _numpy_scan_baseline(
    ranges_np: Any,
    *,
    angle_min: float,
    angle_increment: float,
    minimum: float,
    maximum: float,
    stride: int,
    sensor_pose: tuple[float, float, float],
    repeats: int = 3,
) -> tuple[Any, float]:
    """Return a warmed NumPy reference and median compute time."""
    if np is None:
        raise RuntimeError("NumPy is not installed")
    count = int(ranges_np.size)

    def calculate() -> Any:
        indices = np.arange(count, dtype=np.int64)
        angles = (
            np.float32(angle_min)
            + indices.astype(np.float32) * np.float32(angle_increment)
            + np.float32(sensor_pose[2])
        )
        valid = (
            np.isfinite(ranges_np)
            & (ranges_np >= np.float32(minimum))
            & (ranges_np <= np.float32(maximum))
            & ((indices % max(1, stride)) == 0)
        )
        points = np.empty((count, 3), dtype=np.float32)
        points[:, 0] = np.float32(sensor_pose[0]) + ranges_np * np.cos(angles)
        points[:, 1] = np.float32(sensor_pose[1]) + ranges_np * np.sin(angles)
        points[:, 2] = 0.0
        return points[valid]

    calculate()  # Warm the vectorized CPU path before measurement.
    timings: list[float] = []
    result = None
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        result = calculate()
        timings.append((time.perf_counter() - started) * 1000.0)
    return result, float(np.median(np.asarray(timings, dtype=np.float64)))


def _error(message: str, *, device: str = "") -> dict[str, Any]:
    return {
        "ok": False,
        "raw_points": [],
        "filtered_points": [],
        "colors": [],
        "point_cloud": {},
        "raw_count": 0,
        "filtered_count": 0,
        "device": device,
        "kernel_ms": 0.0,
        "report": {"state": "unavailable", "device": device, "error": message},
    }


def process_laser_scan(
    scan: dict,
    *,
    device: str = "cuda:0",
    filter_min_m: float = 0.1,
    filter_max_m: float = 12.0,
    stride: int = 1,
    sensor_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
    include_raw_points: bool = True,
    compare_numpy: bool = False,
) -> dict[str, Any]:
    """Run one LaserScan conversion/filter/rigid transform through a Warp kernel."""
    if wp is None:
        return _error("NVIDIA Warp is not installed; install warp-lang>=1.15", device=device)
    if np is None:
        return _error("NumPy is not installed; install numpy>=1.24", device=device)
    if not isinstance(scan, dict) or scan.get("kind") != "blacknode.laser-scan-stream":
        return _error("laser_scan must be a blacknode.laser-scan-stream", device=device)
    ranges_value = scan.get("ranges") if isinstance(scan.get("ranges"), list) else []
    if not ranges_value:
        return _error("laser_scan contains no ranges", device=device)
    try:
        selected_device = wp.get_device(device)
    except Exception as exc:
        return _error(f"Warp device {device!r} is unavailable ({type(exc).__name__}: {exc})", device=device)
    minimum = max(float(scan.get("range_min") or 0.0), float(filter_min_m))
    sensor_maximum = float(scan.get("range_max") or 0.0)
    requested_maximum = max(minimum, float(filter_max_m))
    maximum = min(sensor_maximum, requested_maximum) if sensor_maximum > 0.0 else requested_maximum
    clean_stride = max(1, int(stride))
    pose = tuple(float(value) for value in sensor_pose)
    try:
        ranges_np = np.asarray(ranges_value, dtype=np.float32)
        count = int(ranges_np.size)
        ranges_wp = wp.array(ranges_np, dtype=wp.float32, device=selected_device)
        raw_wp = wp.zeros(count, dtype=wp.vec3, device=selected_device)
        filtered_wp = wp.zeros(count, dtype=wp.vec3, device=selected_device)
        colors_wp = wp.zeros(count, dtype=wp.vec3, device=selected_device)
        raw_valid_wp = wp.zeros(count, dtype=wp.int32, device=selected_device)
        filtered_valid_wp = wp.zeros(count, dtype=wp.int32, device=selected_device)
        launch_inputs = [
            ranges_wp,
            float(scan.get("angle_min") or 0.0),
            float(scan.get("angle_increment") or 0.0),
            minimum,
            maximum,
            pose[0],
            pose[1],
            pose[2],
            clean_stride,
        ]
        launch_outputs = [raw_wp, filtered_wp, colors_wp, raw_valid_wp, filtered_valid_wp]
        if compare_numpy:
            wp.launch(
                _laser_scan_points_kernel,
                dim=count,
                inputs=launch_inputs,
                outputs=launch_outputs,
                device=selected_device,
            )
            wp.synchronize_device(selected_device)
        warp_timings: list[float] = []
        for _ in range(3 if compare_numpy else 1):
            start = time.perf_counter()
            wp.launch(
                _laser_scan_points_kernel,
                dim=count,
                inputs=launch_inputs,
                outputs=launch_outputs,
                device=selected_device,
            )
            wp.synchronize_device(selected_device)
            warp_timings.append((time.perf_counter() - start) * 1000.0)
        kernel_ms = float(np.median(np.asarray(warp_timings, dtype=np.float64)))
        filtered_values = filtered_wp.numpy()
        color_values = colors_wp.numpy()
        filtered_mask = filtered_valid_wp.numpy().astype(bool)
        if include_raw_points:
            raw_values = raw_wp.numpy()
            raw_mask = raw_valid_wp.numpy().astype(bool)
        else:
            raw_values = None
            raw_mask = None
    except Exception as exc:
        return _error(f"Warp LaserScan processing failed ({type(exc).__name__}: {exc})", device=device)

    raw_count = int(np.count_nonzero(
        np.isfinite(ranges_np) & (ranges_np > 0.0) & (ranges_np < 1.0e20)
    ))
    raw_points = (
        raw_values[raw_mask].astype(float).tolist()
        if raw_values is not None and raw_mask is not None
        else []
    )
    filtered_points = filtered_values[filtered_mask].astype(float).tolist()
    colors = color_values[filtered_mask].astype(float).tolist()
    numpy_ms: float | None = None
    max_abs_error: float | None = None
    warp_end_to_end_ms: float | None = None
    if compare_numpy:
        try:
            numpy_points, numpy_ms = _numpy_scan_baseline(
                ranges_np,
                angle_min=float(scan.get("angle_min") or 0.0),
                angle_increment=float(scan.get("angle_increment") or 0.0),
                minimum=minimum,
                maximum=maximum,
                stride=clean_stride,
                sensor_pose=pose,
            )
            warp_points_np = filtered_values[filtered_mask]
            if numpy_points.shape == warp_points_np.shape and numpy_points.size:
                max_abs_error = float(np.max(np.abs(numpy_points - warp_points_np)))
            elif numpy_points.size == 0 and warp_points_np.size == 0:
                max_abs_error = 0.0

            def warp_end_to_end() -> Any:
                pipeline_ranges = wp.array(ranges_np, dtype=wp.float32, device=selected_device)
                pipeline_raw = wp.zeros(count, dtype=wp.vec3, device=selected_device)
                pipeline_filtered = wp.zeros(count, dtype=wp.vec3, device=selected_device)
                pipeline_colors = wp.zeros(count, dtype=wp.vec3, device=selected_device)
                pipeline_raw_valid = wp.zeros(count, dtype=wp.int32, device=selected_device)
                pipeline_filtered_valid = wp.zeros(count, dtype=wp.int32, device=selected_device)
                wp.launch(
                    _laser_scan_points_kernel,
                    dim=count,
                    inputs=[
                        pipeline_ranges,
                        float(scan.get("angle_min") or 0.0),
                        float(scan.get("angle_increment") or 0.0),
                        minimum,
                        maximum,
                        pose[0],
                        pose[1],
                        pose[2],
                        clean_stride,
                    ],
                    outputs=[
                        pipeline_raw,
                        pipeline_filtered,
                        pipeline_colors,
                        pipeline_raw_valid,
                        pipeline_filtered_valid,
                    ],
                    device=selected_device,
                )
                wp.synchronize_device(selected_device)
                pipeline_values = pipeline_filtered.numpy()
                pipeline_mask = pipeline_filtered_valid.numpy().astype(bool)
                return pipeline_values[pipeline_mask]

            warp_end_to_end()  # Warm allocator, transfers, and compaction.
            pipeline_timings: list[float] = []
            for _ in range(3):
                pipeline_started = time.perf_counter()
                warp_end_to_end()
                pipeline_timings.append((time.perf_counter() - pipeline_started) * 1000.0)
            warp_end_to_end_ms = float(np.median(np.asarray(pipeline_timings, dtype=np.float64)))
        except Exception:
            numpy_ms = None
            max_abs_error = None
            warp_end_to_end_ms = None
    report = {
        "state": "ready",
        "implementation": "NVIDIA Warp kernel",
        "device": str(selected_device),
        "input_count": count,
        "raw_count": raw_count,
        "raw_output_count": len(raw_points),
        "filtered_count": len(filtered_points),
        "removed_count": raw_count - len(filtered_points),
        "filter_min_m": minimum,
        "filter_max_m": maximum,
        "downsample_stride": clean_stride,
        "sensor_pose": {"x_m": pose[0], "y_m": pose[1], "yaw_rad": pose[2]},
        "kernel_ms": round(kernel_ms, 4),
        "benchmark": {
            "dtype": "float32",
            "sample_count": count,
            "warmup_runs": 1 if compare_numpy else 0,
            "measured_runs": 3 if compare_numpy else 1,
            "warp_gpu_ms": round(kernel_ms, 4),
            "numpy_cpu_ms": round(numpy_ms, 4) if numpy_ms is not None else None,
            "warp_kernel_speedup": round(numpy_ms / kernel_ms, 3)
            if numpy_ms is not None and kernel_ms > 0.0 else None,
            "warp_end_to_end_ms": round(warp_end_to_end_ms, 4)
            if warp_end_to_end_ms is not None else None,
            "end_to_end_speedup": round(numpy_ms / warp_end_to_end_ms, 3)
            if numpy_ms is not None and warp_end_to_end_ms is not None and warp_end_to_end_ms > 0.0
            else None,
            "max_abs_error_m": max_abs_error,
        },
    }
    point_cloud = {
        "kind": "blacknode.point-cloud-frame",
        "schema_version": 1,
        "frame": "base_link",
        "source_frame": str(scan.get("frame") or "laser"),
        "source_time_ns": int(scan.get("source_time_ns") or 0),
        "receive_time_ns": time.time_ns(),
        "point_count": len(filtered_points),
        "points_xyz": filtered_points,
        "colors_rgb": colors,
        "processing": copy.deepcopy(report),
    }
    return {
        "ok": True,
        "raw_points": raw_points,
        "filtered_points": filtered_points,
        "colors": colors,
        "point_cloud": point_cloud,
        "raw_count": raw_count,
        "filtered_count": len(filtered_points),
        "device": str(selected_device),
        "kernel_ms": round(kernel_ms, 4),
        "report": report,
    }


@node(
    name="WarpLaserScanFilter",
    component="spatial-processing",
    category=_CATEGORY,
    description=(
        "Convert LaserScan ranges to XY points, filter and downsample them, "
        "transform sensor coordinates into base_link, and color by distance in one Warp kernel."
    ),
    inputs={
        "laser_scan": Dict,
        "device": Enum(["cuda:0", "cpu"], default="cuda:0"),
        "filter_min_m": Float(default=0.1),
        "filter_max_m": Float(default=12.0),
        "downsample_stride": Int(default=1),
        "sensor_x_m": Float(default=0.0),
        "sensor_y_m": Float(default=0.0),
        "sensor_yaw_rad": Float(default=0.0),
        "include_raw_points": Bool(default=True),
        "compare_numpy": Bool(default=False),
    },
    outputs={
        "ok": Bool,
        "raw_points": List,
        "filtered_points": List,
        "colors": List,
        "point_cloud": Dict,
        "raw_count": Int,
        "filtered_count": Int,
        "device": Text,
        "kernel_ms": Float,
        "report": Dict,
    },
)
def warp_laser_scan_filter(ctx: dict) -> dict:
    return process_laser_scan(
        ctx.get("laser_scan") if isinstance(ctx.get("laser_scan"), dict) else {},
        device=str(ctx.get("device") or "cuda:0"),
        filter_min_m=max(0.0, float(ctx.get("filter_min_m") or 0.1)),
        filter_max_m=max(0.0, float(ctx.get("filter_max_m") or 12.0)),
        stride=max(1, int(ctx.get("downsample_stride") or 1)),
        sensor_pose=(
            float(ctx.get("sensor_x_m") or 0.0),
            float(ctx.get("sensor_y_m") or 0.0),
            float(ctx.get("sensor_yaw_rad") or 0.0),
        ),
        include_raw_points=bool(ctx.get("include_raw_points", True)),
        compare_numpy=bool(ctx.get("compare_numpy", False)),
    )


def run_viewer_loop(
    *,
    scan_source: Callable[[], dict],
    device: str,
    filter_min_m: float,
    filter_max_m: float,
    stride: int,
    sensor_pose: tuple[float, float, float],
    show_raw: bool,
    show_filtered: bool,
    point_radius: float,
    fps: int,
    animate_scan: bool = True,
    scan_hz: float = 0.25,
    show_rays: bool = True,
    ray_trail_count: int = 96,
    accumulate_hits: bool = True,
    compare_numpy: bool = False,
    title: str,
) -> None:
    """Render the latest scan until the interactive Warp window closes."""
    if wp is None:
        raise RuntimeError("NVIDIA Warp is not installed; install warp-lang>=1.15")
    import pyglet
    import warp.render

    renderer = warp.render.OpenGLRenderer(
        title=title,
        fps=fps,
        up_axis="Z",
        screen_width=1100,
        screen_height=800,
        near_plane=0.01,
        far_plane=100.0,
        camera_pos=(0.0, 0.0, 12.0),
        camera_front=(0.0, 0.0, -1.0),
        camera_up=(0.0, 1.0, 0.0),
        background_color=(0.025, 0.035, 0.055),
        draw_grid=True,
        draw_sky=False,
        draw_axis=True,
        show_info=True,
        axis_scale=0.5,
        device=device,
    )
    modes: list[Any]
    if compare_numpy:
        modes = ["warp", "numpy", "difference"]
        mode_index = 0
    else:
        modes = [(True, True), (True, False), (False, True)]
        try:
            mode_index = modes.index((bool(show_raw), bool(show_filtered)))
        except ValueError:
            mode_index = 0

    animation_started = time.perf_counter()
    paused = False
    paused_phase = 0.0
    mode_color_dirty = True

    def on_key(symbol: int, modifiers: int):
        del modifiers
        nonlocal animation_started, mode_color_dirty, mode_index, paused, paused_phase
        if symbol == pyglet.window.key.SPACE:
            mode_index = (mode_index + 1) % len(modes)
            mode_color_dirty = True
            return pyglet.event.EVENT_HANDLED
        if symbol == pyglet.window.key.P and animate_scan:
            now = time.perf_counter()
            if paused:
                animation_started = now - paused_phase / max(0.01, scan_hz)
                paused = False
            else:
                paused_phase = ((now - animation_started) * max(0.01, scan_hz)) % 1.0
                paused = True
            return pyglet.event.EVENT_HANDLED
        if symbol == pyglet.window.key.R and animate_scan:
            animation_started = time.perf_counter()
            paused_phase = 0.0
            paused = False
            return pyglet.event.EVENT_HANDLED
        return None

    renderer.register_key_press_callback(on_key)
    last_identity: tuple[Any, ...] | None = None
    processed = _error("waiting for first LaserScan", device=device)
    source_points_wp = None
    revealed_points_wp = None
    comparison_colors: dict[str, Any] = {}
    frame = 0
    while renderer.is_running():
        scan = scan_source()
        identity = (
            scan.get("source_time_ns"),
            scan.get("receive_time_ns"),
            len(scan.get("ranges") or []),
        ) if isinstance(scan, dict) else None
        if scan and identity != last_identity:
            processed = process_laser_scan(
                scan,
                device=device,
                filter_min_m=filter_min_m,
                filter_max_m=filter_max_m,
                stride=stride,
                sensor_pose=sensor_pose,
                include_raw_points=show_raw and not compare_numpy,
                compare_numpy=compare_numpy,
            )
            filtered_points = processed.get("filtered_points") or []
            if accumulate_hits and filtered_points:
                source_points_wp = wp.array(
                    np.asarray(filtered_points, dtype=np.float32),
                    dtype=wp.vec3,
                    device=device,
                )
                revealed_points_wp = wp.empty(len(filtered_points), dtype=wp.vec3, device=device)
            else:
                source_points_wp = None
                revealed_points_wp = None
            if compare_numpy and filtered_points:
                count = len(filtered_points)
                warp_colors = np.asarray(processed.get("colors") or [], dtype=np.float32)
                if warp_colors.shape != (count, 3):
                    warp_colors = np.tile((0.0, 0.75, 1.0), (count, 1)).astype(np.float32)
                benchmark = processed.get("report", {}).get("benchmark", {})
                error = benchmark.get("max_abs_error_m")
                agrees = isinstance(error, (int, float)) and float(error) <= 1.0e-5
                comparison_colors = {
                    "warp": warp_colors,
                    "numpy": np.tile((1.0, 0.35, 0.03), (count, 1)).astype(np.float32),
                    "difference": np.tile(
                        (0.15, 1.0, 0.2) if agrees else (1.0, 0.05, 0.15),
                        (count, 1),
                    ).astype(np.float32),
                }
            mode_color_dirty = True
            last_identity = identity
        filtered_points = processed.get("filtered_points") or []
        filtered_colors = processed.get("colors") or []
        raw_points = processed.get("raw_points") or []
        point_count = max(len(raw_points), len(filtered_points))
        if animate_scan and point_count:
            elapsed = max(0.0, time.perf_counter() - animation_started)
            phase = paused_phase if paused else (elapsed * max(0.01, scan_hz)) % 1.0
            reveal_count = max(1, min(point_count, int(phase * point_count) + 1))
            sweep_number = int(elapsed * max(0.01, scan_hz)) + 1
        else:
            phase = 1.0
            reveal_count = point_count
            sweep_number = 1
        if compare_numpy:
            raw_visible = False
            filtered_visible = True
            visible_raw = []
            visible_filtered = filtered_points
            active_index = min(max(0, len(filtered_points) - 1), int(phase * len(filtered_points)))
            if accumulate_hits and source_points_wp is not None and revealed_points_wp is not None:
                wp.launch(
                    _reveal_points_kernel,
                    dim=len(filtered_points),
                    inputs=[source_points_wp, reveal_count, -1000.0],
                    outputs=[revealed_points_wp],
                    device=device,
                )
                visible_filtered_render = revealed_points_wp
            else:
                visible_filtered_render = visible_filtered
        elif accumulate_hits:
            raw_visible, filtered_visible = modes[mode_index]
            visible_raw = raw_points[:min(reveal_count, len(raw_points))]
            visible_filtered = filtered_points[:min(reveal_count, len(filtered_points))]
            visible_filtered_render = visible_filtered
            active_index = max(0, len(visible_filtered) - 1)
        else:
            raw_visible, filtered_visible = modes[mode_index]
            visible_raw = raw_points
            visible_filtered = filtered_points
            visible_filtered_render = visible_filtered
            active_index = min(
                max(0, len(visible_filtered) - 1),
                int(phase * len(visible_filtered)),
            )
        visible_colors = filtered_colors[:len(visible_filtered)]
        renderer.begin_frame(frame / max(1, fps))
        renderer.render_sphere(
            "robot_origin",
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            radius=max(0.04, point_radius * 2.0),
            color=(1.0, 0.35, 0.15),
        )
        if visible_raw:
            renderer.render_points(
                "raw_lidar",
                visible_raw,
                radius=point_radius,
                colors=[(0.32, 0.35, 0.4)] * len(visible_raw),
                as_spheres=False,
                visible=raw_visible,
            )
        if visible_filtered:
            renderer.render_points(
                "warp_filtered_lidar",
                visible_filtered_render,
                radius=point_radius * 1.35,
                colors=(
                    comparison_colors.get(str(modes[mode_index]), visible_colors)
                    if compare_numpy else visible_colors
                ),
                as_spheres=False,
                visible=filtered_visible,
            )
            if compare_numpy and mode_color_dirty:
                instancer = renderer._shape_instancers.get("warp_filtered_lidar")
                palette = comparison_colors.get(str(modes[mode_index]))
                if instancer is not None and palette is not None:
                    instancer.update_colors(palette, palette)
                mode_color_dirty = False
            current_point = visible_filtered[active_index]
            renderer.render_sphere(
                "active_lidar_hit",
                pos=current_point,
                rot=(0.0, 0.0, 0.0, 1.0),
                radius=max(0.045, point_radius * 2.4),
                color=(1.0, 0.85, 0.08),
            )
            if show_rays:
                trail_end = active_index + 1
                trail_start = max(0, trail_end - max(1, ray_trail_count))
                ray_vertices: list[tuple[float, float, float] | list[float]] = []
                ray_indices: list[int] = []
                origin = (sensor_pose[0], sensor_pose[1], 0.0)
                for hit in visible_filtered[trail_start:trail_end]:
                    vertex = len(ray_vertices)
                    ray_vertices.extend((origin, hit))
                    ray_indices.extend((vertex, vertex + 1))
                renderer.render_line_list(
                    "lidar_ray_trail",
                    ray_vertices,
                    ray_indices,
                    color=(0.05, 0.48, 0.9),
                    radius=max(0.0025, point_radius * 0.16),
                )
                renderer.render_line_list(
                    "active_lidar_beam",
                    [origin, current_point],
                    [0, 1],
                    color=(1.0, 0.82, 0.04),
                    radius=max(0.006, point_radius * 0.36),
                )
        renderer.end_frame()
        if compare_numpy:
            benchmark = processed.get("report", {}).get("benchmark", {})
            selected_mode = str(modes[mode_index])
            mode_label = {
                "warp": "Warp GPU (cyan)",
                "numpy": "NumPy CPU (orange)",
                "difference": "agreement (green=match, red=error)",
            }[selected_mode]
            comparison_label = (
                f"Warp kernel {benchmark.get('warp_gpu_ms')} ms | Warp E2E {benchmark.get('warp_end_to_end_ms')} ms | "
                f"NumPy E2E {benchmark.get('numpy_cpu_ms')} ms | {benchmark.get('end_to_end_speedup')}x E2E | "
                f"error {benchmark.get('max_abs_error_m')} m | "
            )
        else:
            mode_label = "both" if raw_visible and filtered_visible else "raw" if raw_visible else "Warp-filtered"
            comparison_label = ""
        scan_label = (
            f"sweep {sweep_number} | {phase * 100:5.1f}% | {'PAUSED' if paused else f'{scan_hz:g} Hz'}"
            if animate_scan else "complete scan"
        )
        renderer.window.set_caption(
            f"{title} | {scan_label} | processed {processed.get('report', {}).get('input_count', 0):,} rays | "
            f"displayed {point_count:,} hits | {comparison_label}{mode_label} | "
            "Space: compare  P: pause  R: restart"
        )
        frame += 1
    renderer.close()


@node(
    name="WarpLiDARViewer",
    component="spatial-processing",
    category=_CATEGORY,
    description=(
        "Start or stop an interactive Warp OpenGL window for one normalized "
        "LaserScan, showing raw gray and Warp-filtered cyan points together."
    ),
    inputs={
        "action": Enum(["stop", "start"], default="stop"),
        "laser_scan": Dict,
        "viewer_id": Text(default="warp_lidar_static"),
        "device": Enum(["cuda:0", "cpu"], default="cuda:0"),
        "filter_min_m": Float(default=0.1),
        "filter_max_m": Float(default=12.0),
        "downsample_stride": Int(default=2),
        "sensor_x_m": Float(default=0.0),
        "sensor_y_m": Float(default=0.0),
        "sensor_yaw_rad": Float(default=0.0),
        "show_raw": Bool(default=True),
        "show_filtered": Bool(default=True),
        "point_radius_m": Float(default=0.025),
        "fps": Int(default=30),
        "animate_scan": Bool(default=True),
        "scan_hz": Float(default=0.25),
        "show_rays": Bool(default=True),
        "ray_trail_count": Int(default=96),
        "accumulate_hits": Bool(default=True),
        "compare_numpy": Bool(default=False),
    },
    outputs={
        "running": Bool,
        "viewer": Dict,
        "report": Text,
    },
)
def warp_lidar_viewer(ctx: dict) -> dict:
    viewer_id = str(ctx.get("viewer_id") or "warp_lidar_static").strip()
    if str(ctx.get("action") or "stop") == "stop":
        stopped = viewer_rt.stop_viewer(viewer_id)
        return {
            "running": False,
            "viewer": {"viewer_id": viewer_id, "state": "stopped"},
            "report": f"Warp LiDAR viewer stopped {int(stopped.get('stopped') or 0)} process(es)",
        }
    options = {
        "device": str(ctx.get("device") or "cuda:0"),
        "filter_min_m": max(0.0, float(ctx.get("filter_min_m") or 0.1)),
        "filter_max_m": max(0.0, float(ctx.get("filter_max_m") or 12.0)),
        "stride": max(1, int(ctx.get("downsample_stride") or 1)),
        "sensor_x_m": float(ctx.get("sensor_x_m") or 0.0),
        "sensor_y_m": float(ctx.get("sensor_y_m") or 0.0),
        "sensor_yaw_rad": float(ctx.get("sensor_yaw_rad") or 0.0),
        "show_raw": bool(ctx.get("show_raw", True)),
        "show_filtered": bool(ctx.get("show_filtered", True)),
        "point_radius_m": max(0.001, float(ctx.get("point_radius_m") or 0.025)),
        "fps": max(1, min(120, int(ctx.get("fps") or 30))),
        "animate_scan": bool(ctx.get("animate_scan", True)),
        "scan_hz": max(0.01, min(20.0, float(ctx.get("scan_hz") or 0.25))),
        "show_rays": bool(ctx.get("show_rays", True)),
        "ray_trail_count": max(1, min(4096, int(ctx.get("ray_trail_count") or 96))),
        "accumulate_hits": bool(ctx.get("accumulate_hits", True)),
        "compare_numpy": bool(ctx.get("compare_numpy", False)),
    }
    started = viewer_rt.start_viewer(
        viewer_id=viewer_id,
        scan=ctx.get("laser_scan") if isinstance(ctx.get("laser_scan"), dict) else {},
        options=options,
    )
    if not started.get("ok"):
        reason = str(started.get("error") or "could not start Warp LiDAR viewer")
        return {
            "running": False,
            "viewer": {"viewer_id": viewer_id, "state": "failed", "error": reason},
            "report": reason,
        }
    return {
        "running": True,
        "viewer": {
            "kind": "blacknode.lidar-viewer",
            "schema_version": 1,
            "viewer_id": viewer_id,
            "state": "running",
            "device": options["device"],
            "controls": {
                "space": (
                    "cycle Warp GPU / NumPy CPU / agreement"
                    if options["compare_numpy"]
                    else "cycle raw / filtered / both"
                ),
                "p": "pause or resume scan sweep",
                "r": "restart scan sweep",
                "escape": "close",
            },
        },
        "report": (
            "Warp LiDAR viewer running with animated sweep; "
            + (
                "Space changes comparison mode, P pauses, and R restarts"
                if options["compare_numpy"]
                else "Space changes layers, P pauses, and R restarts"
            )
        ),
    }
