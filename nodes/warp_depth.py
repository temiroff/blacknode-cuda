"""Live metric-depth loading and pinhole projection backed by NVIDIA Warp."""
from __future__ import annotations

import json
import io
import math
import struct
import time
import urllib.error
import urllib.request
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - package diagnostics own this path
    np = None

try:
    import warp as wp
except Exception:  # pragma: no cover - discovery must work without Warp
    wp = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - package diagnostics own this path
    Image = None


_MAGIC = b"BNDEPTH1"
_MAX_FRAME_BYTES = 256 * 1024 * 1024


if wp is not None:

    @wp.kernel
    def _cleanup_depth_kernel(
        depth_m: wp.array(dtype=wp.float32),
        previous_depth_m: wp.array(dtype=wp.float32),
        width: wp.int32,
        height: wp.int32,
        minimum_m: wp.float32,
        maximum_m: wp.float32,
        spatial_filter: wp.int32,
        spatial_max_delta_m: wp.float32,
        hole_fill: wp.int32,
        minimum_neighbors: wp.int32,
        outlier_rejection: wp.int32,
        outlier_max_delta_m: wp.float32,
        temporal_smoothing: wp.float32,
        temporal_max_delta_m: wp.float32,
        history_valid: wp.int32,
        cleaned_depth_m: wp.array(dtype=wp.float32),
        stats: wp.array(dtype=wp.int32),
    ):
        """Edge-aware cleanup with motion-gated temporal stabilization.

        ``stats`` contains valid, hole-filled, outlier-replaced, and temporal
        blend counts. The small counter buffer avoids copying a full-resolution
        classification image back to the host.
        """
        index = wp.tid()
        u = index % width
        v = index // width
        center = depth_m[index]
        center_valid = wp.isfinite(center) and center >= minimum_m and center <= maximum_m

        neighbor_sum = wp.float32(0.0)
        neighbor_min = maximum_m
        neighbor_max = minimum_m
        neighbor_count = wp.int32(0)
        if u > 0:
            value = depth_m[index - 1]
            if wp.isfinite(value) and value >= minimum_m and value <= maximum_m:
                neighbor_sum += value
                neighbor_min = wp.min(neighbor_min, value)
                neighbor_max = wp.max(neighbor_max, value)
                neighbor_count += 1
        if u + 1 < width:
            value = depth_m[index + 1]
            if wp.isfinite(value) and value >= minimum_m and value <= maximum_m:
                neighbor_sum += value
                neighbor_min = wp.min(neighbor_min, value)
                neighbor_max = wp.max(neighbor_max, value)
                neighbor_count += 1
        if v > 0:
            value = depth_m[index - width]
            if wp.isfinite(value) and value >= minimum_m and value <= maximum_m:
                neighbor_sum += value
                neighbor_min = wp.min(neighbor_min, value)
                neighbor_max = wp.max(neighbor_max, value)
                neighbor_count += 1
        if v + 1 < height:
            value = depth_m[index + width]
            if wp.isfinite(value) and value >= minimum_m and value <= maximum_m:
                neighbor_sum += value
                neighbor_min = wp.min(neighbor_min, value)
                neighbor_max = wp.max(neighbor_max, value)
                neighbor_count += 1

        consensus = (
            neighbor_count >= minimum_neighbors
            and neighbor_max - neighbor_min <= spatial_max_delta_m
        )
        cleaned = center
        if not center_valid:
            if hole_fill != 0 and consensus:
                cleaned = neighbor_sum / wp.float32(neighbor_count)
                center_valid = True
                wp.atomic_add(stats, 1, 1)
            else:
                cleaned_depth_m[index] = wp.float32(0.0)
                return
        elif outlier_rejection != 0 and consensus:
            neighbor_mean = neighbor_sum / wp.float32(neighbor_count)
            if wp.abs(center - neighbor_mean) > outlier_max_delta_m:
                cleaned = neighbor_mean
                wp.atomic_add(stats, 2, 1)
        if spatial_filter != 0 and consensus:
            neighbor_mean = neighbor_sum / wp.float32(neighbor_count)
            if wp.abs(cleaned - neighbor_mean) <= spatial_max_delta_m:
                cleaned = (cleaned + neighbor_sum) / wp.float32(neighbor_count + 1)

        previous = previous_depth_m[index]
        if (
            history_valid != 0
            and temporal_smoothing > 0.0
            and wp.isfinite(previous)
            and previous >= minimum_m
            and previous <= maximum_m
            and wp.abs(previous - cleaned) <= temporal_max_delta_m
        ):
            cleaned = cleaned * (1.0 - temporal_smoothing) + previous * temporal_smoothing
            wp.atomic_add(stats, 3, 1)

        cleaned_depth_m[index] = cleaned
        wp.atomic_add(stats, 0, 1)

    @wp.kernel
    def _project_depth_kernel(
        depth_m: wp.array(dtype=wp.float32),
        width: wp.int32,
        height: wp.int32,
        fx: wp.float32,
        fy: wp.float32,
        cx: wp.float32,
        cy: wp.float32,
        minimum_m: wp.float32,
        maximum_m: wp.float32,
        stride: wp.int32,
        sample_width: wp.int32,
        sensor_x: wp.float32,
        sensor_y: wp.float32,
        sensor_z: wp.float32,
        sensor_roll: wp.float32,
        sensor_pitch: wp.float32,
        sensor_yaw: wp.float32,
        points: wp.array(dtype=wp.vec3),
        normals: wp.array(dtype=wp.vec3),
        colors: wp.array(dtype=wp.vec3),
        confidence: wp.array(dtype=wp.float32),
        valid: wp.array(dtype=wp.int32),
        source_indices: wp.array(dtype=wp.int32),
    ):
        output_index = wp.tid()
        u = (output_index % sample_width) * stride
        v = (output_index // sample_width) * stride
        if u >= width or v >= height:
            return
        index = v * width + u
        depth = depth_m[index]
        keep = (
            depth >= minimum_m
            and depth <= maximum_m
            and wp.isfinite(depth)
        )
        if not keep:
            return

        # Blacknode viewer coordinates are forward, left, up. Camera depth is
        # forward Z with image X right and image Y down.
        forward = depth
        left = -(wp.float32(u) - cx) * depth / fx
        up = -(wp.float32(v) - cy) * depth / fy
        point = wp.vec3(forward, left, up)
        normal = wp.vec3(-1.0, 0.0, 0.0)
        normal_confidence = wp.float32(0.35)
        if u + 1 < width and v + 1 < height:
            right_depth = depth_m[index + 1]
            down_depth = depth_m[index + width]
            if (
                wp.isfinite(right_depth)
                and wp.isfinite(down_depth)
                and right_depth >= minimum_m
                and right_depth <= maximum_m
                and down_depth >= minimum_m
                and down_depth <= maximum_m
            ):
                right_point = wp.vec3(
                    right_depth,
                    -(wp.float32(u + 1) - cx) * right_depth / fx,
                    -(wp.float32(v) - cy) * right_depth / fy,
                )
                down_point = wp.vec3(
                    down_depth,
                    -(wp.float32(u) - cx) * down_depth / fx,
                    -(wp.float32(v + 1) - cy) * down_depth / fy,
                )
                candidate = wp.cross(right_point - point, down_point - point)
                length = wp.length(candidate)
                if length > 1.0e-8:
                    normal = candidate / length
                    normal_confidence = wp.clamp(wp.abs(normal[0]), 0.0, 1.0)

        distance_fraction = wp.clamp(
            (depth - minimum_m) / wp.max(maximum_m - minimum_m, 1.0e-6),
            0.0,
            1.0,
        )
        cr = wp.cos(sensor_roll)
        sr = wp.sin(sensor_roll)
        cp = wp.cos(sensor_pitch)
        sp = wp.sin(sensor_pitch)
        cyaw = wp.cos(sensor_yaw)
        syaw = wp.sin(sensor_yaw)
        r00 = cyaw * cp
        r01 = cyaw * sp * sr - syaw * cr
        r02 = cyaw * sp * cr + syaw * sr
        r10 = syaw * cp
        r11 = syaw * sp * sr + cyaw * cr
        r12 = syaw * sp * cr - cyaw * sr
        r20 = -sp
        r21 = cp * sr
        r22 = cp * cr
        points[output_index] = wp.vec3(
            sensor_x + r00 * point[0] + r01 * point[1] + r02 * point[2],
            sensor_y + r10 * point[0] + r11 * point[1] + r12 * point[2],
            sensor_z + r20 * point[0] + r21 * point[1] + r22 * point[2],
        )
        normals[output_index] = wp.vec3(
            r00 * normal[0] + r01 * normal[1] + r02 * normal[2],
            r10 * normal[0] + r11 * normal[1] + r12 * normal[2],
            r20 * normal[0] + r21 * normal[1] + r22 * normal[2],
        )
        confidence[output_index] = normal_confidence
        colors[output_index] = wp.vec3(
            0.05 + 0.15 * distance_fraction,
            0.85 - 0.35 * distance_fraction,
            1.0,
        ) * (0.55 + 0.45 * normal_confidence)
        valid[output_index] = 1
        source_indices[output_index] = index


class WarpDepthProcessingState:
    """Reusable device buffers and temporal history for one managed camera."""

    def __init__(self) -> None:
        self.device = ""
        self.width = 0
        self.height = 0
        self.sample_count = 0
        self.depth_input = None
        self.previous_depth = None
        self.filtered_depth = None
        self.points = None
        self.normals = None
        self.colors = None
        self.confidence = None
        self.valid = None
        self.source_indices = None
        self.stats = None
        self.history_valid = False
        self.filter_signature: tuple[Any, ...] | None = None
        self.last_source_time_ns = 0
        self.last_stats = [0, 0, 0, 0]
        self.allocations = 0

    def prepare(self, selected_device: Any, width: int, height: int, sample_count: int) -> bool:
        key = (str(selected_device), int(width), int(height), int(sample_count))
        current = (self.device, self.width, self.height, self.sample_count)
        if key == current and self.depth_input is not None:
            return True
        pixel_count = int(width * height)
        self.device, self.width, self.height, self.sample_count = key
        self.depth_input = wp.zeros(pixel_count, dtype=wp.float32, device=selected_device)
        self.previous_depth = wp.zeros(pixel_count, dtype=wp.float32, device=selected_device)
        self.filtered_depth = wp.zeros(pixel_count, dtype=wp.float32, device=selected_device)
        self.points = wp.zeros(sample_count, dtype=wp.vec3, device=selected_device)
        self.normals = wp.zeros(sample_count, dtype=wp.vec3, device=selected_device)
        self.colors = wp.zeros(sample_count, dtype=wp.vec3, device=selected_device)
        self.confidence = wp.zeros(sample_count, dtype=wp.float32, device=selected_device)
        self.valid = wp.zeros(sample_count, dtype=wp.int32, device=selected_device)
        self.source_indices = wp.zeros(sample_count, dtype=wp.int32, device=selected_device)
        self.stats = wp.zeros(4, dtype=wp.int32, device=selected_device)
        self.history_valid = False
        self.filter_signature = None
        self.last_source_time_ns = 0
        self.last_stats = [0, 0, 0, 0]
        self.allocations += 1
        return False


def _error(
    message: str,
    *,
    device: str = "",
    state: str = "unavailable",
    worker_alive: bool = False,
) -> dict[str, Any]:
    return {
        "ok": False,
        "filtered_points": [],
        "colors": [],
        "normals": [],
        "confidence": [],
        "point_cloud": {},
        "device": device,
        "kernel_ms": 0.0,
        "report": {
            "state": state,
            "device": device,
            "worker_alive": worker_alive,
            "source_fresh": False,
            "error": message,
        },
    }


def _decode_binary(payload: bytes) -> tuple[Any, dict[str, Any]]:
    if np is None:
        raise RuntimeError("NumPy is required for metric-depth decoding")
    if len(payload) < 12 or not payload.startswith(_MAGIC):
        raise ValueError("depth endpoint did not return a Blacknode metric-depth frame")
    header_size = struct.unpack("<I", payload[8:12])[0]
    if header_size <= 0 or header_size > 64 * 1024 or 12 + header_size > len(payload):
        raise ValueError("metric-depth frame header is invalid")
    header = json.loads(payload[12:12 + header_size].decode("utf-8"))
    width = int(header.get("width") or 0)
    height = int(header.get("height") or 0)
    step = int(header.get("step") or 0)
    encoding = str(header.get("encoding") or "").strip().lower()
    bytes_per_pixel = 4 if encoding == "32fc1" else 2 if encoding in {"16uc1", "mono16"} else 0
    if width <= 0 or height <= 0 or bytes_per_pixel == 0 or step < width * bytes_per_pixel:
        raise ValueError("metric-depth frame dimensions or encoding are unsupported")
    raw = payload[12 + header_size:]
    required = height * step
    if required > _MAX_FRAME_BYTES or len(raw) < required:
        raise ValueError("metric-depth frame payload is truncated or too large")
    dtype = np.dtype(
        (">f4" if header.get("is_bigendian") else "<f4")
        if encoding == "32fc1"
        else (">u2" if header.get("is_bigendian") else "<u2")
    )
    rows = np.frombuffer(raw[:required], dtype=np.uint8).reshape((height, step))
    packed = rows[:, : width * bytes_per_pixel].copy()
    return packed.view(dtype).reshape((height, width)).astype(np.float32), header


def load_metric_depth(depth_stream: dict[str, Any]) -> tuple[Any, dict[str, Any], float]:
    """Load one provider-neutral depth frame through inline replay or compact HTTP."""
    if np is None:
        raise RuntimeError("NumPy is required for metric-depth loading")
    source = depth_stream.get("frame_source") if isinstance(depth_stream.get("frame_source"), dict) else {}
    if source.get("kind") != "blacknode.depth-frame-source":
        raise ValueError("depth_stream.frame_source must be a blacknode.depth-frame-source")
    started = time.perf_counter()
    transport = str(source.get("transport") or "").strip().lower()
    if transport == "inline":
        width = int(source.get("width") or 0)
        height = int(source.get("height") or 0)
        values = source.get("depth_m") if isinstance(source.get("depth_m"), list) else []
        if width <= 0 or height <= 0 or len(values) != width * height:
            raise ValueError("inline depth frame dimensions do not match depth_m")
        depth = np.asarray(values, dtype=np.float32).reshape((height, width))
        header = {
            "width": width,
            "height": height,
            "encoding": "32FC1",
            "frame_id": str(source.get("frame") or depth_stream.get("frame") or "camera_depth"),
            "stamp_sec": 0,
            "stamp_nanosec": int(source.get("source_time_ns") or 0),
            "received_at_ns": int(source.get("source_time_ns") or time.time_ns()),
        }
        scale = 1.0
    elif transport == "http-binary":
        url = str(source.get("url") or "").strip()
        if not url:
            raise ValueError("binary depth frame URL is empty")
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/vnd.blacknode.metric-depth-frame"},
        )
        try:
            with urllib.request.urlopen(request, timeout=1.5) as response:
                payload = response.read(_MAX_FRAME_BYTES + 64 * 1024 + 13)
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise RuntimeError(f"metric depth fetch failed ({type(exc).__name__}: {exc})") from exc
        if len(payload) > _MAX_FRAME_BYTES + 64 * 1024 + 12:
            raise ValueError("metric-depth response exceeded the 256 MB frame limit")
        depth, header = _decode_binary(payload)
        scale = float(source.get("depth_scale") or depth_stream.get("depth_scale") or 1.0)
    else:
        raise ValueError(f"unsupported depth-frame transport {transport!r}")
    encoding = str(header.get("encoding") or source.get("encoding") or depth_stream.get("encoding") or "").lower()
    if encoding != "32fc1":
        depth = depth * np.float32(max(0.0, scale))
    return depth, header, (time.perf_counter() - started) * 1000.0


def load_color_frame(
    frame_stream: dict[str, Any],
    *,
    width: int,
    height: int,
) -> tuple[Any, float]:
    """Load the latest aligned RGB image from a compact frame-stream contract."""
    if np is None:
        raise RuntimeError("NumPy is required for RGB frame loading")
    started = time.perf_counter()
    if isinstance(frame_stream.get("pixels_rgb"), list):
        source_width = int(frame_stream.get("width") or 0)
        source_height = int(frame_stream.get("height") or 0)
        pixels = np.asarray(frame_stream["pixels_rgb"], dtype=np.uint8)
        if source_width <= 0 or source_height <= 0 or pixels.size != source_width * source_height * 3:
            raise ValueError("inline RGB dimensions do not match pixels_rgb")
        image = pixels.reshape((source_height, source_width, 3))
        if (source_width, source_height) != (width, height):
            if Image is None:
                raise RuntimeError("Pillow is required to align RGB and depth resolutions")
            image = np.asarray(Image.fromarray(image, "RGB").resize((width, height), Image.Resampling.BILINEAR))
    else:
        if Image is None:
            raise RuntimeError("Pillow is required for RGB snapshot decoding")
        url = str(frame_stream.get("snapshot_url") or "").strip()
        if not url:
            raise ValueError("RGB frame stream has no snapshot_url")
        request = urllib.request.Request(url, headers={"Accept": "image/jpeg,image/png"})
        try:
            with urllib.request.urlopen(request, timeout=1.5) as response:
                payload = response.read(32 * 1024 * 1024 + 1)
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise RuntimeError(f"RGB frame fetch failed ({type(exc).__name__}: {exc})") from exc
        if len(payload) > 32 * 1024 * 1024:
            raise ValueError("RGB snapshot exceeded the 32 MB frame limit")
        with Image.open(io.BytesIO(payload)) as decoded:
            image = np.asarray(decoded.convert("RGB").resize((width, height), Image.Resampling.BILINEAR))
    return np.ascontiguousarray(image, dtype=np.uint8), (time.perf_counter() - started) * 1000.0


def _numpy_project(
    depth: Any,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    minimum_m: float,
    maximum_m: float,
    stride: int,
) -> Any:
    height, width = depth.shape
    rows, columns = np.indices((height, width), dtype=np.float32)
    mask = (
        np.isfinite(depth)
        & (depth >= np.float32(minimum_m))
        & (depth <= np.float32(maximum_m))
        & ((rows.astype(np.int32) % stride) == 0)
        & ((columns.astype(np.int32) % stride) == 0)
    )
    values = depth[mask]
    return np.column_stack((
        values,
        -(columns[mask] - np.float32(cx)) * values / np.float32(fx),
        -(rows[mask] - np.float32(cy)) * values / np.float32(fy),
    )).astype(np.float32)


def process_depth_stream(
    depth_stream: dict[str, Any],
    stage: dict[str, Any],
    *,
    device: str = "cuda:0",
    color_stream: dict[str, Any] | None = None,
    color_mode: str = "rgb",
    processing_state: WarpDepthProcessingState | None = None,
) -> dict[str, Any]:
    if wp is None:
        return _error("NVIDIA Warp is not installed; install warp-lang>=1.15", device=device)
    if np is None:
        return _error("NumPy is not installed; install numpy>=1.24", device=device)
    if depth_stream.get("kind") != "blacknode.depth-stream":
        return _error("source must be a blacknode.depth-stream", device=device)
    if stage.get("kind") != "blacknode.warp-depth-projector":
        return _error("connect WarpDepthProjector.stage to Viewer.depth_projection", device=device)
    try:
        selected_device = wp.get_device(device)
        depth, header, fetch_ms = load_metric_depth(depth_stream)
        source = depth_stream.get("frame_source") if isinstance(depth_stream.get("frame_source"), dict) else {}
        received_at_ns = int(header.get("received_at_ns") or 0)
        age_seconds = (
            max(0.0, (time.time_ns() - received_at_ns) / 1_000_000_000.0)
            if received_at_ns > 0
            else None
        )
        stale_after_seconds = max(0.1, float(stage.get("stale_after_seconds") or 2.0))
        if (
            source.get("transport") == "http-binary"
            and age_seconds is not None
            and age_seconds > stale_after_seconds
        ):
            return _error(
                f"metric depth frame is stale ({age_seconds:.3f}s old)",
                device=device,
                state="stale",
                worker_alive=True,
            )
        source_time_ns = (
            int(header.get("stamp_sec") or 0) * 1_000_000_000
            + int(header.get("stamp_nanosec") or 0)
        ) or int(header.get("received_at_ns") or 0)
        calibration = depth_stream.get("calibration") if isinstance(depth_stream.get("calibration"), dict) else {}
        height, width = depth.shape
        fx = float(calibration.get("fx") or 0.0)
        fy = float(calibration.get("fy") or 0.0)
        calibration_width = int(calibration.get("width") or 0)
        calibration_height = int(calibration.get("height") or 0)
        cx = float(calibration.get("cx") if calibration_width > 0 and calibration.get("cx") is not None else (width - 1) * 0.5)
        cy = float(calibration.get("cy") if calibration_height > 0 and calibration.get("cy") is not None else (height - 1) * 0.5)
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("depth camera calibration requires positive fx and fy")
        minimum_m = max(0.001, float(stage.get("minimum_depth_m") or 0.1))
        maximum_m = max(minimum_m, float(stage.get("maximum_depth_m") or 8.0))
        requested_stride = max(1, int(stage.get("stride") or 2))
        maximum_points = max(64, min(250_000, int(stage.get("maximum_points") or 50_000)))
        stride = requested_stride
        while math.ceil(width / stride) * math.ceil(height / stride) > maximum_points:
            stride += 1
        sample_width = math.ceil(width / stride)
        sample_height = math.ceil(height / stride)
        sample_count = int(sample_width * sample_height)
        sensor_x = float(stage.get("sensor_x_m") or 0.0)
        sensor_y = float(stage.get("sensor_y_m") or 0.0)
        sensor_z = float(stage.get("sensor_z_m") or 0.0)
        sensor_roll = float(stage.get("sensor_roll_rad") or 0.0)
        sensor_pitch = float(stage.get("sensor_pitch_rad") or 0.0)
        sensor_yaw = float(stage.get("sensor_yaw_rad") or 0.0)
        spatial_filter = bool(stage.get("spatial_filter", True))
        spatial_max_delta_m = max(0.001, min(1.0, float(stage.get("spatial_max_delta_m", 0.04))))
        hole_fill = bool(stage.get("hole_fill", True))
        minimum_neighbors = max(2, min(4, int(stage.get("minimum_neighbors", 3))))
        outlier_rejection = bool(stage.get("outlier_rejection", True))
        outlier_max_delta_m = max(0.001, min(2.0, float(stage.get("outlier_max_delta_m", 0.12))))
        temporal_smoothing = max(0.0, min(0.95, float(stage.get("temporal_smoothing", 0.35))))
        temporal_max_delta_m = max(0.001, min(2.0, float(stage.get("temporal_max_delta_m", 0.08))))
        cleanup_enabled = spatial_filter or hole_fill or outlier_rejection or temporal_smoothing > 0.0
        filter_signature = (
            minimum_m, maximum_m, spatial_filter, spatial_max_delta_m, hole_fill,
            minimum_neighbors, outlier_rejection, outlier_max_delta_m,
            temporal_smoothing, temporal_max_delta_m,
        )
        state = processing_state if isinstance(processing_state, WarpDepthProcessingState) else WarpDepthProcessingState()
        buffers_reused = state.prepare(selected_device, width, height, sample_count)
        if state.filter_signature != filter_signature:
            state.history_valid = False
            state.filter_signature = filter_signature
            state.last_source_time_ns = 0
        values = np.ascontiguousarray(depth.reshape(-1), dtype=np.float32)
        started = time.perf_counter()
        state.depth_input.assign(values)
        duplicate_frame = bool(
            cleanup_enabled
            and state.history_valid
            and source_time_ns > 0
            and source_time_ns == state.last_source_time_ns
        )
        if cleanup_enabled and not duplicate_frame:
            state.stats.fill_(0)
            wp.launch(
                _cleanup_depth_kernel,
                dim=int(values.size),
                inputs=[
                    state.depth_input, state.previous_depth, width, height,
                    minimum_m, maximum_m, int(spatial_filter), spatial_max_delta_m,
                    int(hole_fill), minimum_neighbors, int(outlier_rejection),
                    outlier_max_delta_m, temporal_smoothing, temporal_max_delta_m,
                    int(state.history_valid),
                ],
                outputs=[state.filtered_depth, state.stats],
                device=selected_device,
            )
            cleaned_depth = state.filtered_depth
        elif cleanup_enabled:
            cleaned_depth = state.previous_depth
        else:
            cleaned_depth = state.depth_input
        state.valid.fill_(0)
        wp.launch(
            _project_depth_kernel,
            dim=sample_count,
            inputs=[
                cleaned_depth, width, height, fx, fy, cx, cy, minimum_m, maximum_m,
                stride, sample_width,
                sensor_x, sensor_y, sensor_z, sensor_roll, sensor_pitch, sensor_yaw,
            ],
            outputs=[
                state.points, state.normals, state.colors, state.confidence,
                state.valid, state.source_indices,
            ],
            device=selected_device,
        )
        wp.synchronize_device(selected_device)
        if cleanup_enabled and not duplicate_frame:
            state.last_stats = [int(value) for value in state.stats.numpy().tolist()]
            state.previous_depth, state.filtered_depth = state.filtered_depth, state.previous_depth
            state.history_valid = True
            state.last_source_time_ns = source_time_ns
        points = state.points.numpy()
        normals = state.normals.numpy()
        colors = state.colors.numpy()
        confidence = state.confidence.numpy()
        source_indices = state.source_indices.numpy()
        mask = state.valid.numpy().astype(bool, copy=False)
        pipeline_ms = (time.perf_counter() - started) * 1000.0
        selected = np.flatnonzero(mask)
        projected = points[selected]
        projected_normals = normals[selected]
        projected_colors = colors[selected]
        requested_color_mode = str(color_mode or "depth").strip().lower()
        if requested_color_mode not in {"depth", "rgb", "ir"}:
            requested_color_mode = "depth"
        color_fetch_ms = 0.0
        color_registered = False
        color_error = "" if requested_color_mode == "depth" else f"{requested_color_mode.upper()} source unavailable"
        if requested_color_mode != "depth" and isinstance(color_stream, dict) and color_stream:
            try:
                color_image, color_fetch_ms = load_color_frame(
                    color_stream,
                    width=width,
                    height=height,
                )
                color_indices = source_indices[selected].astype(np.int64, copy=False)
                projected_colors = color_image.reshape((-1, 3))[color_indices].astype(np.float32) / np.float32(255.0)
                color_registered = True
                color_error = ""
            except Exception as exc:
                color_error = f"{type(exc).__name__}: {exc}"
        projected_confidence = confidence[selected]
        cpu_ms = 0.0
        maximum_error = 0.0
        if stage.get("compare_cpu"):
            cpu_started = time.perf_counter()
            reference_depth = (
                state.previous_depth.numpy().reshape((height, width))
                if cleanup_enabled
                else depth
            )
            cpu_points = _numpy_project(
                reference_depth,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                minimum_m=minimum_m,
                maximum_m=maximum_m,
                stride=stride,
            )
            cpu_ms = (time.perf_counter() - cpu_started) * 1000.0
            cr, sr = math.cos(sensor_roll), math.sin(sensor_roll)
            cp, sp = math.cos(sensor_pitch), math.sin(sensor_pitch)
            cyaw, syaw = math.cos(sensor_yaw), math.sin(sensor_yaw)
            rotation = np.asarray([
                [cyaw * cp, cyaw * sp * sr - syaw * cr, cyaw * sp * cr + syaw * sr],
                [syaw * cp, syaw * sp * sr + cyaw * cr, syaw * sp * cr - cyaw * sr],
                [-sp, cp * sr, cp * cr],
            ], dtype=np.float32)
            cpu_points = cpu_points @ rotation.T + np.asarray([sensor_x, sensor_y, sensor_z], dtype=np.float32)
            warp_points = points[mask]
            maximum_error = float(np.max(np.abs(cpu_points - warp_points))) if len(cpu_points) else 0.0
    except Exception as exc:
        return _error(f"Warp depth projection failed ({type(exc).__name__}: {exc})", device=device)

    cleanup_stats = state.last_stats if cleanup_enabled else [int(np.count_nonzero(mask)), 0, 0, 0]
    report = {
        "state": "ready",
        "backend": "warp",
        "device": str(selected_device),
        "width": width,
        "height": height,
        "input_pixels": int(width * height),
        "candidate_points": sample_count,
        "valid_points": int(np.count_nonzero(mask)),
        "display_points": int(len(projected)),
        "stride": stride,
        "requested_stride": requested_stride,
        "buffers_reused": buffers_reused,
        "buffer_allocations": state.allocations,
        "duplicate_frame": duplicate_frame,
        "minimum_depth_m": minimum_m,
        "maximum_depth_m": maximum_m,
        "fetch_ms": float(fetch_ms),
        "age_seconds": age_seconds,
        "stale_after_seconds": stale_after_seconds,
        "worker_alive": True,
        "source_fresh": True,
        "pipeline_ms": float(pipeline_ms),
        "cpu_ms": float(cpu_ms),
        "speedup": float(cpu_ms / pipeline_ms) if cpu_ms > 0.0 and pipeline_ms > 0.0 else 0.0,
        "max_error_m": float(maximum_error),
        "mean_confidence": float(np.mean(projected_confidence)) if len(projected_confidence) else 0.0,
        "cleanup": {
            "enabled": cleanup_enabled,
            "spatial_filter": spatial_filter,
            "spatial_max_delta_m": spatial_max_delta_m,
            "hole_fill": hole_fill,
            "minimum_neighbors": minimum_neighbors,
            "outlier_rejection": outlier_rejection,
            "outlier_max_delta_m": outlier_max_delta_m,
            "temporal_smoothing": temporal_smoothing,
            "temporal_max_delta_m": temporal_max_delta_m,
            "valid_pixels": int(cleanup_stats[0]),
            "holes_filled": int(cleanup_stats[1]),
            "outliers_replaced": int(cleanup_stats[2]),
            "temporally_blended": int(cleanup_stats[3]),
        },
        "target_frame": str(stage.get("target_frame") or "base_link"),
        "sensor_extrinsics": {
            "x_m": sensor_x,
            "y_m": sensor_y,
            "z_m": sensor_z,
            "roll_rad": sensor_roll,
            "pitch_rad": sensor_pitch,
            "yaw_rad": sensor_yaw,
        },
        "encoding": str(header.get("encoding") or depth_stream.get("encoding") or ""),
        "source_time_ns": source_time_ns,
        "color_registered": color_registered,
        "color_mode": requested_color_mode,
        "color_applied": requested_color_mode if color_registered else "depth",
        "color_fetch_ms": float(color_fetch_ms),
        "color_source": str((color_stream or {}).get("snapshot_url") or ""),
        "color_error": color_error,
    }
    point_cloud = {
        "kind": "blacknode.point-cloud-frame",
        "schema_version": 1,
        "frame": str(stage.get("target_frame") or "base_link"),
        "source_frame": str(header.get("frame_id") or depth_stream.get("frame") or "camera_depth"),
        "source_time_ns": source_time_ns,
        "receive_time_ns": time.time_ns(),
        "point_count": int(len(projected)),
        "points_xyz": projected.astype(float).tolist(),
        "colors_rgb": projected_colors.astype(float).tolist(),
        "normals_xyz": projected_normals.astype(float).tolist(),
        "confidence": projected_confidence.astype(float).tolist(),
        "processing": report,
    }
    return {
        "ok": True,
        "filtered_points": point_cloud["points_xyz"],
        "colors": point_cloud["colors_rgb"],
        "normals": point_cloud["normals_xyz"],
        "confidence": point_cloud["confidence"],
        "point_cloud": point_cloud,
        "device": str(selected_device),
        "kernel_ms": float(pipeline_ms),
        "report": report,
    }
