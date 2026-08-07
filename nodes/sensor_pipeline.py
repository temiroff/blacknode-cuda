"""Declarative Warp sensor-compute stages executed by managed viewers."""
from __future__ import annotations

import math

from blacknode.node import Bool, Dict, Float, Int, Text, node


_CATEGORY = "NVIDIA CUDA"


@node(
    name="WarpParticleLocalization",
    component="spatial-processing",
    category=_CATEGORY,
    description=(
        "Configure dense GPU pose-hypothesis scoring for a managed SLAM "
        "session. Connect stage to SLAM.particle_localization; SLAM keeps the "
        "scan, map, hypotheses, and scores in one live compute owner."
    ),
    inputs={
        "enabled": Bool(default=True),
        "particles": Int(default=16_384),
        "position_spread_m": Float(default=0.8),
        "yaw_spread_deg": Float(default=18.0),
        "display_particles": Int(default=1_500),
        "random_seed": Int(default=7),
        "compare_cpu": Bool(default=False),
    },
    outputs={
        "stage": Dict,
        "enabled": Bool,
        "workload": Text,
        "report": Text,
    },
    primary_inputs=["enabled", "particles", "position_spread_m", "yaw_spread_deg"],
    primary_outputs=["stage", "report"],
)
def warp_particle_localization(ctx: dict) -> dict:
    enabled = bool(ctx.get("enabled", True))
    particles = max(64, min(65_536, int(ctx.get("particles") or 16_384)))
    compare_cpu = bool(ctx.get("compare_cpu", False))
    comparison_limited = bool(compare_cpu and particles > 4_096)
    if comparison_limited:
        particles = 4_096
    position_spread_m = max(
        0.02,
        min(10.0, float(ctx.get("position_spread_m") or 0.8)),
    )
    yaw_spread_rad = math.radians(
        max(0.5, min(180.0, float(ctx.get("yaw_spread_deg") or 18.0)))
    )
    display_particles = max(
        32,
        min(particles, 4_000, int(ctx.get("display_particles") or 1_500)),
    )
    stage = {
        "kind": "blacknode.warp-particle-localization",
        "schema_version": 1,
        "enabled": enabled,
        "particle_count": particles,
        "position_spread_m": position_spread_m,
        "yaw_spread_rad": yaw_spread_rad,
        "display_particles": display_particles,
        "random_seed": int(ctx.get("random_seed") or 7),
        "compare_cpu": compare_cpu,
        "comparison_limited": comparison_limited,
    }
    state = "enabled" if enabled else "disabled"
    return {
        "stage": stage,
        "enabled": enabled,
        "workload": f"{particles:,} pose hypotheses per fresh scan",
        "report": (
            f"Warp particle localization {state}: {particles:,} hypotheses, "
            f"±{position_spread_m:.2f} m, "
            f"±{math.degrees(yaw_spread_rad):.1f}°; managed SLAM executes the stage"
            + ("; CPU comparison bounded to 4,096 equal-work hypotheses" if comparison_limited else "")
        ),
    }


@node(
    name="WarpDynamicOccupancy",
    component="spatial-processing",
    category=_CATEGORY,
    description=(
        "Configure Warp hash-grid motion classification for a managed SLAM "
        "session. Connect stage to SLAM.dynamic_occupancy; registered scans, "
        "device buffers, velocity estimation, and rendering stay in one live owner."
    ),
    inputs={
        "enabled": Bool(default=True),
        "stable_radius_m": Float(default=0.04),
        "tracking_radius_m": Float(default=0.35),
        "minimum_speed_mps": Float(default=0.04),
        "maximum_age_s": Float(default=0.6),
        "maximum_points": Int(default=65_536),
        "display_points": Int(default=1_500),
        "trail_seconds": Float(default=0.35),
        "compare_cpu": Bool(default=False),
    },
    outputs={
        "stage": Dict,
        "enabled": Bool,
        "workload": Text,
        "report": Text,
    },
    primary_inputs=["enabled", "stable_radius_m", "tracking_radius_m", "minimum_speed_mps"],
    primary_outputs=["stage", "report"],
)
def warp_dynamic_occupancy(ctx: dict) -> dict:
    enabled = bool(ctx.get("enabled", True))
    stable_radius_m = max(0.01, min(1.0, float(ctx.get("stable_radius_m") or 0.04)))
    tracking_radius_m = max(
        stable_radius_m + 0.01,
        min(5.0, float(ctx.get("tracking_radius_m") or 0.35)),
    )
    minimum_speed_mps = max(0.0, min(20.0, float(ctx.get("minimum_speed_mps") or 0.04)))
    maximum_age_s = max(0.05, min(5.0, float(ctx.get("maximum_age_s") or 0.6)))
    maximum_points = max(64, min(65_536, int(ctx.get("maximum_points") or 65_536)))
    display_points = max(16, min(4_000, maximum_points, int(ctx.get("display_points") or 1_500)))
    trail_seconds = max(0.05, min(2.0, float(ctx.get("trail_seconds") or 0.35)))
    stage = {
        "kind": "blacknode.warp-dynamic-occupancy",
        "schema_version": 1,
        "enabled": enabled,
        "stable_radius_m": stable_radius_m,
        "tracking_radius_m": tracking_radius_m,
        "minimum_speed_mps": minimum_speed_mps,
        "maximum_age_s": maximum_age_s,
        "maximum_points": maximum_points,
        "display_points": display_points,
        "trail_seconds": trail_seconds,
        "compare_cpu": bool(ctx.get("compare_cpu", False)),
    }
    state = "enabled" if enabled else "disabled"
    return {
        "stage": stage,
        "enabled": enabled,
        "workload": f"up to {maximum_points:,} registered returns per fresh scan",
        "report": (
            f"Warp dynamic occupancy {state}: stable ≤{stable_radius_m:.2f} m, "
            f"track ≤{tracking_radius_m:.2f} m, moving ≥{minimum_speed_mps:.2f} m/s; "
            "managed SLAM executes the HashGrid stage"
        ),
    }


@node(
    name="WarpDepthProjector",
    component="spatial-processing",
    category=_CATEGORY,
    description=(
        "Configure live metric-depth validation, pinhole deprojection, surface "
        "normals, and confidence for a managed Viewer. Connect a provider-neutral "
        "depth stream to Viewer.source and this stage to Viewer.depth_projection."
    ),
    inputs={
        "enabled": Bool(default=True),
        "minimum_depth_m": Float(default=0.1),
        "maximum_depth_m": Float(default=8.0),
        "downsample_stride": Int(default=2),
        "maximum_points": Int(default=50_000),
        "spatial_filter": Bool(default=True),
        "spatial_max_delta_m": Float(default=0.04),
        "hole_fill": Bool(default=True),
        "minimum_neighbors": Int(default=3),
        "outlier_rejection": Bool(default=True),
        "outlier_max_delta_m": Float(default=0.12),
        "temporal_smoothing": Float(default=0.35),
        "temporal_max_delta_m": Float(default=0.08),
        "stale_after_seconds": Float(default=2.0),
        "target_frame": Text(default="base_link"),
        "sensor_x_m": Float(default=0.0),
        "sensor_y_m": Float(default=0.0),
        "sensor_z_m": Float(default=0.0),
        "sensor_roll_rad": Float(default=0.0),
        "sensor_pitch_rad": Float(default=0.0),
        "sensor_yaw_rad": Float(default=0.0),
        "compare_cpu": Bool(default=False),
    },
    outputs={
        "stage": Dict,
        "enabled": Bool,
        "workload": Text,
        "report": Text,
    },
    primary_inputs=[
        "enabled", "minimum_depth_m", "maximum_depth_m", "downsample_stride",
        "temporal_smoothing",
    ],
    primary_outputs=["stage", "report"],
)
def warp_depth_projector(ctx: dict) -> dict:
    enabled = bool(ctx.get("enabled", True))
    minimum_depth_m = max(0.001, min(100.0, float(ctx.get("minimum_depth_m") or 0.1)))
    maximum_depth_m = max(
        minimum_depth_m,
        min(1_000.0, float(ctx.get("maximum_depth_m") or 8.0)),
    )
    stride = max(1, min(32, int(ctx.get("downsample_stride") or 2)))
    maximum_points = max(64, min(250_000, int(ctx.get("maximum_points") or 50_000)))
    spatial_filter = bool(ctx.get("spatial_filter", True))
    spatial_max_delta_m = max(0.001, min(1.0, float(ctx.get("spatial_max_delta_m", 0.04))))
    hole_fill = bool(ctx.get("hole_fill", True))
    minimum_neighbors = max(2, min(4, int(ctx.get("minimum_neighbors", 3))))
    outlier_rejection = bool(ctx.get("outlier_rejection", True))
    outlier_max_delta_m = max(0.001, min(2.0, float(ctx.get("outlier_max_delta_m", 0.12))))
    temporal_smoothing = max(0.0, min(0.95, float(ctx.get("temporal_smoothing", 0.35))))
    temporal_max_delta_m = max(0.001, min(2.0, float(ctx.get("temporal_max_delta_m", 0.08))))
    stage = {
        "kind": "blacknode.warp-depth-projector",
        "schema_version": 1,
        "enabled": enabled,
        "minimum_depth_m": minimum_depth_m,
        "maximum_depth_m": maximum_depth_m,
        "stride": stride,
        "maximum_points": maximum_points,
        "spatial_filter": spatial_filter,
        "spatial_max_delta_m": spatial_max_delta_m,
        "hole_fill": hole_fill,
        "minimum_neighbors": minimum_neighbors,
        "outlier_rejection": outlier_rejection,
        "outlier_max_delta_m": outlier_max_delta_m,
        "temporal_smoothing": temporal_smoothing,
        "temporal_max_delta_m": temporal_max_delta_m,
        "stale_after_seconds": max(0.1, min(60.0, float(ctx.get("stale_after_seconds") or 2.0))),
        "target_frame": str(ctx.get("target_frame") or "base_link").strip() or "base_link",
        "sensor_x_m": float(ctx.get("sensor_x_m") or 0.0),
        "sensor_y_m": float(ctx.get("sensor_y_m") or 0.0),
        "sensor_z_m": float(ctx.get("sensor_z_m") or 0.0),
        "sensor_roll_rad": float(ctx.get("sensor_roll_rad") or 0.0),
        "sensor_pitch_rad": float(ctx.get("sensor_pitch_rad") or 0.0),
        "sensor_yaw_rad": float(ctx.get("sensor_yaw_rad") or 0.0),
        "compare_cpu": bool(ctx.get("compare_cpu", False)),
    }
    return {
        "stage": stage,
        "enabled": enabled,
        "workload": f"one projected point per {stride} × {stride} depth pixels",
        "report": (
            f"Warp depth projection {'enabled' if enabled else 'disabled'}: "
            f"{minimum_depth_m:.2f}–{maximum_depth_m:.2f} m, stride {stride}; "
            f"edge-aware cleanup and {temporal_smoothing:.2f} temporal smoothing; "
            "managed Viewer owns live binary frames and persistent device buffers"
        ),
    }


@node(
    name="WarpTSDFIntegration",
    component="spatial-processing",
    category=_CATEGORY,
    description=(
        "Configure a bounded device-resident TSDF volume for a managed RGB-D "
        "Viewer. Each fresh pose-registered depth frame updates the persistent volume."
    ),
    inputs={
        "enabled": Bool(default=True),
        "require_pose": Bool(default=True),
        "voxel_size_m": Float(default=0.08),
        "truncation_m": Float(default=0.24),
        "volume_radius_m": Float(default=3.0),
        "volume_origin_x_m": Float(default=0.0),
        "volume_origin_y_m": Float(default=0.0),
        "volume_origin_z_m": Float(default=-1.0),
        "maximum_voxels": Int(default=2_000_000),
        "samples_per_ray": Int(default=7),
        "integrate_color": Bool(default=True),
    },
    outputs={"stage": Dict, "enabled": Bool, "workload": Text, "report": Text},
    primary_inputs=["enabled", "voxel_size_m", "volume_radius_m", "require_pose"],
    primary_outputs=["stage", "report"],
)
def warp_tsdf_integration(ctx: dict) -> dict:
    enabled = bool(ctx.get("enabled", True))
    voxel_size = max(0.01, min(0.5, float(ctx.get("voxel_size_m") or 0.08)))
    radius = max(0.25, min(25.0, float(ctx.get("volume_radius_m") or 3.0)))
    truncation = max(voxel_size, min(2.0, float(ctx.get("truncation_m") or voxel_size * 3.0)))
    maximum_voxels = max(8_000, min(8_000_000, int(ctx.get("maximum_voxels") or 2_000_000)))
    samples = max(3, min(15, int(ctx.get("samples_per_ray") or 7)))
    if samples % 2 == 0:
        samples += 1
    dimension = max(2, int(math.ceil(radius * 2.0 / voxel_size)))
    requested_voxels = dimension ** 3
    stage = {
        "kind": "blacknode.warp-tsdf-integration",
        "schema_version": 1,
        "enabled": enabled,
        "require_pose": bool(ctx.get("require_pose", True)),
        "voxel_size_m": voxel_size,
        "truncation_m": truncation,
        "volume_radius_m": radius,
        "volume_origin_x_m": float(ctx.get("volume_origin_x_m") or 0.0),
        "volume_origin_y_m": float(ctx.get("volume_origin_y_m") or 0.0),
        "volume_origin_z_m": float(ctx.get("volume_origin_z_m") if ctx.get("volume_origin_z_m") is not None else -1.0),
        "maximum_voxels": maximum_voxels,
        "samples_per_ray": samples,
        "integrate_color": bool(ctx.get("integrate_color", True)),
        "requested_voxels": requested_voxels,
    }
    fits = requested_voxels <= maximum_voxels
    return {
        "stage": stage,
        "enabled": enabled,
        "workload": f"{dimension}³ = {requested_voxels:,} persistent TSDF voxels",
        "report": (
            f"Warp TSDF {'enabled' if enabled else 'disabled'}: {voxel_size:.3f} m voxels, "
            f"{truncation:.3f} m truncation, {radius:.2f} m radius, {samples} samples/ray"
            + ("" if fits else f"; configuration exceeds the {maximum_voxels:,}-voxel limit")
        ),
    }


@node(
    name="WarpSurfaceExtraction",
    component="spatial-processing",
    category=_CATEGORY,
    description=(
        "Configure parallel extraction of confidence-colored surface voxels "
        "from the managed Warp TSDF volume for the shared 3D Viewer."
    ),
    inputs={
        "enabled": Bool(default=True),
        "iso_level": Float(default=0.0),
        "surface_band": Float(default=0.2),
        "minimum_weight": Float(default=1.0),
        "maximum_points": Int(default=60_000),
    },
    outputs={"stage": Dict, "enabled": Bool, "workload": Text, "report": Text},
    primary_inputs=["enabled", "surface_band", "minimum_weight", "maximum_points"],
    primary_outputs=["stage", "report"],
)
def warp_surface_extraction(ctx: dict) -> dict:
    enabled = bool(ctx.get("enabled", True))
    stage = {
        "kind": "blacknode.warp-surface-extraction",
        "schema_version": 1,
        "enabled": enabled,
        "iso_level": max(-1.0, min(1.0, float(ctx.get("iso_level") or 0.0))),
        "surface_band": max(0.01, min(1.0, float(ctx.get("surface_band") or 0.2))),
        "minimum_weight": max(1.0, float(ctx.get("minimum_weight") or 1.0)),
        "maximum_points": max(64, min(250_000, int(ctx.get("maximum_points") or 60_000))),
    }
    return {
        "stage": stage,
        "enabled": enabled,
        "workload": f"up to {stage['maximum_points']:,} extracted surface points",
        "report": (
            f"Warp surface extraction {'enabled' if enabled else 'disabled'}: "
            f"iso {stage['iso_level']:.2f} ± {stage['surface_band']:.2f}, "
            f"minimum weight {stage['minimum_weight']:.1f}"
        ),
    }


@node(
    name="WarpSensorFusion",
    component="spatial-processing",
    category=_CATEGORY,
    description=(
        "Configure synchronized LiDAR, metric-depth, and aligned-RGB fusion for "
        "a managed Viewer, including Warp HashGrid residuals and bounded "
        "extrinsic-calibration hypothesis scoring."
    ),
    inputs={
        "enabled": Bool(default=True),
        "require_pose": Bool(default=True),
        "synchronization_tolerance_s": Float(default=0.1),
        "maximum_alignment_distance_m": Float(default=0.35),
        "minimum_depth_confidence": Float(default=0.1),
        "maximum_points": Int(default=60_000),
        "calibration_search": Bool(default=True),
        "calibration_translation_m": Float(default=0.1),
        "calibration_yaw_deg": Float(default=3.0),
        "calibration_steps": Int(default=3),
        "compare_cpu": Bool(default=False),
    },
    outputs={"stage": Dict, "enabled": Bool, "workload": Text, "report": Text},
    primary_inputs=[
        "enabled", "maximum_alignment_distance_m", "calibration_search",
        "calibration_translation_m", "calibration_yaw_deg",
    ],
    primary_outputs=["stage", "report"],
)
def warp_sensor_fusion(ctx: dict) -> dict:
    enabled = bool(ctx.get("enabled", True))
    steps = max(1, min(5, int(ctx.get("calibration_steps") or 3)))
    if steps % 2 == 0:
        steps += 1
    calibration_search = bool(ctx.get("calibration_search", True))
    hypotheses = steps ** 3 if calibration_search else 1
    maximum_points = max(64, min(250_000, int(ctx.get("maximum_points") or 60_000)))
    stage = {
        "kind": "blacknode.warp-sensor-fusion",
        "schema_version": 1,
        "enabled": enabled,
        "require_pose": bool(ctx.get("require_pose", True)),
        "synchronization_tolerance_s": max(
            0.001, min(5.0, float(ctx.get("synchronization_tolerance_s") or 0.1))
        ),
        "maximum_alignment_distance_m": max(
            0.01, min(5.0, float(ctx.get("maximum_alignment_distance_m") or 0.35))
        ),
        "minimum_depth_confidence": max(
            0.0, min(1.0, float(ctx.get("minimum_depth_confidence") or 0.1))
        ),
        "maximum_points": maximum_points,
        "calibration_search": calibration_search,
        "calibration_translation_m": max(
            0.0, min(1.0, float(ctx.get("calibration_translation_m") or 0.1))
        ),
        "calibration_yaw_deg": max(
            0.0, min(20.0, float(ctx.get("calibration_yaw_deg") or 3.0))
        ),
        "calibration_steps": steps,
        "compare_cpu": bool(ctx.get("compare_cpu", False)),
    }
    return {
        "stage": stage,
        "enabled": enabled,
        "workload": (
            f"up to {maximum_points:,} fused points across {hypotheses:,} "
            "calibration hypotheses"
        ),
        "report": (
            f"Warp sensor fusion {'enabled' if enabled else 'disabled'}: "
            f"match ≤{stage['maximum_alignment_distance_m']:.2f} m, "
            f"sync ≤{stage['synchronization_tolerance_s']:.3f} s, "
            f"{hypotheses} calibration hypotheses"
        ),
    }
@node(
    name="WarpTrajectoryEvaluator",
    component="spatial-processing",
    category=_CATEGORY,
    description=(
        "Configure parallel, bounded trajectory scoring for a managed SLAM "
        "session. Connect stage to SLAM.trajectory_evaluation; the stage "
        "visualizes safe and unsafe candidates but never commands motion."
    ),
    inputs={
        "enabled": Bool(default=True),
        "goal_x_m": Float(default=3.0),
        "goal_y_m": Float(default=0.0),
        "trajectories": Int(default=2_048),
        "time_steps": Int(default=48),
        "horizon_s": Float(default=3.0),
        "maximum_linear_speed_mps": Float(default=0.7),
        "maximum_angular_speed_rps": Float(default=1.5),
        "robot_radius_m": Float(default=0.18),
        "clearance_margin_m": Float(default=0.18),
        "display_trajectories": Int(default=96),
        "compare_cpu": Bool(default=False),
    },
    outputs={
        "stage": Dict,
        "enabled": Bool,
        "workload": Text,
        "report": Text,
    },
    primary_inputs=["enabled", "goal_x_m", "goal_y_m", "trajectories", "horizon_s"],
    primary_outputs=["stage", "report"],
)
def warp_trajectory_evaluator(ctx: dict) -> dict:
    enabled = bool(ctx.get("enabled", True))
    requested = max(64, min(65_536, int(ctx.get("trajectories") or 2_048)))
    compare_cpu = bool(ctx.get("compare_cpu", False))
    comparison_limited = bool(compare_cpu and requested > 512)
    trajectories = 512 if comparison_limited else requested
    time_steps = max(8, min(128, int(ctx.get("time_steps") or 48)))
    horizon_s = max(0.25, min(12.0, float(ctx.get("horizon_s") or 3.0)))
    maximum_linear_speed_mps = max(
        0.05,
        min(5.0, float(ctx.get("maximum_linear_speed_mps") or 0.7)),
    )
    maximum_angular_speed_rps = max(
        0.05,
        min(8.0, float(ctx.get("maximum_angular_speed_rps") or 1.5)),
    )
    robot_radius_m = max(0.03, min(2.5, float(ctx.get("robot_radius_m") or 0.18)))
    clearance_margin_m = max(
        0.01,
        min(3.0, float(ctx.get("clearance_margin_m") or 0.18)),
    )
    display_trajectories = max(
        8,
        min(512, trajectories, int(ctx.get("display_trajectories") or 96)),
    )
    stage = {
        "kind": "blacknode.warp-trajectory-evaluator",
        "schema_version": 1,
        "enabled": enabled,
        "goal_x_m": float(ctx.get("goal_x_m") if ctx.get("goal_x_m") is not None else 3.0),
        "goal_y_m": float(ctx.get("goal_y_m") if ctx.get("goal_y_m") is not None else 0.0),
        "trajectory_count": trajectories,
        "requested_trajectories": requested,
        "time_steps": time_steps,
        "horizon_s": horizon_s,
        "maximum_linear_speed_mps": maximum_linear_speed_mps,
        "maximum_angular_speed_rps": maximum_angular_speed_rps,
        "robot_radius_m": robot_radius_m,
        "clearance_margin_m": clearance_margin_m,
        "display_trajectories": display_trajectories,
        "compare_cpu": compare_cpu,
        "comparison_limited": comparison_limited,
        "commands_motion": False,
    }
    state = "enabled" if enabled else "disabled"
    return {
        "stage": stage,
        "enabled": enabled,
        "workload": f"{trajectories:,} trajectories × {time_steps} future steps per fresh scan",
        "report": (
            f"Warp trajectory evaluation {state}: goal "
            f"({stage['goal_x_m']:.2f}, {stage['goal_y_m']:.2f}) m, "
            f"{trajectories:,} bounded candidates over {horizon_s:.2f} s; "
            "visualization and scoring only—motion commands remain disarmed"
            + ("; CPU comparison bounded to 512 equal-work trajectories" if comparison_limited else "")
        ),
    }
