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
