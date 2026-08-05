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
        "stable_radius_m": Float(default=0.08),
        "tracking_radius_m": Float(default=0.45),
        "minimum_speed_mps": Float(default=0.12),
        "maximum_age_s": Float(default=0.5),
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
    stable_radius_m = max(0.01, min(1.0, float(ctx.get("stable_radius_m") or 0.08)))
    tracking_radius_m = max(
        stable_radius_m + 0.01,
        min(5.0, float(ctx.get("tracking_radius_m") or 0.45)),
    )
    minimum_speed_mps = max(0.0, min(20.0, float(ctx.get("minimum_speed_mps") or 0.12)))
    maximum_age_s = max(0.05, min(5.0, float(ctx.get("maximum_age_s") or 0.5)))
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
