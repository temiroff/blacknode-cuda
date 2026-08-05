"""Generic managed SLAM node backed by Warp LaserScan processing."""
from __future__ import annotations

import math

from blacknode.node import Bool, Dict, Enum, Float, Int, Text, node

from . import slam_runtime


_CATEGORY = "NVIDIA CUDA"
runtime_status = slam_runtime.runtime_status
stop_runtime_services = slam_runtime.stop_runtime_services


def slam_options(values: dict) -> dict:
    """Normalize the shared SLAM node and standalone-application options."""
    return {
        "filter_min_m": max(0.0, float(values.get("filter_min_m") or 0.1)),
        "filter_max_m": max(0.1, float(values.get("filter_max_m") or 12.0)),
        "stride": max(1, int(values.get("downsample_stride") or 1)),
        "sensor_x_m": float(values.get("sensor_x_m") or 0.0),
        "sensor_y_m": float(values.get("sensor_y_m") or 0.0),
        "sensor_yaw_rad": float(values.get("sensor_yaw_rad") or 0.0),
        "robot_length_m": max(0.02, min(5.0, float(values.get("robot_length_m") or 0.25))),
        "robot_width_m": max(0.02, min(5.0, float(values.get("robot_width_m") or 0.22))),
        "robot_height_m": max(0.01, min(2.0, float(values.get("robot_height_m") or 0.08))),
        "map_resolution_m": max(0.01, min(1.0, float(values.get("map_resolution_m") or 0.05))),
        "occupancy_radius_m": max(2.0, min(100.0, float(values.get("occupancy_radius_m") or 20.0))),
        "match_linear_window_m": max(0.05, min(3.0, float(values.get("match_linear_window_m") or 0.4))),
        "match_angular_window_rad": math.radians(max(0.5, min(90.0, float(values.get("match_angular_window_deg") or 10.0)))),
        "tracking_min_score": max(0.0, min(1.0, float(values.get("tracking_min_score") if values.get("tracking_min_score") is not None else 0.2))),
        "mapping_min_score": max(0.0, min(1.0, float(values.get("mapping_min_score") if values.get("mapping_min_score") is not None else 0.3))),
        "keyframe_translation_m": max(0.02, float(values.get("keyframe_translation_m") or 0.15)),
        "keyframe_rotation_rad": math.radians(max(0.5, float(values.get("keyframe_rotation_deg") or 8.0))),
        "keyframe_interval_s": max(0.1, float(values.get("keyframe_interval_s") or 2.0)),
        "loop_closure_radius_m": max(0.1, min(10.0, float(values.get("loop_closure_radius_m") or 1.0))),
        "loop_closure_min_score": max(0.05, min(1.0, float(values.get("loop_closure_min_score") or 0.55))),
        "loop_closure_min_separation": max(3, int(values.get("loop_closure_min_separation") or 20)),
        "max_keyframes": max(20, min(5_000, int(values.get("max_keyframes") or 400))),
        "max_map_points": max(1_000, min(250_000, int(values.get("max_map_points") or 50_000))),
        "pose_sync_tolerance_s": max(0.01, min(10.0, float(values.get("pose_sync_tolerance_s") or 0.25))),
        "pose_parent_frame": str(values.get("pose_parent_frame") or "odom").strip(),
        "pose_child_frame": str(values.get("pose_child_frame") or "auto").strip(),
        "fps": max(1, min(120, int(values.get("fps") or 30))),
        "point_radius_m": 0.025,
        "show_raw": False,
        "show_filtered": True,
        "animate_scan": True,
        "show_rays": True,
    }


@node(
    name="SLAM",
    component="spatial-processing",
    category=_CATEGORY,
    description=(
        "Localize a live LaserScan stream, build a bounded metric map, detect "
        "loop closures, optimize its pose graph, and render the result."
    ),
    inputs={
        "action": Enum(["status", "start", "clear", "pause", "resume", "stop"], default="status"),
        "source": Dict,
        "odometry": Dict,
        "particle_localization": Dict,
        "dynamic_occupancy": Dict,
        "trajectory_evaluation": Dict,
        "slam_id": Text(default="slam"),
        "mode": Enum(["editor", "device"], default="editor"),
        "device": Enum(["cuda:0", "cpu"], default="cuda:0"),
        "filter_min_m": Float(default=0.1),
        "filter_max_m": Float(default=12.0),
        "downsample_stride": Int(default=1),
        "sensor_x_m": Float(default=0.0),
        "sensor_y_m": Float(default=0.0),
        "sensor_yaw_rad": Float(default=0.0),
        "robot_length_m": Float(default=0.25),
        "robot_width_m": Float(default=0.22),
        "robot_height_m": Float(default=0.08),
        "map_resolution_m": Float(default=0.05),
        "occupancy_radius_m": Float(default=20.0),
        "match_linear_window_m": Float(default=0.4),
        "match_angular_window_deg": Float(default=10.0),
        "tracking_min_score": Float(default=0.2),
        "mapping_min_score": Float(default=0.3),
        "keyframe_translation_m": Float(default=0.15),
        "keyframe_rotation_deg": Float(default=8.0),
        "keyframe_interval_s": Float(default=2.0),
        "loop_closure_radius_m": Float(default=1.0),
        "loop_closure_min_score": Float(default=0.55),
        "loop_closure_min_separation": Int(default=20),
        "max_keyframes": Int(default=400),
        "max_map_points": Int(default=50_000),
        "pose_sync_tolerance_s": Float(default=0.25),
        "pose_parent_frame": Text(default="odom"),
        "pose_child_frame": Text(default="auto"),
        "fps": Int(default=30),
    },
    outputs={
        "running": Bool,
        "live": Bool,
        "scene": Dict,
        "pose": Dict,
        "map": Dict,
        "status": Dict,
        "viewer": Dict,
        "report": Text,
    },
    primary_inputs=["source", "odometry", "particle_localization", "dynamic_occupancy", "trajectory_evaluation", "action", "mode"],
    primary_outputs=["scene", "pose", "map", "status", "report"],
    live=True,
)
def slam(ctx: dict) -> dict:
    action = str(ctx.get("action") or "status").strip().lower()
    slam_id = str(ctx.get("slam_id") or "slam").strip()
    if action == "stop":
        stopped = slam_runtime.stop_slam(slam_id)
        return {
            "running": False,
            "live": False,
            "scene": {},
            "pose": {},
            "map": {},
            "status": {"kind": "blacknode.slam-status", "schema_version": 1, "state": "stopped", "error": ""},
            "viewer": {"viewer_id": slam_id, "state": "stopped"},
            "report": f"SLAM stopped {int(stopped.get('stopped') or 0)} session(s)",
        }
    if action == "clear":
        return slam_runtime.clear_slam(slam_id)
    if action == "pause":
        return slam_runtime.set_mapping(slam_id, False)
    if action == "resume":
        return slam_runtime.set_mapping(slam_id, True)
    if action == "status":
        return slam_runtime.slam_status(slam_id)
    if action != "start":
        return {
            "running": False,
            "live": False,
            "scene": {},
            "pose": {},
            "map": {},
            "status": {"state": "error", "error": "Unsupported SLAM action"},
            "viewer": {},
            "report": "SLAM action must be status, start, clear, pause, resume, or stop",
        }
    source = ctx.get("source") if isinstance(ctx.get("source"), dict) else {}
    odometry = ctx.get("odometry") if isinstance(ctx.get("odometry"), dict) else {}
    particle_localization = (
        ctx.get("particle_localization")
        if isinstance(ctx.get("particle_localization"), dict)
        else {}
    )
    dynamic_occupancy = (
        ctx.get("dynamic_occupancy")
        if isinstance(ctx.get("dynamic_occupancy"), dict)
        else {}
    )
    trajectory_evaluation = (
        ctx.get("trajectory_evaluation")
        if isinstance(ctx.get("trajectory_evaluation"), dict)
        else {}
    )
    if particle_localization and particle_localization.get("kind") != "blacknode.warp-particle-localization":
        return {
            "running": False,
            "live": False,
            "scene": {},
            "pose": {},
            "map": {},
            "status": {"state": "error", "error": "particle_localization must be a WarpParticleLocalization stage"},
            "viewer": {},
            "report": "Connect WarpParticleLocalization.stage to SLAM.particle_localization",
        }
    if dynamic_occupancy and dynamic_occupancy.get("kind") != "blacknode.warp-dynamic-occupancy":
        return {
            "running": False,
            "live": False,
            "scene": {},
            "pose": {},
            "map": {},
            "status": {"state": "error", "error": "dynamic_occupancy must be a WarpDynamicOccupancy stage"},
            "viewer": {},
            "report": "Connect WarpDynamicOccupancy.stage to SLAM.dynamic_occupancy",
        }
    if trajectory_evaluation and trajectory_evaluation.get("kind") != "blacknode.warp-trajectory-evaluator":
        return {
            "running": False,
            "live": False,
            "scene": {},
            "pose": {},
            "map": {},
            "status": {"state": "error", "error": "trajectory_evaluation must be a WarpTrajectoryEvaluator stage"},
            "viewer": {},
            "report": "Connect WarpTrajectoryEvaluator.stage to SLAM.trajectory_evaluation",
        }
    reader = ctx.get("__message_stream_reader__")
    return slam_runtime.start_slam(
        slam_id=slam_id,
        node_id=str(ctx.get("__node_id__") or ""),
        source=source,
        odometry_source=odometry,
        mode=str(ctx.get("mode") or "editor"),
        device=str(ctx.get("device") or "cuda:0"),
        options=slam_options(ctx),
        particle_localization=particle_localization,
        dynamic_occupancy=dynamic_occupancy,
        trajectory_evaluation=trajectory_evaluation,
        source_reader=reader if callable(reader) else None,
    )
