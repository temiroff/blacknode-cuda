# Sensor Viewer and Warp Compute Roadmap

Blacknode exposes Warp sensor calculations as reusable graph stages while a
managed viewer or mapping node owns live scheduling and device-resident memory.
This keeps workflows understandable and keeps every sensor frame out of the
one-shot cook path.

## Architecture

```text
sensor stream -> Warp stage descriptor -> managed compute/viewer -> scene
```

- Stage nodes declare algorithms and bounded options through typed ports.
- A stage output is connected visibly to the managed node that executes it.
- The managed node launches kernels for every fresh sensor frame and publishes
  live outputs through the existing runtime-status contract.
- Stage implementations progressively keep intermediate arrays on the selected
  Warp device. The phase-one baseline includes hypothesis upload, scoring,
  synchronization and score readback in its reported pipeline time; later
  phases can make those buffers persistent without changing graph wiring.
  Scene payloads carry compact display samples rather than full compute state.
- Templates provide one clear demonstration per workload. A combined sensor
  lab can compose the same stages after their individual contracts are stable.

## Sensor-to-viewer list

Each normalized physical sensor has a dedicated diagnostic view so an operator
can inspect that feed independently before relying on downstream fusion or
autonomy.

| Sensor | Dedicated viewer | Status | What the operator sees |
|---|---|---|---|
| RGB camera | `Camera` / `CameraStream` preview | Implemented | Live image, stream health and freshness |
| 2D LiDAR | `Viewer` | Implemented | Live scan rays, filtered points, coverage and registered history |
| LiDAR SLAM | `SLAM` | Implemented | Map, B-logo robot pose, scan, occupancy and localization state |
| Metric depth | `Viewer` + `WarpDepthProjector` | Implemented | Calibrated 3D depth surface, confidence and projection timing |
| IMU | `IMUViewer` | Implemented | Quaternion-driven 3D B-logo robot, body/world XYZ axes, roll/pitch/yaw, rates, acceleration and freshness |
| RGB-D reconstruction | `Viewer` + `WarpTSDFIntegration` + `WarpSurfaceExtraction` | Implemented | Persistent colored room surface, coverage and timing |
| Cross-sensor fusion | `Viewer` + `WarpSensorFusion` | Implemented | LiDAR and colorized depth geometry, alignment residuals, synchronization and confidence |

The IMU view is a sensor-state diagnostic and does not claim GPU acceleration;
it lives in `blacknode-perception` so it remains available on systems with no
CUDA device. Warp stages remain graph-visible where parallel sensor compute is
materially useful.

## Delivery phases

### 1. Dense LiDAR pose hypotheses

Status: implemented in `WarpParticleLocalization`.

- Inputs: SLAM map, current LaserScan and pose prior owned by `SLAM`.
- Work: score thousands of pose hypotheses against every valid LiDAR return.
- Visual: confidence-colored particle cloud and uncertainty around the robot.
- Metrics: hypotheses, beams, total score evaluations, synchronized
  upload/score/readback pipeline time, optional CPU comparison and correctness
  error.
- Template: `ROS2 Warp Particle Localization`.

### 2. Dynamic occupancy

Status: implemented in `WarpDynamicOccupancy`.

- Inputs: registered LiDAR history and optional depth point cloud.
- Work: Warp hash-grid neighbor searches, temporal residuals and velocity
  estimates.
- Visual: fixed walls remain cyan and coherent moving objects use orange points.
- Node: `WarpDynamicOccupancy` stage connected to `SLAM.dynamic_occupancy`.
- Template: `ROS2 Warp Dynamic Occupancy`.

### 3. Parallel trajectory evaluation

Status: implemented in `WarpTrajectoryEvaluator`.

- Inputs: occupancy/clearance field, robot state, goal and dynamic obstacles.
- Work: evaluate thousands of bounded candidate trajectories over future time
  steps. This stage only scores trajectories; it never arms or commands motion.
- Visual: unsafe paths red, safe paths green and the best candidate cyan.
- Node: `WarpTrajectoryEvaluator` connected to `SLAM.trajectory_evaluation`.
- Template: `ROS2 Warp Navigation Lab`.

### 4. Live depth projection

Status: implemented in `WarpDepthProjector`.

- Prerequisite: a provider-neutral depth-frame contract that exposes metric
  depth and calibration to a managed GPU consumer without JSON-expanding every
  pixel. `blacknode-perception` remains the depth-camera capability owner.
- Inputs: depth frames, camera intrinsics and calibrated sensor extrinsics.
- Work: depth validation, deprojection, filtering, normals and confidence.
- Visual: a live metric 3D point surface using the shared viewer controls.
- Stage: `WarpDepthProjector` connected to `Viewer.depth_projection`.
- Template: `ROS2 Warp Depth Cloud`.

### 5. Live IMU orientation

Status: implemented in `IMUViewer` from `blacknode-perception`.

- Inputs: a normalized `blacknode.imu-stream` or generic ROS 2
  `sensor_msgs/msg/Imu` message stream.
- Work: normalize and validate the quaternion, derive roll/pitch/yaw, preserve
  angular velocity, linear acceleration, frame, message age and freshness.
- Visual: a separate 3D B-logo robot with rotated body XYZ axes and a fixed
  world frame; rotating the physical robot updates the model live.
- Safety: messages that declare orientation unavailable and zero-length
  quaternions are rejected; stale samples remain visible but are not labeled
  live.
- Templates: `IMU Orientation Lab` and `ROS 2 IMU Orientation Viewer`.

### 6. RGB-D volumetric reconstruction

Status: implemented in `WarpTSDFIntegration` and `WarpSurfaceExtraction`.

- Inputs: projected depth, color and registered camera pose.
- Work: bounded device-resident TSDF integration with color evidence and
  parallel confidence-weighted surface extraction.
- Visual: a holographic room surface fills in while the camera moves.
- Stages: `WarpTSDFIntegration` and `WarpSurfaceExtraction` connected to the
  managed `Viewer`; volume state stays outside workflow JSON.
- Templates: `Warp TSDF Reconstruction Lab` and `ROS2 Warp RGB-D Reconstruction`.

### 7. LiDAR, depth and camera fusion

Status: implemented in `WarpSensorFusion`.

- Inputs: stable LiDAR, depth-camera and camera contracts plus calibrated
  extrinsics and synchronized timestamps.
- Work: pose-register LiDAR and metric depth, color depth with the aligned RGB
  stream, query nearest LiDAR neighbors with a Warp `HashGrid`, and score a
  bounded translation/yaw calibration lattice.
- Visual: cyan LiDAR, RGB depth points blended from green alignment to red
  residual, matched/unmatched counts, residual percentiles, confidence,
  synchronization delta and selected calibration correction.
- Safety and bounds: stale or unsynchronized pairs are rejected, point counts
  and calibration hypotheses are capped, and the stage does not command robot
  motion.
- Templates: `Warp Sensor Fusion Lab` and `ROS2 Warp Sensor Fusion`.

## Benchmark contract

Every accelerated stage reports the selected device, data dimensions, warm-up
state, synchronized pipeline time, transfer time when applicable and an
equivalent CPU result when comparison is enabled. GPU speedup claims require
the same input and workload on both backends. Hardware-free tests exercise the
CPU or replay contract and do not establish Jetson timing.
