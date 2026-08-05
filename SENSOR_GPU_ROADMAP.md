# Warp Sensor Compute Roadmap

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
- Visual: fixed walls remain cyan; moving objects and velocity trails use warm
  colors.
- Node: `WarpDynamicOccupancy` stage connected to `SLAM.dynamic_occupancy`.
- Template: `ROS2 Warp Dynamic Occupancy`.

### 3. Parallel trajectory evaluation

- Inputs: occupancy/clearance field, robot state, goal and dynamic obstacles.
- Work: evaluate thousands of bounded candidate trajectories over future time
  steps. This stage only scores trajectories; it never arms or commands motion.
- Visual: unsafe paths red, safe paths green and the best candidate cyan.
- Planned node: `WarpTrajectoryEvaluator`.
- Planned template: `ROS2 Warp Navigation Lab`.

### 4. Live depth projection

- Prerequisite: a provider-neutral depth-frame contract that exposes metric
  depth and calibration to a managed GPU consumer without JSON-expanding every
  pixel. `blacknode-perception` remains the depth-camera capability owner.
- Inputs: depth frames, camera intrinsics and calibrated sensor extrinsics.
- Work: depth validation, deprojection, filtering, normals and confidence.
- Visual: a live metric 3D point surface using the shared viewer controls.
- Planned stage: `WarpDepthProjector`.
- Planned template: `ROS2 Warp Depth Cloud`.

### 5. RGB-D volumetric reconstruction

- Inputs: projected depth, color and registered camera pose.
- Work: sparse TSDF integration, occupancy/free-space evidence and surface
  extraction.
- Visual: a holographic room surface fills in while the camera moves.
- Planned stages: `WarpTSDFIntegration` and `WarpSurfaceExtraction`.
- Planned template: `ROS2 Warp RGB-D Reconstruction`.

### 6. LiDAR, depth and camera fusion

- Inputs: stable LiDAR, depth-camera and camera contracts plus calibrated
  extrinsics and synchronized timestamps.
- Work: fused occupancy, LiDAR-to-image projection and batched calibration
  hypothesis scoring.
- Visual: colorized geometry, cross-sensor alignment residuals and confidence.
- Planned template: `ROS2 Warp Sensor Fusion Lab`.

## Benchmark contract

Every accelerated stage reports the selected device, data dimensions, warm-up
state, synchronized pipeline time, transfer time when applicable and an
equivalent CPU result when comparison is enabled. GPU speedup claims require
the same input and workload on both backends. Hardware-free tests exercise the
CPU or replay contract and do not establish Jetson timing.
