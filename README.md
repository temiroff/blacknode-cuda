# blacknode-cuda

`blacknode-cuda` adds GPU capability checks, image processing, tensor operations, Warp spatial processing, and optional benchmarks to Blacknode workflows.

## Requirements and install

GPU execution requires an NVIDIA GPU and compatible CUDA 12.x driver. The package still loads when CUDA or optional GPU libraries are unavailable and returns structured readiness errors.

```powershell
blacknode packages install https://github.com/temiroff/blacknode-cuda.git
```

Use **Packages → blacknode-cuda → Install prerequisites** after installation.

## Standalone Jetson SLAM

The spatial-processing component includes a named `slam` application and the
same React/WebGL viewer used by the Blacknode editor node:

```bash
blacknode run slam --device cuda:0
```

It subscribes to `/scan` and `/odom`, runs filtering, matching, occupancy ray
tracing, and map updates through the managed Warp SLAM runtime, serves the
packaged viewer at `http://127.0.0.1:7780`, and opens it locally. The viewer
retains the editor controls, camera behavior, colors, robot marker, live sweep,
floor fill, walls, and runtime metrics. `Ctrl+C` stops SLAM and the two ROS 2
subscriptions cleanly.

The standalone application enables Warp trajectory evaluation by default.
Hold **Shift** and left-click the map to place a fixed map-frame goal; the live
session immediately rescores its candidate paths without restarting SLAM.

Useful options:

```bash
blacknode run slam --fullscreen
blacknode run slam --no-open --host 0.0.0.0
blacknode run slam --scan-topic /scan --odometry-topic /odom
blacknode run slam --native --device cuda:0
```

`--native` selects the CUDA-registered OpenGL point-buffer viewer. The packaged
WebGL viewer remains the exact editor-view presentation and keeps Warp CUDA for
SLAM calculations. Binding beyond loopback exposes the viewer to the selected
network interface; use the managed device network policy for remote access.

Package releases build the viewer from the shared editor component with:

```bash
cd editor
npm run build:slam-viewer
```

## Components

| Component | Default | Main nodes |
|---|---:|---|
| `capability` | On | `GPUCapability`, `GPURequirement` |
| `image-processing` | On | `CUDAImageFilter`, `CUDAImageFilterStream` |
| `tensor-operations` | On | `TensorCoreGEMM`, `CUTLASS` |
| `spatial-processing` | On | `Viewer`, `SLAM`, `WarpParticleLocalization`, `WarpDynamicOccupancy`, `WarpDepthProjector`, `WarpTSDFIntegration`, `WarpSurfaceExtraction`, `WarpSensorFusion`, `WarpLaserScanFilter` |
| `benchmarks` | Off | `CUTLASSGemm` |

`CUDAImageFilter` processes one image per cook. `CUDAImageFilterStream` manages a live MJPEG filter service. `Viewer` connects to a scan stream and optional pose streams, processes LaserScan messages with Warp, and renders in the editor or a native OpenGL window. Fresh `Odometry` and `PoseStamped` messages register scan history directly. A `TFMessage` stream can chain `pose_parent_frame` to `pose_child_frame`; `auto` targets the LaserScan frame. Connect `/tf_static` through a second generic ROS2 stream with `qos=transient_local` when fixed links are separate. The view provides 3D orbit controls, a counterclockwise ray sweep, reported angular coverage, bounded history, and an accumulation toggle; Off freezes the registered world cloud while the latest sweep remains visible on top. LaserScan geometry remains planar. Older specialized viewer types remain available for saved workflows but are hidden from new graphs.

`SLAM` consumes the same generic scan stream and optional odometry stream. It retains every valid range sample by default after range filtering, deskews beams against synchronized odometry across each scan, performs coarse-to-fine scan matching, confidence-gates tracking corrections and map insertion, and averages repeated observations in metric map cells. Every valid real LiDAR return also launches one Warp ray-tracing thread: traversed cells fill a persistent free-floor occupancy layer whose world origin and cell centers stay fixed for the session, while repeated endpoints become confidence-gated occupied wall cells. Warp classifies and packs the complete grid at two bits per cell; the editor uploads that fixed-size payload directly as a nearest-neighbor WebGL texture, preserving every cell while avoiding growing point/color JSON. On CUDA, the fixed reference grid remains GPU-resident and Warp scores the complete coarse-to-fine pose-candidate set in parallel; CPU mode retains the NumPy matcher and caches its expanded lookup until the reference changes. The editor renders the fixed unknown extent, filled free floor, occupied walls, live scan, and robot as distinct layers. The viewer reports the Warp device, ray count, free and occupied cell counts, grid size, synchronized kernel/copy and matching time, and encoding time so the accelerated work is directly inspectable. `occupancy_radius_m` bounds the fixed grid, while `map_resolution_m` controls its cell size. Fresh stationary odometry prevents a moving person or object from dragging the fixed map, while improved whole-scan evidence overrides lagging odometry as soon as the scan matcher resolves motion. Large delayed corrections are bounded per scan so the robot pose catches up progressively instead of jumping. With accumulation off, the visible map remains frozen and a hidden registered scan keeps localization active; after Clear, the first new scan establishes that tracking reference while accumulated points stay empty. It adds loop-closure constraints, optimizes its pose graph, and publishes the map, estimated pose, status, and interactive scene. The scene keeps discovered points in the fixed `map` frame while the robot pose and current scan move through it. Tune `tracking_min_score` to reject uncertain pose corrections and `mapping_min_score` to prevent uncertain keyframes from entering the dark accumulated map. In `mode=device`, its native Jetson viewer writes Warp output directly into CUDA-registered OpenGL point buffers and falls back to the standard renderer when graphics interop is unavailable. Open the **ROS2 SLAM** template, select the paired device, set the topic names, and press **Go live**.

`WarpParticleLocalization` is a graph-visible compute stage for `SLAM`. Connect
its `stage` output to `SLAM.particle_localization` to score up to 65,536 pose
hypotheses against every fresh scan while the managed SLAM session retains GPU
ownership. The viewer draws a confidence-colored hypothesis cloud and reports
hypotheses × beams, total score evaluations, synchronized pipeline time, an
explicit CPU limit when CUDA is unavailable, and optional same-workload CPU
comparison. Phase-one pipeline timing includes hypothesis upload, Warp scoring,
synchronization, and score readback. Open **ROS2 Warp Particle Localization**
for the complete wiring.

`WarpDynamicOccupancy` is a graph-visible motion-analysis stage for `SLAM`.
Connect its `stage` output to `SLAM.dynamic_occupancy`; the managed session
compares consecutive pose-registered scans using a Warp `HashGrid`, keeps fixed
returns in the cyan map, and overlays coherent motion as orange points. The
viewer reports query count, moving returns, synchronized
HashGrid pipeline time, mean speed, and an optional same-workload CPU
comparison. Motion is evaluated against a held temporal reference so slow
movement accumulates above sensor jitter, unmatched returns remain transient
until confirmed, and only confirmed-static endpoints contribute occupied map
evidence. Free rays reduce bounded occupancy evidence so removed objects clear
from the map. A Warp coherence pass rejects isolated edge flicker, and only
confirmed-static returns can influence scan matching so a moving foreground
object cannot drag the fixed walls. The viewer marks coherent motion with
orange points only. The Warp classifier also checks the persistent occupied
grid before temporal matching, so wall returns revealed after an occluder moves
are restored directly as static background instead of flashing as motion. Open
**ROS2 Warp Dynamic Occupancy** for the complete wiring.

`WarpTrajectoryEvaluator` is a graph-visible, visualization-only planning stage
for `SLAM`. Connect its `stage` output to `SLAM.trajectory_evaluation`; the
managed session scores thousands of bounded differential-drive arcs against
the fixed occupancy map and predicted coherent motion. The viewer draws unsafe
paths red, safe paths green, the highest-scoring safe candidate cyan, and
reports trajectories × future steps plus synchronized Warp pipeline time. The
stage never arms or commands the robot. In the embedded SLAM viewer, hold
**Shift** and left-click to update the connected evaluator's `goal_x_m` and
`goal_y_m` values and rescore the running session directly. Open **ROS2 Warp
Navigation Lab** for the complete wiring.

`WarpDepthProjector` is a graph-visible metric-depth stage for `Viewer`.
Connect a provider-neutral `blacknode.depth-stream` to `Viewer.source` and the
stage to `Viewer.depth_projection`. The managed viewer fetches compact binary
frames from live providers, validates depth, deprojects calibrated pixels,
applies the configured sensor extrinsics, estimates surface normals and
confidence, and renders the current 3D surface in the selected target frame.
Dense depth pixels remain outside workflow JSON. Worker presence and frame
freshness are reported separately; stale binary frames stop live presentation.
Open **ROS2 Warp Depth Cloud** for the complete wiring.

`WarpTSDFIntegration` and `WarpSurfaceExtraction` extend that same managed
viewer into persistent RGB-D reconstruction. Each fresh calibrated depth frame
is aligned with the latest RGB snapshot, registered by synchronized odometry,
integrated into a bounded device-resident TSDF volume, and extracted as a
compact colored surface scene. Duplicate frames do not reintegrate. The viewer
reports integration samples, observed and surface voxels, RGB registration,
pose registration, and synchronized Warp timing. **Clear volume** resets the
persistent volume and pauses integration; **Accumulate** resumes it. Open
**Warp TSDF Reconstruction Lab** for hardware-free verification or **ROS2 Warp
RGB-D Reconstruction** for the live sensor graph.

`WarpSensorFusion` combines the current LiDAR scan and projected metric-depth
surface in the shared 3D viewer, with the latest aligned RGB coloring the depth
geometry. A bounded Warp `HashGrid` search evaluates small translation and yaw
calibration hypotheses, then publishes matched and unmatched geometry,
residual heat colors, confidence, frame synchronization error, and the selected
extrinsic correction. The managed viewer keeps dense sensor frames outside
workflow JSON and rejects pairs outside the configured synchronization window.
Open **Warp Sensor Fusion Lab** for hardware-free verification or **ROS2 Warp
Sensor Fusion** for aligned camera, `/scan`, and `/odom` streams.

## Included workflows

- GPU image filtering and live filter streaming
- CUTLASS image and sustained GEMM examples
- LiDAR starter workflows supplied by `blacknode-perception`, which owns the normalized sensor contract
- ROS 2 device stream to the generic live `Viewer`
- ROS 2 scan and odometry streams to live `SLAM`
- ROS 2 scan and odometry streams through a visible `WarpParticleLocalization`
  stage into live `SLAM`
- ROS 2 scan and odometry streams through a visible `WarpDynamicOccupancy`
  stage into live `SLAM`
- ROS 2 scan and odometry streams through visible `WarpDynamicOccupancy` and
  `WarpTrajectoryEvaluator` stages into live `SLAM`
- ROS 2 metric depth through `WarpDepthProjector` into the shared live `Viewer`
- ROS 2 depth, aligned RGB, and odometry through persistent Warp TSDF
  integration and surface extraction into the shared live `Viewer`
- LiDAR, metric depth, aligned RGB, and pose through `WarpSensorFusion` into a
  color-separated shared 3D alignment view

See [Warp Sensor Compute Roadmap](SENSOR_GPU_ROADMAP.md) for the delivered
localization and dynamic-occupancy stages plus trajectory evaluation, live
depth projection, RGB-D reconstruction, and LiDAR/depth/RGB fusion.

Select `cpu` for portable Warp verification or `cuda:0` for GPU execution. Enable `compare_numpy` when a warmed correctness and timing comparison is needed. Benchmarks report the device, data shape/type, warmup conditions, synchronized timing, and correctness result.

## Development

```powershell
python -m pytest packages/blacknode-cuda/tests
```

GPU-dependent tests skip when hardware is unavailable. Validate changed templates with `blacknode validate`. See [AGENTS.md](AGENTS.md) for GPU fallback and benchmark rules.
