# blacknode-cuda

**Real GPU compute nodes for [Blacknode](https://github.com/temiroff/Blacknode).**

Install this Blacknode **extension package** to add GPU capability, image
processing, tensor-operation, and optional benchmark blocks to the visual
workflow editor. Internal kernels compile and execute on the local NVIDIA GPU
and provide validated primitives for those public components.

## Requirements

- The [Blacknode](https://github.com/temiroff/Blacknode) main app
- An NVIDIA GPU with a CUDA 12.x driver (for actual compute)
- Python deps from `requirements.txt` (CuPy, NumPy, Pillow, Warp, and Pyglet)

No GPU? The package still installs and loads fine — every node returns a
structured "GPU not available" result instead of failing the graph, so
workflows built with these nodes stay viewable and editable on any machine.

## Install

From the Blacknode repo root, the one-liner:

```bash
blacknode packages install git@github.com:temiroff/blacknode-cuda.git
```

Or by hand:

```bash
git clone git@github.com:temiroff/blacknode-cuda.git packages/blacknode-cuda
pip install -r packages/blacknode-cuda/requirements.txt
```

Then restart Blacknode, or press **Reload** in the editor's Packages tab.
Verify with:

```bash
blacknode packages list
# blacknode-cuda 0.1.0 [ok] 8 nodes  .../packages/blacknode-cuda
```

The nodes appear in the editor palette under the **NVIDIA GPU** category, and
the example workflows show up in the Templates tab.

## The nodes

| Node | What it does |
|---|---|
| `CUDAImageFilter` | GPU image filters (grayscale, gaussian blur, sobel edges, invert, ...) wired to Blacknode's image ports — one call, one filtered image |
| `CUDAImageFilterStream` | The same filters running continuously as a live video feed — start/stop a background process that reads an upstream MJPEG source and re-serves its own GPU-filtered stream |
| `WarpLaserScanFilter` | Converts normalized LaserScan ranges to XY, filters and downsamples samples, applies the sensor-to-robot transform, and colors points by distance in a Warp kernel |
| `WarpLiDARViewer` | Starts or stops an interactive OpenGL development view with raw gray and Warp-filtered cyan points |
| `WarpSLAMDiscoveryViewer` | Drives a synthetic rover around a planned route while Warp LiDAR scans reveal unknown obstacles into a persistent RViz-style map |
| `TensorCoreGEMM` | WMMA Tensor Core half-precision matrix multiply via NVRTC |
| `CUTLASS` | CUTLASS tensor operation running through Blacknode's sandboxed worker |
| `CUTLASSGemm` | Optional sustained CUTLASS benchmark |
| `GPUCapability` | Detect the local GPU: name, compute capability, memory, driver |
| `GPURequirement` | Gate a workflow on a minimum GPU capability (preflight check) |

## Components

| Component | Default | Purpose |
|---|---:|---|
| `capability` | On | GPU, driver, and toolkit preflight |
| `image-processing` | On | One-shot and managed-stream image filters |
| `tensor-operations` | On | Tensor Core and CUTLASS operations |
| `spatial-processing` | On | Warp point filtering, transforms, distance colors, and LiDAR visualization |
| `benchmarks` | Off | Sustained benchmark workloads |

Compilation and launch utilities live under `internal/kernels`; they are
package infrastructure rather than a selectable workflow component.

## Templates

Ready-made workflows in `templates/`, loadable from the editor's Templates tab:

- **GPU Image Filter** — load an image, filter it on the GPU, view the result
- **CUDA Image Filter Livestream** — start a ROS 2 camera MJPEG stream, run a
  GPU filter continuously on every frame, and watch the live filtered preview
  update on its own (see **Live video vs. one-shot filtering** below)
- **CUTLASS GPU Burn** — sustained CUTLASS GEMM benchmark
- **CUTLASS Image Showcase** — convolution path on real images

The LiDAR starter workflows are supplied by `blacknode-perception/lidar`, which
owns the normalized sensor capability and ROS 2 adapter. `blacknode-cuda`
supplies the reusable `WarpLaserScanFilter` and `WarpLiDARViewer` compute
blocks. The static lab uses `device=cpu` for portable verification; select
`cuda:0` for GPU execution and the interactive viewer.

In the viewer, the robot origin is red, raw samples are gray, and filtered
samples are cyan with distance shading. The animated sweep accumulates hit
points, draws recent rays in blue, and highlights the active beam and hit in
yellow. The title reports sweep progress and visible/total point counts. Press
**Space** to cycle both/raw/filtered, **P** to pause or resume the sweep, **R**
to restart it, and **Escape** to close.

For million-ray stress tests, the Warp kernel processes every input ray while
`downsample_stride` bounds the point set sent to the OpenGL debug view. The
accumulation view preallocates one fixed GPU point buffer and moves undiscovered
hits outside the camera, avoiding per-frame geometry allocation. Enable
`compare_numpy` to run warmed Warp and NumPy measurements on the same float32
scan. **Space** then cycles cyan Warp, orange NumPy, and a green/red numerical
agreement view while the title reports timings, speedup, and maximum error.

The SLAM discovery viewer is a separate synthetic mapping lab. It starts in a
three-quarter perspective above the arena with the route and floor grid hidden.
Seven semi-transparent ghost obstacle volumes use softly blended
blue/cyan/green faces and brighter luminous edges while GPU-generated radar
rings expand smoothly from the rover. Current returns and persistent occupancy
points use the same spatial gradient from electric blue through cyan to signal
green, with round point sprites and translucent scan rings. Each scan distributes
one million rays across 64 vertical LiDAR layers. Returns are intersected against
full 3D obstacle and boundary volumes, then accumulated into persistent XYZ
voxels so the point cloud grows up walls and around newly viewed faces as the
rover moves. A second persistent layer marks explored free space with soft-white
floor points, while occupied 3D returns retain the blue/cyan/green gradient.
Only the bright current-scan overlay is replaced on each scan; accumulated floor
and voxel points remain until reset. The ring animation stays bounded at seven
rings with 192 segments each. The title reports route progress, scan count,
LiDAR rate, vertical layers, voxel capacity, radar geometry, and current Warp
pipeline time. Set
`show_paths=true` when route debugging is useful. Click and drag the **left
mouse button** to rotate the camera, use **W/A/S/D** or the
arrow keys to move, and use the **mouse wheel** to zoom. Press **P** to pause,
**R** to clear the map and restart, and **Escape** to close.
This lab uses the known synthetic rover pose to demonstrate fast scan
accumulation; localization estimation and loop closure are outside its scope.

On `cuda:0`, the steady-state SLAM path is GPU-resident. Warp kernels raycast,
atomically claim occupancy cells, generate animated scan rings, and write
directly into OpenGL vertex buffers mapped through `RegisteredGLBuffer`. The
scan loop does not call `numpy()`, deduplicate with NumPy, or upload Python
point/line lists. Native `GL_POINTS` renders the accumulated point cloud and
`GL_LINES` renders the bounded radar geometry after CUDA unmaps the buffers.
The title's `pipeline` time covers buffer mapping, Warp kernels, GPU
synchronization, and unmapping; selecting `cpu` retains the registered-buffer
helper's portable copy fallback.

## Live video vs. one-shot filtering

`CUDAImageFilter` is a pure function: one cook, one image in, one filtered
image out. Wiring it after a live camera source and repeatedly re-cooking it
(even with Blacknode's live-recook mode) is not real video — every recook
walks the whole upstream graph again, which is far slower than actual frame
rate.

`CUDAImageFilterStream` is the real video path, matching how
`CameraROS2Subscribe`/`TrackingObject` already work: cook it **once** with
`action=start` and it launches a dedicated background process
(`scripts/cuda_filter_stream_server.py`) that polls an upstream snapshot URL
(e.g. `CameraROS2Subscribe`'s `snapshot_url` output) in a tight loop, filters
each frame on the GPU, and serves its own live MJPEG stream. Wire its
`preview` output into `OutputImage` and the canvas updates live with zero
further cooking. Cook it again with `action=stop` (or a different
`stream_id`) to stop it — this only stops the filter relay, not the
underlying camera stream.

Changing `op`, `amount`, `source_url`, `max_fps`, `max_width`, or
`jpeg_quality` on an already-running filter stream (e.g. picking a different
filter from the editor's dropdown) also takes effect on the next cook without
restarting the process: `start_filter_stream` detects the stream is already
running for that `stream_id` and PATCHes its `/config.json` over HTTP instead
of killing and respawning it. This matters beyond convenience — a naive
restart-on-every-cook also meant any *unrelated* downstream node's Run
would restart this node's whole upstream chain (the graph engine always
re-walks every ancestor on a cook), churning the camera/tracker connections
too. The live-patch path makes that re-walk a cheap no-op instead.

No GPU? The background process still starts and serves its stream endpoints;
`/health.json` reports a structured "CUDA not available" error per frame
instead of crashing, matching the rest of this package's no-GPU contract.

## Updating / removing

```bash
cd packages/blacknode-cuda && git pull     # update
rm -rf packages/blacknode-cuda             # remove — base Blacknode keeps working
```

## Development

Coding agents should read [`AGENTS.md`](AGENTS.md) before changing this package.
It defines the package boundary, GPU fallback contract, benchmark requirements,
and verification commands.

After loading, the modules are importable through Blacknode's stable package
alias:

```python
from blacknode.pkg.blacknode_cuda import cuda
```

The suite in `tests/` runs automatically when you run `pytest` from the
Blacknode repo root (the core collects `packages/*/tests/`). GPU-dependent
tests skip cleanly on machines without CuPy or an NVIDIA GPU.

This package is also the **reference implementation** for writing your own
Blacknode extension package — see
[docs/packages.md](https://github.com/temiroff/Blacknode/blob/master/docs/packages.md)
for the manifest format and discovery rules.

## License

Apache-2.0, same as Blacknode.
