# blacknode-cuda

`blacknode-cuda` adds GPU capability checks, image processing, tensor operations, Warp spatial processing, and optional benchmarks to Blacknode workflows.

## Requirements and install

GPU execution requires an NVIDIA GPU and compatible CUDA 12.x driver. The package still loads when CUDA or optional GPU libraries are unavailable and returns structured readiness errors.

```powershell
blacknode packages install https://github.com/temiroff/blacknode-cuda.git
```

Use **Packages → blacknode-cuda → Install prerequisites** after installation.

## Components

| Component | Default | Main nodes |
|---|---:|---|
| `capability` | On | `GPUCapability`, `GPURequirement` |
| `image-processing` | On | `CUDAImageFilter`, `CUDAImageFilterStream` |
| `tensor-operations` | On | `TensorCoreGEMM`, `CUTLASS` |
| `spatial-processing` | On | `Viewer`, `WarpLaserScanFilter` |
| `benchmarks` | Off | `CUTLASSGemm` |

`CUDAImageFilter` processes one image per cook. `CUDAImageFilterStream` manages a live MJPEG filter service. `Viewer` connects to a scan stream and optional pose streams, processes LaserScan messages with Warp, and renders in the editor or a native OpenGL window. Fresh `Odometry` and `PoseStamped` messages register scan history directly. A `TFMessage` stream can chain `pose_parent_frame` to `pose_child_frame`; `auto` targets the LaserScan frame. Connect `/tf_static` through a second generic ROS2 stream with `qos=transient_local` when fixed links are separate. The view provides 3D orbit controls, a counterclockwise ray sweep, reported angular coverage, bounded history, and an accumulation toggle. LaserScan geometry remains planar. Older specialized viewer types remain available for saved workflows but are hidden from new graphs.

## Included workflows

- GPU image filtering and live filter streaming
- CUTLASS image and sustained GEMM examples
- LiDAR starter workflows supplied by `blacknode-perception`, which owns the normalized sensor contract
- ROS 2 device stream to the generic live `Viewer`

Select `cpu` for portable Warp verification or `cuda:0` for GPU execution. Enable `compare_numpy` when a warmed correctness and timing comparison is needed. Benchmarks report the device, data shape/type, warmup conditions, synchronized timing, and correctness result.

## Development

```powershell
python -m pytest packages/blacknode-cuda/tests
```

GPU-dependent tests skip when hardware is unavailable. Validate changed templates with `blacknode validate`. See [AGENTS.md](AGENTS.md) for GPU fallback and benchmark rules.
