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
| `spatial-processing` | On | `WarpLaserScanFilter`, `WarpLiDARViewer`, `WarpSLAMDiscoveryViewer` |
| `benchmarks` | Off | `CUTLASSGemm` |

`CUDAImageFilter` processes one image per cook. `CUDAImageFilterStream` starts or updates one managed MJPEG filter service; frames continue without graph recooks. Warp nodes convert and filter LaserScan data, transform points, and provide interactive LiDAR/SLAM development views.

## Included workflows

- GPU image filtering and live filter streaming
- CUTLASS image and sustained GEMM examples
- LiDAR starter workflows supplied by `blacknode-perception`, which owns the normalized sensor contract

Select `cpu` for portable Warp verification or `cuda:0` for GPU execution. Enable `compare_numpy` when a warmed correctness and timing comparison is needed. Benchmarks report the device, data shape/type, warmup conditions, synchronized timing, and correctness result.

## Development

```powershell
python -m pytest packages/blacknode-cuda/tests
```

GPU-dependent tests skip when hardware is unavailable. Validate changed templates with `blacknode validate`. See [AGENTS.md](AGENTS.md) for GPU fallback and benchmark rules.
