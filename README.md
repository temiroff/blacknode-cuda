# blacknode-cuda

**Real GPU compute nodes for [Blacknode](https://github.com/temiroff/Blacknode).**

Install this Blacknode **extension package** to add CUDA compute blocks to the
visual workflow editor: kernels
compile and execute on your local NVIDIA GPU and report measured timings,
CPU baselines, speedups, and correctness checks.

## Requirements

- The [Blacknode](https://github.com/temiroff/Blacknode) main app
- An NVIDIA GPU with a CUDA 12.x driver (for actual compute)
- Python deps from `requirements.txt` (CuPy, NumPy, Pillow)

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
| `CUDAKernelLab` | Curated GPU ops (vector add, saxpy, matmul, softmax, FFT, mandelbrot, monte-carlo π, ...) with measured GPU vs CPU timings and a NumPy correctness check |
| `CUDACustomKernel` | Write your own CUDA C kernel in the node, compiled at runtime with NVRTC (`cupy.RawKernel`) — includes starter templates |
| `CUDAImageFilter` | GPU image filters (grayscale, gaussian blur, sobel edges, invert, ...) wired to Blacknode's image ports — one call, one filtered image |
| `CUDAImageFilterStream` | The same filters running continuously as a live video feed — start/stop a background process that reads an upstream MJPEG source and re-serves its own GPU-filtered stream |
| `TensorCoreGEMM` | WMMA Tensor Core half-precision matrix multiply via NVRTC |
| `CUTLASS` / `CUTLASSGemm` | CUTLASS GEMM running through Blacknode's sandboxed worker |
| `GPUCapability` | Detect the local GPU: name, compute capability, memory, driver |
| `GPURequirement` | Gate a workflow on a minimum GPU capability (preflight check) |

## Templates

Ready-made workflows in `templates/`, loadable from the editor's Templates tab:

- **NVIDIA CUDA Lab** — run and benchmark the curated op catalogue
- **GPU Image Filter** — load an image, filter it on the GPU, view the result
- **CUDA Image Filter Livestream** — start a ROS 2 camera MJPEG stream, run a
  GPU filter continuously on every frame, and watch the live filtered preview
  update on its own (see **Live video vs. one-shot filtering** below)
- **CUTLASS GPU Burn** — sustained CUTLASS GEMM benchmark
- **CUTLASS Image Showcase** — convolution path on real images

## Live video vs. one-shot filtering

`CUDAImageFilter` is a pure function: one cook, one image in, one filtered
image out. Wiring it after a live camera source and repeatedly re-cooking it
(even with Blacknode's live-recook mode) is not real video — every recook
walks the whole upstream graph again, which is far slower than actual frame
rate.

`CUDAImageFilterStream` is the real video path, matching how
`ROS2ImageStream`/`CV2ColorObjectStream` already work: cook it **once** with
`action=start` and it launches a dedicated background process
(`scripts/cuda_filter_stream_server.py`) that polls an upstream snapshot URL
(e.g. `ROS2ImageStream`'s `snapshot_url` output) in a tight loop, filters
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
