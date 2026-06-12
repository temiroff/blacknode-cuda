# blacknode-cuda

**Real GPU compute nodes for [Blacknode](https://github.com/temiroff/Blacknode).**

This is a Blacknode **extension package** — it does not run on its own. It
plugs CUDA compute blocks into the Blacknode visual workflow editor: kernels
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
| `CUDAImageFilter` | GPU image filters (grayscale, gaussian blur, sobel edges, invert, ...) wired to Blacknode's image ports |
| `TensorCoreGEMM` | WMMA Tensor Core half-precision matrix multiply via NVRTC |
| `CUTLASS` / `CUTLASSGemm` | CUTLASS GEMM running through Blacknode's sandboxed worker |
| `GPUCapability` | Detect the local GPU: name, compute capability, memory, driver |
| `GPURequirement` | Gate a workflow on a minimum GPU capability (preflight check) |

## Templates

Ready-made workflows in `templates/`, loadable from the editor's Templates tab:

- **NVIDIA CUDA Lab** — run and benchmark the curated op catalogue
- **GPU Image Filter** — load an image, filter it on the GPU, view the result
- **CUTLASS GPU Burn** — sustained CUTLASS GEMM benchmark
- **CUTLASS Image Showcase** — convolution path on real images

## Updating / removing

```bash
cd packages/blacknode-cuda && git pull     # update
rm -rf packages/blacknode-cuda             # remove — base Blacknode keeps working
```

## Development

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
