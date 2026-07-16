# blacknode-cuda Agent Instructions

This is an independent extension-package repository. Check and commit its Git
state separately from the Blacknode core checkout that may contain it.

## Scope

Keep CUDA, CuPy, NVRTC, CUTLASS, GPU capability detection, and GPU-backed image
stream behavior here. Put generic graph/runtime behavior in Blacknode core.

## Development rules

- Preserve package discovery through `blacknode-package.toml` and `nodes/`.
- Guard GPU imports so the package loads without CUDA, CuPy, or an NVIDIA GPU.
- Return structured capability/runtime errors instead of failing package load.
- Keep one-shot filters separate from managed stream nodes. A stream cook starts
  or updates one background service; it must not poll by repeatedly cooking.
- Synchronize GPU work before reporting timings. Compare against a correct,
  warmed baseline and state the device, dtype, shape, and warmup conditions.
- Avoid silent CPU fallbacks in nodes presented as GPU benchmarks.
- Declare new pip imports and Docker requirements in the manifest and README.
- Add or update a package template when it is the clearest integration proof.

## Verification

From the Blacknode root:

```powershell
python -m pytest packages/blacknode-cuda/tests
```

GPU-dependent tests must skip clearly when hardware is unavailable. Also
validate any changed package template with `blacknode validate`.

See the Blacknode `docs/packages.md` and `blacknode-development` skill for the
shared package contract.
