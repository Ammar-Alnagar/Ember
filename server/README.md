# Ember Python gRPC Server

Python gRPC server that loads models, runs inference, and communicates with the Rust router.

## Requirements

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)
- CUDA 13.2+ for GPU acceleration

## Install

```bash
# From the repo root — recommended
make install-server        # GPU build
make install-server-cpu    # CPU-only build

# Or directly
cd server
uv sync --frozen \
  --extra gen --extra bnb --extra accelerate \
  --extra compressed-tensors --extra quantize \
  --extra peft --extra outlines --extra torch \
  --active --python=3.11
make gen-server-raw
kernels download .         # pre-built GPU kernel artifacts
```

## Run (development)

```bash
make server-dev
# or
cd server && make run-dev
```

## GPU kernels

The server uses Rust GPU kernels from the `backends/cutile-kernels` crate (cuTile Rust).
They are compiled as part of `cargo build` — no separate step is needed.

See the [GPU Kernels guide](../docs/source/conceptual/cutile_kernels.md) for details.

## Quantization

The server supports: bitsandbytes, GPTQ, AWQ, EETQ, Marlin, EXL2, fp8.

Pass `--quantize <method>` to `text-generation-server serve` or use the launcher flag.

## Tests

```bash
make python-server-tests   # from repo root
# or
cd server && pytest tests/
```
