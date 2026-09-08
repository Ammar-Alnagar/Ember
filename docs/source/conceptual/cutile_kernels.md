# GPU Kernels (cuTile Rust)

Ember uses [cuTile Rust](https://github.com/NVlabs/cutile-rs) from NVlabs to implement its
performance-critical GPU kernels entirely in safe Rust, replacing the previous CUDA C++ extension
modules.

## What are the cuTile kernels?

cuTile Rust is a tile-based GPU kernel DSL that compiles Rust code directly to CUDA.
Kernels are expressed as ordinary Rust functions annotated with `#[cutile::entry()]` and
JIT-compiled to optimized GPU code on first launch. Subsequent calls reuse the cached binary.

The kernel crate lives at `backends/cutile-kernels/` and is part of the Rust workspace.

## Implemented kernels

| Kernel | Source file | Replaces |
|--------|------------|----------|
| `masked_softmax_f32` | `src/softmax.rs` | `fused_attention_cuda.cu` |
| `masked_softmax_f16` | `src/softmax.rs` | `fused_attention_cuda.cu` |
| `masked_softmax_bf16` | `src/softmax.rs` | `fused_bloom_attention_cuda.cu` |
| `q4_reconstruct` | `src/quant.rs` | `q4_matrix.cu` |
| `q4_matmul` | `src/quant.rs` | `q4_matmul.cu` |
| `column_remap` | `src/quant.rs` | column remap utility |

### Masked softmax

Computes attention softmax with an additive mask over shape `[batch, kv_len]`:

```
out[b, i] = softmax(scores[b, :] + mask[b, :])[i]
```

Inputs widened to f32 internally for f16/bf16 variants — output cast back to the original dtype.

### Q4 kernels

Weights are packed as 4-bit values: each `u32` word holds 8 nibbles. A separate
`f16` scale tensor of shape `[groups, out_features]` provides per-group quantization
scale factors. Zero-point is symmetric (value 8).

- `q4_reconstruct` — unpacks a `[k/8, n]` weight tensor to full `f16 [k, n]`.
- `q4_matmul` — fused Q4 × f16 GEMM, shape `[m, k] × [k/8, n] → [m, n]`.
- `column_remap` — reorders columns of a weight matrix using an index tensor.

## Requirements

| Component | Minimum version |
|-----------|----------------|
| CUDA | 13.2 |
| NVIDIA GPU | sm_80 (Ampere or newer) |
| Rust | 1.85 |
| cuTile Rust | 0.3.1 |

Older GPUs (Volta / Turing) are not supported by cuTile Rust.

## Feature flags

The crate compiles in two modes:

| Feature | Description |
|---------|-------------|
| *(none)* | CPU-stub mode — all entry points return `KernelError::NoGpu`. Safe for CI and CPU-only machines. |
| `cuda` | Full GPU mode. Requires CUDA 13.2+ and an sm_80+ GPU. |
| `gpu` | Alias for `cuda`. |

## Building

```bash
# CPU-stub mode (no GPU required)
cargo build -p ember-cutile-kernels

# Full GPU mode
cargo build -p ember-cutile-kernels --features cuda

# Via the unified setup script
./setup.sh             # GPU machine — full build
./setup.sh --cpu-only  # CI / dev machine without GPU
```

## Using the kernels from Rust

```rust
use ember_cutile_kernels::softmax::masked_softmax_f32;

let result = masked_softmax_f32(&stream, &scores, &mask, &mut out);
match result {
    Ok(()) => { /* output is ready on the GPU */ }
    Err(KernelError::NoGpu) => { /* built without --features cuda */ }
    Err(e) => eprintln!("kernel error: {e}"),
}
```

## Error handling

All entry points return `Result<(), KernelError>`.

| Variant | Meaning |
|---------|---------|
| `KernelError::NoGpu` | Compiled without `--features cuda`. |
| `KernelError::Cutile(e)` | cuTile / CUDA runtime error (`cuda` feature only). |
| `KernelError::Shape(msg)` | Tensor shape or rank mismatch. |

## Performance notes

cuTile Rust JIT-compiles kernels at first use and caches the resulting CUBIN.
Warm launch overhead is approximately 1.6 µs per call (cuTile Rust 0.3.1 benchmark figures).
For production deployments, the cache persists across process restarts.

On NVIDIA B200, cuTile Rust benchmarks reach ~91 % of peak memory bandwidth for element-wise
operations and ~92 % of dense f16 peak FLOPS for GEMM.
