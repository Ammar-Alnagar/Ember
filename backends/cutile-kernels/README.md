# ember-cutile-kernels

GPU kernels for Ember TGI, implemented with [cuTile Rust](https://github.com/NVlabs/cutile-rs)
from NVlabs. This crate replaces all legacy CUDA C++ extension modules.

## Kernels

| Function | File | Operation |
|----------|------|-----------|
| `softmax::masked_softmax_f32` | `src/softmax.rs` | Attention softmax with additive mask, f32 |
| `softmax::masked_softmax_f16` | `src/softmax.rs` | Same, f16 (widened to f32 internally) |
| `softmax::masked_softmax_bf16` | `src/softmax.rs` | Same, bf16 |
| `quant::q4_reconstruct` | `src/quant.rs` | Unpack Q4 weights → f16 |
| `quant::q4_matmul` | `src/quant.rs` | Fused Q4 × f16 GEMM |
| `quant::column_remap` | `src/quant.rs` | Reorder weight matrix columns |

## Requirements

| | Minimum |
|-|---------|
| Rust | 1.85 |
| CUDA | 13.2 |
| GPU | sm_80 (Ampere+) |
| cuTile Rust | 0.3.1 |

## Feature flags

| Feature | Behaviour |
|---------|-----------|
| *(none)* | CPU-stub mode. All entry points return `Err(KernelError::NoGpu)`. Compiles on any machine — used in CI. |
| `cuda` | Full GPU mode. Requires CUDA 13.2+ and an sm_80+ device. |
| `gpu` | Alias for `cuda`. |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
ember-cutile-kernels = { path = "../backends/cutile-kernels" }

[features]
cuda = ["ember-cutile-kernels/cuda"]
```

```rust
use ember_cutile_kernels::{
    softmax::masked_softmax_f32,
    quant::{q4_matmul, q4_reconstruct},
    KernelError,
};

// Masked softmax: scores and mask are [batch, kv_len]
let result = masked_softmax_f32(&stream, &scores, &mask, &mut out);

// Q4 matmul: input [m, k], qweight [k/8, n], out [m, n]
let result = q4_matmul(&stream, &input, &qweight, &scales, &mut out, group_size);
```

## Error handling

```rust
match masked_softmax_f32(&stream, &scores, &mask, &mut out) {
    Ok(()) => { /* output ready on GPU */ }
    Err(KernelError::NoGpu)       => { /* no cuda feature */ }
    Err(KernelError::Shape(msg))  => { /* shape mismatch */ }
    Err(KernelError::Cutile(e))   => { /* CUDA runtime error */ }
}
```

## Building

```bash
# CPU-stub (no CUDA needed, fast CI check)
cargo build -p ember-cutile-kernels

# Full GPU build
cargo build -p ember-cutile-kernels --features cuda --profile release-opt
```

## Testing

```bash
# CPU-stub tests (always pass without GPU)
cargo test -p ember-cutile-kernels

# GPU tests (requires sm_80+ GPU and CUDA 13.2+)
cargo test -p ember-cutile-kernels --features cuda
```

## Module layout

```
backends/cutile-kernels/
├── Cargo.toml          dependency: cutile = "0.3.1" (optional, cuda feature)
└── src/
    ├── lib.rs          public API re-exports + type aliases
    ├── error.rs        KernelError enum
    ├── softmax.rs      masked_softmax_f32/f16/bf16
    └── quant.rs        q4_reconstruct, q4_matmul, column_remap
```
