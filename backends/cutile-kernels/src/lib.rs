//! Ember GPU kernels implemented with [cuTile Rust](https://github.com/NVlabs/cutile-rs)
//! from NVlabs, replacing the legacy PyTorch-CUDA C++ extension kernels.
//!
//! # Kernels
//!
//! | Kernel | Replaces |
//! |--------|----------|
//! | [`softmax::masked_softmax_f32`]  | `fused_attention_cuda.cu` |
//! | [`softmax::masked_softmax_f16`]  | `fused_attention_cuda.cu` |
//! | [`softmax::masked_softmax_bf16`] | `fused_bloom_attention_cuda.cu` |
//! | [`quant::q4_reconstruct`]        | `q4_matrix.cu` |
//! | [`quant::q4_matmul`]             | `q4_matmul.cu` |
//! | [`quant::column_remap`]          | column remap utility |
//!
//! # Feature flags
//!
//! * `cuda` — compile and run kernels on a real GPU (requires CUDA 13.2+,
//!   sm_80+, Rust 1.85+).  Without this flag every entry point returns
//!   `Err(KernelError::NoGpu)` — safe for CPU-only CI builds.
//! * `gpu`  — alias for `cuda`.
//!
//! # Example
//!
//! ```rust,ignore
//! use ember_cutile_kernels::softmax::masked_softmax_f32;
//!
//! // scores: [batch, kv_len]  mask: [batch, kv_len]  probs: [batch, kv_len]
//! masked_softmax_f32(&stream, &scores, &mask, &mut probs)?;
//! ```

pub mod error;
pub mod quant;
pub mod softmax;

pub use error::KernelError;

// ── Type aliases so the kernel modules can use bare names ─────────────────────

/// Opaque stream type — maps to `cutile::cuda_async::Stream` with the `cuda`
/// feature, or to a zero-size stub for CPU-only builds.
#[cfg(feature = "cuda")]
pub use cutile::cuda_async::Stream;

#[cfg(not(feature = "cuda"))]
pub struct Stream;

/// Device tensor — maps to `cutile::tensor::Tensor<T>` with the `cuda`
/// feature, or to a zero-size stub for CPU-only builds.
#[cfg(feature = "cuda")]
pub use cutile::tensor::Tensor;

#[cfg(not(feature = "cuda"))]
pub struct Tensor<T>(std::marker::PhantomData<T>);

// ── Scalar type aliases ───────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
pub use cutile::cuda_core::f16 as F16;
#[cfg(feature = "cuda")]
pub use cutile::cuda_core::bf16 as BF16;

/// Stub half-precision type used when the `cuda` feature is disabled.
#[cfg(not(feature = "cuda"))]
#[derive(Debug, Clone, Copy, Default)]
#[repr(transparent)]
pub struct F16(u16);

/// Stub bfloat16 type used when the `cuda` feature is disabled.
#[cfg(not(feature = "cuda"))]
#[derive(Debug, Clone, Copy, Default)]
#[repr(transparent)]
pub struct BF16(u16);
