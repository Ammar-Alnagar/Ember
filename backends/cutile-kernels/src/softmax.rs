//! Masked softmax kernels, replacing the legacy
//! `fused_attention_cuda.cu` and `fused_bloom_attention_cuda.cu`
//! CUDA C++ extensions.
//!
//! # Semantics
//!
//! `masked_softmax(scores, mask, out)` computes:
//!
//! ```text
//! out[b, i] = softmax( scores[b, :] + mask[b, :] )[i]
//! ```
//!
//! where `mask` stores additive biases (−∞ for positions that should be
//! masked, 0 for positions that should not).  The three tensors must have
//! identical shapes `[batch, kv_len]`.
//!
//! # Feature flags
//!
//! Compile with `--features cuda` (or `--features gpu`) to run on a real GPU.
//! Without the feature every entry point returns `Err(KernelError::NoGpu)`.

use crate::error::KernelError;

// ── GPU path ─────────────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
mod gpu {
    use cutile::prelude::*;
    use cutile::api;

    // ── device-side kernels ───────────────────────────────────────────────

    #[cutile::module]
    pub mod softmax_kernels {
        use cutile::core::*;

        // Helper: numerically stable tile-wise softmax over axis 1 (kv dimension).
        // `S = [TILE_BATCH, TILE_KV]`
        fn tile_softmax<const S: [i32; 2]>(
            tile: Tile<f32, S>,
        ) -> Tile<f32, S> {
            // max for numerical stability
            let m = max_tile(tile, [1]);          // shape [TILE_BATCH, 1]
            let shifted = tile - m;
            let e = exp_tile(shifted);
            let s = sum_tile(e, [1]);              // shape [TILE_BATCH, 1]
            e / s
        }

        /// Masked softmax — f32
        #[cutile::entry()]
        pub fn masked_softmax_f32<const S: [i32; 2]>(
            out:    &mut Tensor<f32, S>,
            scores: &Tensor<f32, { [-1, -1] }>,
            mask:   &Tensor<f32, { [-1, -1] }>,
        ) {
            let pid = program_id(0);
            let s = scores.load_tile(shape!(S), [pid, 0]);
            let m = mask.load_tile(shape!(S), [pid, 0]);
            out.store(tile_softmax(s + m));
        }

        /// Masked softmax — f16  (inputs widened to f32 internally)
        #[cutile::entry()]
        pub fn masked_softmax_f16<const S: [i32; 2]>(
            out:    &mut Tensor<f16, S>,
            scores: &Tensor<f16, { [-1, -1] }>,
            mask:   &Tensor<f16, { [-1, -1] }>,
        ) {
            let pid = program_id(0);
            let s = cast_tile::<f32>(scores.load_tile(shape!(S), [pid, 0]));
            let m = cast_tile::<f32>(mask.load_tile(shape!(S), [pid, 0]));
            let r = tile_softmax(s + m);
            out.store(cast_tile::<f16>(r));
        }

        /// Masked softmax — bf16 (inputs widened to f32 internally)
        #[cutile::entry()]
        pub fn masked_softmax_bf16<const S: [i32; 2]>(
            out:    &mut Tensor<bf16, S>,
            scores: &Tensor<bf16, { [-1, -1] }>,
            mask:   &Tensor<bf16, { [-1, -1] }>,
        ) {
            let pid = program_id(0);
            let s = cast_tile::<f32>(scores.load_tile(shape!(S), [pid, 0]));
            let m = cast_tile::<f32>(mask.load_tile(shape!(S), [pid, 0]));
            let r = tile_softmax(s + m);
            out.store(cast_tile::<bf16>(r));
        }
    }

    // ── host-side launchers ───────────────────────────────────────────────

    use softmax_kernels::*;

    /// Tile shape used for the batch dimension.
    const BATCH_TILE: usize = 1;

    /// Launch masked softmax for f32 tensors.
    ///
    /// All three tensors must have shape `[batch, kv_len]` and reside on the
    /// same device.  `out` is written in-place.
    pub fn launch_f32(
        stream:  &cutile::cuda_async::Stream,
        scores:  &cutile::tensor::Tensor<f32>,
        mask:    &cutile::tensor::Tensor<f32>,
        out:     &mut cutile::tensor::Tensor<f32>,
    ) -> Result<(), super::super::KernelError> {
        let [batch, kv_len] = check_shapes(scores, mask, out)?;
        masked_softmax_f32(
            out.partition([BATCH_TILE as i32, kv_len as i32]),
            scores,
            mask,
        )
        .sync_on(stream)
        .map(|_| ())
        .map_err(cutile::error::Error::from)
        .map_err(super::super::KernelError::Cutile)
    }

    /// Launch masked softmax for f16 tensors.
    pub fn launch_f16(
        stream:  &cutile::cuda_async::Stream,
        scores:  &cutile::tensor::Tensor<cutile::cuda_core::f16>,
        mask:    &cutile::tensor::Tensor<cutile::cuda_core::f16>,
        out:     &mut cutile::tensor::Tensor<cutile::cuda_core::f16>,
    ) -> Result<(), super::super::KernelError> {
        let [batch, kv_len] = check_shapes_typed(scores, mask, out)?;
        masked_softmax_f16(
            out.partition([BATCH_TILE as i32, kv_len as i32]),
            scores,
            mask,
        )
        .sync_on(stream)
        .map(|_| ())
        .map_err(cutile::error::Error::from)
        .map_err(super::super::KernelError::Cutile)
    }

    /// Launch masked softmax for bf16 tensors.
    pub fn launch_bf16(
        stream:  &cutile::cuda_async::Stream,
        scores:  &cutile::tensor::Tensor<cutile::cuda_core::bf16>,
        mask:    &cutile::tensor::Tensor<cutile::cuda_core::bf16>,
        out:     &mut cutile::tensor::Tensor<cutile::cuda_core::bf16>,
    ) -> Result<(), super::super::KernelError> {
        let [batch, kv_len] = check_shapes_typed(scores, mask, out)?;
        masked_softmax_bf16(
            out.partition([BATCH_TILE as i32, kv_len as i32]),
            scores,
            mask,
        )
        .sync_on(stream)
        .map(|_| ())
        .map_err(cutile::error::Error::from)
        .map_err(super::super::KernelError::Cutile)
    }

    // ── shape helpers ─────────────────────────────────────────────────────

    fn check_shapes(
        scores: &cutile::tensor::Tensor<f32>,
        mask:   &cutile::tensor::Tensor<f32>,
        out:    &cutile::tensor::Tensor<f32>,
    ) -> Result<[usize; 2], super::super::KernelError> {
        let s = scores.shape();
        if s.len() != 2 {
            return Err(super::super::KernelError::Shape(format!(
                "expected 2D scores, got {:?}",
                s
            )));
        }
        if mask.shape() != s || out.shape() != s {
            return Err(super::super::KernelError::Shape(format!(
                "shape mismatch: scores={s:?} mask={:?} out={:?}",
                mask.shape(),
                out.shape()
            )));
        }
        Ok([s[0], s[1]])
    }

    fn check_shapes_typed<T>(
        scores: &cutile::tensor::Tensor<T>,
        mask:   &cutile::tensor::Tensor<T>,
        out:    &cutile::tensor::Tensor<T>,
    ) -> Result<[usize; 2], super::super::KernelError> {
        let s = scores.shape();
        if s.len() != 2 {
            return Err(super::super::KernelError::Shape(format!(
                "expected 2D scores, got {:?}",
                s
            )));
        }
        if mask.shape() != s || out.shape() != s {
            return Err(super::super::KernelError::Shape(format!(
                "shape mismatch: scores={s:?} mask={:?} out={:?}",
                mask.shape(),
                out.shape()
            )));
        }
        Ok([s[0], s[1]])
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute masked softmax over f32 scores.
///
/// `scores` and `mask` must have shape `[batch, kv_len]`.
/// Results are written into `out`, which must have the same shape.
///
/// On success returns `Ok(())`.  Returns `Err(KernelError::NoGpu)` when
/// compiled without `--features cuda`.
#[allow(unused_variables)]
pub fn masked_softmax_f32(
    stream: &crate::Stream,
    scores: &crate::Tensor<f32>,
    mask: &crate::Tensor<f32>,
    out: &mut crate::Tensor<f32>,
) -> Result<(), KernelError> {
    #[cfg(feature = "cuda")]
    {
        gpu::launch_f32(stream, scores, mask, out)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(KernelError::NoGpu)
    }
}

/// Compute masked softmax over f16 scores.
///
/// See [`masked_softmax_f32`] for semantics.
#[allow(unused_variables)]
pub fn masked_softmax_f16(
    stream: &crate::Stream,
    scores: &crate::Tensor<crate::F16>,
    mask: &crate::Tensor<crate::F16>,
    out: &mut crate::Tensor<crate::F16>,
) -> Result<(), KernelError> {
    #[cfg(feature = "cuda")]
    {
        gpu::launch_f16(stream, scores, mask, out)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(KernelError::NoGpu)
    }
}

/// Compute masked softmax over bf16 scores.
///
/// See [`masked_softmax_f32`] for semantics.
#[allow(unused_variables)]
pub fn masked_softmax_bf16(
    stream: &crate::Stream,
    scores: &crate::Tensor<crate::BF16>,
    mask: &crate::Tensor<crate::BF16>,
    out: &mut crate::Tensor<crate::BF16>,
) -> Result<(), KernelError> {
    #[cfg(feature = "cuda")]
    {
        gpu::launch_bf16(stream, scores, mask, out)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(KernelError::NoGpu)
    }
}
