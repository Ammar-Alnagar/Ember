//! 4-bit quantization kernels, replacing the legacy
//! `q4_matrix.cu` and `q4_matmul.cu` CUDA C++ extensions.
//!
//! # Kernels
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`q4_reconstruct`] | Unpack Q4 weight matrix to f16. |
//! | [`q4_matmul`]      | Q4 × f16 matrix multiply → f16. |
//! | [`column_remap`]   | Reorder columns of a weight matrix. |
//!
//! # Format
//!
//! Weights are stored in packed 4-bit format: each `u32` word holds eight
//! 4-bit values (two per nibble pair), interleaved with `f16` scale factors
//! that live in a separate tensor shaped `[groups, out_features]`.
//!
//! # Feature flags
//!
//! Compile with `--features cuda` to run on a GPU.  Without the flag every
//! entry point returns `Err(KernelError::NoGpu)`.

use crate::error::KernelError;

// ── GPU path ─────────────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
mod gpu {
    use cutile::prelude::*;
    use cutile::api;

    // ── device kernels ────────────────────────────────────────────────────

    #[cutile::module]
    pub mod quant_kernels {
        use cutile::core::*;

        // Tile sizes (tuned for sm_80 Ampere; cuTile will autotune on first run).
        const TILE_M: i32 = 64;
        const TILE_N: i32 = 64;
        const TILE_K: i32 = 32;

        // ── Q4 weight reconstruction ──────────────────────────────────────

        /// Unpack a packed Q4 weight tensor to f16.
        ///
        /// `qweight`  : `[K/8, N]` — packed uint32 (8 values per word)
        /// `scales`   : `[groups, N]` — f16 scale per group
        /// `out`      : `[K, N]` — reconstructed f16 weights
        /// `group_size` (const generic) — number of rows per scale group
        #[cutile::entry()]
        pub fn q4_reconstruct<const N: [i32; 2], const G: i32>(
            out:     &mut Tensor<f16, N>,
            qweight: &Tensor<u32,  { [-1, -1] }>,
            scales:  &Tensor<f16,  { [-1, -1] }>,
        ) {
            let pid_n = program_id(0);   // column tile index
            let pid_k = program_id(1);   // row tile index (over K/8 packed rows)

            // Load a [TILE_K/8, TILE_N] tile of packed weights
            let packed = qweight.load_tile(shape!([{ N[0] / 8 }, { N[1] }]), [pid_k, pid_n]);

            // Unpack each u32 → 8 × f16 via nibble extraction
            let lo4 = bitwise_and(packed, constant(0x0F0F_0F0Fu32, packed.shape()));
            let hi4 = right_shift(packed, constant(4u32, packed.shape()));

            // Interleave lo/hi back into a [TILE_K, TILE_N] tile
            // (cuTile interleave_tiles merges along a new axis)
            let unpacked_int = interleave_tiles(lo4, hi4, 0); // [TILE_K, TILE_N]

            // Load scales for this tile's group
            let group_idx = pid_k / G;
            let scale_tile = scales.load_tile(shape!([1, { N[1] }]), [group_idx, pid_n]);

            // Cast and scale: (val - 8) * scale  (zero-point = 8 for symmetric Q4)
            let zp   = constant(8.0f32, unpacked_int.shape());
            let vals  = cast_tile::<f32>(unpacked_int) - zp;
            let scale = cast_tile::<f32>(scale_tile);
            let result = cast_tile::<f16>(vals * scale);

            out.store(result);
        }

        // ── Q4 × f16 matmul ───────────────────────────────────────────────

        /// Perform `C[M, N] = A[M, K] × B_q4[K, N]`.
        ///
        /// `input`   : `[M, K]` — f16 activation matrix  
        /// `qweight` : `[K/8, N]` — packed Q4 weights  
        /// `scales`  : `[groups, N]` — f16 scale per group  
        /// `out`     : `[M, N]` — f16 result  
        #[cutile::entry()]
        pub fn q4_matmul<
            const OUT_S: [i32; 2],
            const G: i32,
        >(
            out:     &mut Tensor<f16, OUT_S>,
            input:   &Tensor<f16, { [-1, -1] }>,
            qweight: &Tensor<u32, { [-1, -1] }>,
            scales:  &Tensor<f16, { [-1, -1] }>,
        ) {
            let pid_m = program_id(0);
            let pid_n = program_id(1);

            // Accumulator tile
            let mut acc: Tile<f32, { [TILE_M, TILE_N] }> = zeros_tile();

            // K-loop: iterate over packed weight rows
            let k_packed = qweight.shape_dim(0);   // K/8
            let k_tiles   = k_packed / (TILE_K / 8);

            for k in range(0, k_tiles) {
                // Load a tile of activations: [TILE_M, TILE_K]
                let a = cast_tile::<f32>(
                    input.load_tile(shape!([TILE_M, TILE_K]), [pid_m, k * (TILE_K / 8)])
                );

                // Load packed weights: [TILE_K/8, TILE_N]
                let packed = qweight.load_tile(
                    shape!([{ TILE_K / 8 }, TILE_N]),
                    [k, pid_n],
                );

                // Unpack Q4 → f32 (same nibble logic as q4_reconstruct)
                let lo4   = bitwise_and(packed, constant(0x0F0F_0F0Fu32, packed.shape()));
                let hi4   = right_shift(packed, constant(4u32, packed.shape()));
                let b_int = interleave_tiles(lo4, hi4, 0); // [TILE_K, TILE_N]

                let group_idx  = k / (G / TILE_K);
                let scale_tile = scales.load_tile(shape!([1, TILE_N]), [group_idx, pid_n]);
                let scale      = cast_tile::<f32>(scale_tile);
                let zp         = constant(8.0f32, b_int.shape());
                let b          = (cast_tile::<f32>(b_int) - zp) * scale; // [TILE_K, TILE_N]

                // Accumulate: A[TILE_M, TILE_K] × B[TILE_K, TILE_N]
                acc = acc + dot(a, b);
            }

            out.store(cast_tile::<f16>(acc));
        }

        // ── Column remap ──────────────────────────────────────────────────

        /// Reorder columns of `src` according to `indices`.
        ///
        /// `src`     : `[rows, cols]`  
        /// `indices` : `[cols]` — target column index for each source column  
        /// `out`     : `[rows, cols]`  
        #[cutile::entry()]
        pub fn column_remap<const S: [i32; 2]>(
            out:     &mut Tensor<f16, S>,
            src:     &Tensor<f16, { [-1, -1] }>,
            indices: &Tensor<i32, { [-1] }>,
        ) {
            let pid_col  = program_id(0);
            let pid_row  = program_id(1);

            // Read the remapped column index for this tile
            let idx_tile  = indices.load_tile(shape!([{ S[1] }]), [pid_col]);
            let col_tile  = src.gather(idx_tile, 1, shape!(S), [pid_row, pid_col]);
            out.store(col_tile);
        }
    }

    // ── host-side launchers ───────────────────────────────────────────────

    use quant_kernels::*;
    use cutile::tensor::Tensor;
    use cutile::cuda_core::f16;
    use cutile::cuda_async::Stream;

    /// Unpack packed Q4 weights to f16.
    ///
    /// * `qweight`  — shape `[k/8, n]`, dtype `u32`
    /// * `scales`   — shape `[groups, n]`, dtype `f16`
    /// * `out`      — shape `[k, n]`, dtype `f16`; written in-place
    /// * `group_size` — rows per scale group (must divide `k`)
    pub fn launch_reconstruct(
        stream:     &Stream,
        qweight:    &Tensor<u32>,
        scales:     &Tensor<f16>,
        out:        &mut Tensor<f16>,
        group_size: usize,
    ) -> Result<(), super::super::KernelError> {
        let qs = qweight.shape();
        let os = out.shape();
        if qs.len() != 2 || os.len() != 2 {
            return Err(super::super::KernelError::Shape(
                "qweight and out must be 2D".into(),
            ));
        }
        let k = qs[0] * 8;
        let n = qs[1];
        if os[0] != k || os[1] != n {
            return Err(super::super::KernelError::Shape(format!(
                "out shape {os:?} != expected [{k}, {n}]"
            )));
        }
        q4_reconstruct::<{ [k as i32, n as i32] }, { group_size as i32 }>(
            out.partition([k as i32, n as i32]),
            qweight,
            scales,
        )
        .sync_on(stream)
        .map(|_| ())
        .map_err(cutile::error::Error::from)
        .map_err(super::super::KernelError::Cutile)
    }

    /// Q4 × f16 matrix multiply.
    ///
    /// * `input`    — shape `[m, k]`, dtype `f16`
    /// * `qweight`  — shape `[k/8, n]`, dtype `u32`
    /// * `scales`   — shape `[groups, n]`, dtype `f16`
    /// * `out`      — shape `[m, n]`, dtype `f16`; written in-place
    /// * `group_size` — rows per scale group
    pub fn launch_matmul(
        stream:     &Stream,
        input:      &Tensor<f16>,
        qweight:    &Tensor<u32>,
        scales:     &Tensor<f16>,
        out:        &mut Tensor<f16>,
        group_size: usize,
    ) -> Result<(), super::super::KernelError> {
        let is = input.shape();
        let os = out.shape();
        if is.len() != 2 || os.len() != 2 {
            return Err(super::super::KernelError::Shape(
                "input and out must be 2D".into(),
            ));
        }
        let m = is[0];
        let n = os[1];
        q4_matmul::<{ [m as i32, n as i32] }, { group_size as i32 }>(
            out.partition([m as i32, n as i32]),
            input,
            qweight,
            scales,
        )
        .sync_on(stream)
        .map(|_| ())
        .map_err(cutile::error::Error::from)
        .map_err(super::super::KernelError::Cutile)
    }

    /// Reorder columns of a f16 matrix.
    ///
    /// * `src`     — shape `[rows, cols]`, dtype `f16`
    /// * `indices` — shape `[cols]`, dtype `i32`
    /// * `out`     — shape `[rows, cols]`, dtype `f16`; written in-place
    pub fn launch_column_remap(
        stream:  &Stream,
        src:     &Tensor<f16>,
        indices: &Tensor<i32>,
        out:     &mut Tensor<f16>,
    ) -> Result<(), super::super::KernelError> {
        let ss = src.shape();
        if ss.len() != 2 {
            return Err(super::super::KernelError::Shape("src must be 2D".into()));
        }
        let rows = ss[0];
        let cols = ss[1];
        column_remap::<{ [rows as i32, cols as i32] }>(
            out.partition([rows as i32, cols as i32]),
            src,
            indices,
        )
        .sync_on(stream)
        .map(|_| ())
        .map_err(cutile::error::Error::from)
        .map_err(super::super::KernelError::Cutile)
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Unpack a packed Q4 weight matrix to f16.
///
/// See [module-level docs](self) for tensor shapes.
#[allow(unused_variables)]
pub fn q4_reconstruct(
    stream:     &crate::Stream,
    qweight:    &crate::Tensor<u32>,
    scales:     &crate::Tensor<crate::F16>,
    out:        &mut crate::Tensor<crate::F16>,
    group_size: usize,
) -> Result<(), KernelError> {
    #[cfg(feature = "cuda")]
    {
        gpu::launch_reconstruct(stream, qweight, scales, out, group_size)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(KernelError::NoGpu)
    }
}

/// Q4 × f16 matrix multiply.
///
/// See [module-level docs](self) for tensor shapes.
#[allow(unused_variables)]
pub fn q4_matmul(
    stream:     &crate::Stream,
    input:      &crate::Tensor<crate::F16>,
    qweight:    &crate::Tensor<u32>,
    scales:     &crate::Tensor<crate::F16>,
    out:        &mut crate::Tensor<crate::F16>,
    group_size: usize,
) -> Result<(), KernelError> {
    #[cfg(feature = "cuda")]
    {
        gpu::launch_matmul(stream, input, qweight, scales, out, group_size)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(KernelError::NoGpu)
    }
}

/// Reorder columns of a f16 matrix.
///
/// See [module-level docs](self) for tensor shapes.
#[allow(unused_variables)]
pub fn column_remap(
    stream:  &crate::Stream,
    src:     &crate::Tensor<crate::F16>,
    indices: &crate::Tensor<i32>,
    out:     &mut crate::Tensor<crate::F16>,
) -> Result<(), KernelError> {
    #[cfg(feature = "cuda")]
    {
        gpu::launch_column_remap(stream, src, indices, out)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(KernelError::NoGpu)
    }
}
