use thiserror::Error;

/// Errors returned by GPU kernel entry points.
#[derive(Debug, Error)]
pub enum KernelError {
    /// The binary was compiled without the `cuda` feature flag — no GPU
    /// operations are available.  Re-compile with `--features cuda` (or
    /// `--features gpu`).
    #[error("GPU kernels are disabled: recompile with --features cuda")]
    NoGpu,

    /// The underlying cuTile / CUDA runtime returned an error.
    #[cfg(feature = "cuda")]
    #[error("cuTile runtime error: {0}")]
    Cutile(#[from] cutile::error::Error),

    /// Shape or dimension mismatch between arguments.
    #[error("shape mismatch: {0}")]
    Shape(String),
}
