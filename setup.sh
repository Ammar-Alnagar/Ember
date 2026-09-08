#!/usr/bin/env bash
# =============================================================================
# Ember TGI — unified setup & build script
#
# Usage:
#   ./setup.sh [OPTIONS]
#
# Options:
#   --help           Show this message and exit
#   --cpu-only       Skip GPU/CUDA kernel compilation (no CUDA required)
#   --no-server      Skip Python server installation
#   --no-router      Skip Rust router/launcher build
#   --dev            Build in debug mode instead of release
#   --cuda-arch LIST CUDA arch list, e.g. "8.0;8.6;9.0+PTX"  (default: auto)
#   --python VER     Python version to use             (default: 3.11)
#
# What this script does
# ---------------------
# 1.  Check system dependencies (Rust, protoc, uv / Python, optionally CUDA)
# 2.  Build the Rust workspace (router + launcher + benchmark + kernels)
# 3.  Install the Python server via uv
# 4.  Download pre-built kernel artifacts (kernels download .)
#
# Run on a GPU machine without flags for a full GPU-enabled build.
# Run with --cpu-only for CI / dev workstations without a GPU.
# =============================================================================

set -euo pipefail
IFS=$'\n\t'

# ── defaults ──────────────────────────────────────────────────────────────────
CPU_ONLY=false
NO_SERVER=false
NO_ROUTER=false
DEV_BUILD=false
CUDA_ARCH="auto"
PYTHON_VERSION="3.11"

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)
            sed -n '/^# Usage/,/^# ===/{s/^# \{0,3\}//; p}' "$0"
            exit 0
            ;;
        --cpu-only)  CPU_ONLY=true ;;
        --no-server) NO_SERVER=true ;;
        --no-router) NO_ROUTER=true ;;
        --dev)       DEV_BUILD=true ;;
        --cuda-arch)
            [[ $# -ge 2 ]] || { echo "Error: --cuda-arch requires a value" >&2; exit 1; }
            CUDA_ARCH="$2"
            shift
            ;;
        --python)
            [[ $# -ge 2 ]] || { echo "Error: --python requires a value" >&2; exit 1; }
            PYTHON_VERSION="$2"
            shift
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

# ── helpers ───────────────────────────────────────────────────────────────────
info()  { echo -e "\033[1;32m[setup]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[warn]\033[0m $*" >&2; }
error() { echo -e "\033[1;31m[error]\033[0m $*" >&2; exit 1; }

require_cmd() {
    command -v "$1" &>/dev/null || error "'$1' is required but not found. $2"
}

# ── dependency checks ─────────────────────────────────────────────────────────
info "Checking dependencies…"

require_cmd cargo  "Install Rust: https://rustup.rs"
require_cmd protoc "Install protobuf: sudo apt install -y protobuf-compiler  (or brew install protobuf)"

RUST_MIN="1.85.0"
RUST_VER=$(rustc --version | awk '{print $2}')
if [[ "$(printf '%s\n' "$RUST_MIN" "$RUST_VER" | sort -V | head -1)" != "$RUST_MIN" ]]; then
    error "Rust >= $RUST_MIN required, found $RUST_VER. Run: rustup update stable"
fi

if ! $CPU_ONLY; then
    if ! command -v nvcc &>/dev/null && ! command -v nvidia-smi &>/dev/null; then
        warn "CUDA not detected — falling back to --cpu-only build."
        warn "Pass --cpu-only explicitly to silence this warning."
        CPU_ONLY=true
    fi
fi

if ! $NO_SERVER; then
    if ! command -v uv &>/dev/null; then
        info "uv not found — installing via the official installer…"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

# ── Rust build ────────────────────────────────────────────────────────────────
if ! $NO_ROUTER; then
    # Build cargo args as an array so each token is a separate argument.
    # This avoids word-splitting bugs when passing flags like --features cuda.
    CARGO_ARGS=(build)

    if $DEV_BUILD; then
        CARGO_ARGS+=(--profile dev)
    else
        CARGO_ARGS+=(--profile release-opt)
    fi

    if ! $CPU_ONLY; then
        CARGO_ARGS+=(--features cuda)
        if [[ "$CUDA_ARCH" != "auto" ]]; then
            export CUDA_ARCH_LIST="$CUDA_ARCH"
            info "Targeting CUDA architectures: $CUDA_ARCH_LIST"
        fi
    fi

    info "Building Rust workspace (${CARGO_ARGS[*]})…"
    cargo "${CARGO_ARGS[@]}"

    # Tests use the same feature set
    TEST_ARGS=(test)
    if ! $CPU_ONLY; then
        TEST_ARGS+=(--features cuda)
    fi

    info "Running Rust tests…"
    cargo "${TEST_ARGS[@]}"
fi

# ── Python server ─────────────────────────────────────────────────────────────
if ! $NO_SERVER; then
    info "Installing Python ${PYTHON_VERSION} server…"
    uv python install "${PYTHON_VERSION}"

    # Build uv sync extras as an array for the same reason
    EXTRAS=(
        --extra gen
        --extra bnb
        --extra accelerate
        --extra compressed-tensors
        --extra quantize
        --extra peft
        --extra outlines
        --extra torch
    )

    cd server

    if ! $CPU_ONLY; then
        info "Downloading pre-built GPU kernel artifacts…"
        uv sync --frozen "${EXTRAS[@]}" --no-install-project --active
        make gen-server-raw
        kernels download . || warn "kernels download failed — GPU kernels may not be available."
    fi

    uv sync --frozen "${EXTRAS[@]}" --active --python="${PYTHON_VERSION}"
    uv pip install "nvidia-nccl-cu12==2.25.1" 2>/dev/null || true

    cd ..
fi

# ── done ──────────────────────────────────────────────────────────────────────
info "Build complete."
info ""
info "Quick-start:"
info "  text-generation-launcher --model-id <model-id> --port 8080"
info ""
if $CPU_ONLY; then
    info "CPU-only build — GPU kernels return KernelError::NoGpu at runtime."
    info "Re-run without --cpu-only on a CUDA-capable machine for GPU support."
fi
