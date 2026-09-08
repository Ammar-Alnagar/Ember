# Installation from source

> **Tip:** The recommended way to run Ember is via Docker. See [Quick Tour](./quicktour),
> [Nvidia GPU install](./installation_nvidia), or [AMD GPU install](./installation_amd).
> Only follow the steps below if you need a bare-metal install.

## One-command setup

Clone the repository and run the unified setup script:

```bash
git clone https://github.com/huggingface/ember.git
cd ember
./setup.sh          # GPU machine (requires CUDA 13.2+, sm_80+)
./setup.sh --cpu-only --no-server   # CI or laptop without a GPU
```

The script checks all prerequisites, builds the Rust workspace, installs the Python server,
and downloads pre-built GPU kernel artifacts in one step. Run `./setup.sh --help` for all options.

---

## Manual installation

If you prefer step-by-step control, follow the sections below.

### 1 — System dependencies

**Rust (≥ 1.85)**

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**protoc (Protocol Buffer compiler)**

Required by the Rust gRPC build scripts. Install the official binary to get a version that
supports proto3 optional fields:

```bash
# Linux
PROTOC_VERSION=25.3
curl -fOL "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-x86_64.zip" \
  -o protoc.zip
sudo unzip -o protoc.zip -d /usr/local bin/protoc
sudo unzip -o protoc.zip -d /usr/local 'include/*'
rm protoc.zip

# macOS (Homebrew)
brew install protobuf
```

**Python (≥ 3.9) + uv**

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
```

**CUDA (GPU builds only)**

CUDA 13.2 or newer is required for the cuTile Rust kernels. Check your version:

```bash
nvcc --version
```

For older CUDA installations see [Nvidia GPU install](./installation_nvidia).

### 2 — Clone the repository

```bash
git clone https://github.com/huggingface/ember.git
cd ember
```

### 3 — Build the Rust workspace

```bash
# CPU-only (no CUDA needed)
cargo build --profile release-opt

# With GPU kernels (requires CUDA 13.2+)
cargo build --profile release-opt --features cuda
```

This builds the router, launcher, benchmark, and the
[cuTile kernel crate](./conceptual/cutile_kernels) in one step.

### 4 — Install the Python server

```bash
cd server
uv sync --frozen \
  --extra gen --extra bnb --extra accelerate \
  --extra compressed-tensors --extra quantize \
  --extra peft --extra outlines --extra torch \
  --active --python=3.11
make gen-server-raw
kernels download .   # downloads pre-built GPU kernel artifacts
cd ..
```

### 5 — Verify

```bash
text-generation-launcher --help
text-generation-server --help
```

---

## Make targets

| Target | Description |
|--------|-------------|
| `make setup` | Full GPU build + Python server |
| `make setup-cpu` | CPU-only build, no server |
| `make setup-dev` | CPU-only, debug mode |
| `make install` | Server + router + launcher (GPU) |
| `make install-cpu` | Server + router + launcher (CPU) |
| `make build-kernels` | cuTile kernel crate only (GPU) |
| `make build-kernels-cpu` | cuTile kernel crate stub (no GPU needed) |
| `make rust-tests` | Rust unit tests |
| `make lint` | `cargo clippy` |

---

## Optional dependencies

**OpenSSL + gcc** — needed on some Linux machines:

```bash
sudo apt-get install libssl-dev gcc -y
```

**Nix** — see [Local install (Nix)](../README.md#local-install-nix) in the README for a
fully reproducible environment.
