# =============================================================================
# Ember TGI — Makefile
#
# Quick-start (GPU machine):
#   make setup                  # full GPU build + Python server
#
# Quick-start (CPU-only / CI):
#   make setup-cpu              # build without CUDA, no server
#
# Individual targets:
#   make install-server         # Python server only
#   make install-router         # Rust router/launcher (GPU build)
#   make install-router-cpu     # Rust router/launcher (no GPU)
#   make install-launcher       # launcher binary
#   make install-benchmark      # benchmark binary
#   make install                # everything (GPU)
#   make install-cpu            # everything (CPU-only)
#   make rust-tests             # run Rust unit tests
#   make lint                   # cargo clippy
# =============================================================================

# ── Unified setup ─────────────────────────────────────────────────────────────

## Full GPU setup: build Rust workspace + Python server
setup:
	bash setup.sh

## CPU-only setup: no CUDA required, no Python server
setup-cpu:
	bash setup.sh --cpu-only --no-server

## Developer convenience: CPU-only, debug mode, no server
setup-dev:
	bash setup.sh --cpu-only --no-server --dev

# ── Python server ─────────────────────────────────────────────────────────────

install-server:
	cd server && make install

install-server-cpu:
	cd server && make install-server

# ── Rust binaries ─────────────────────────────────────────────────────────────

install-router:
	cargo install --path backends/v3/

install-router-cpu:
	cargo install --path backends/v3/ --no-default-features

install-launcher:
	cargo install --path launcher/

install-benchmark:
	cargo install --path benchmark/

# ── Combined installs ──────────────────────────────────────────────────────────

install: install-server install-router install-launcher

install-cpu: install-server-cpu install-router-cpu install-launcher

# ── GPU kernels (cuTile-rs) ───────────────────────────────────────────────────

## Build only the cuTile Rust kernels crate with GPU support
build-kernels:
	cargo build --profile release-opt -p ember-cutile-kernels --features cuda

## Build kernels in CPU-stub mode (no CUDA required)
build-kernels-cpu:
	cargo build -p ember-cutile-kernels

# ── Testing ───────────────────────────────────────────────────────────────────

rust-tests: install-router install-launcher
	cargo test

rust-tests-cpu:
	cargo test

python-server-tests:
	HF_HUB_ENABLE_HF_TRANSFER=1 pytest -s -vv -m "not private" server/tests

python-client-tests:
	pytest clients/python/tests

python-tests: python-server-tests python-client-tests

install-integration-tests:
	cd integration-tests && pip install -r requirements.txt
	cd clients/python && pip install .

integration-tests: install-integration-tests
	pytest -s -vv -m "not private" integration-tests

update-integration-tests: install-integration-tests
	pytest -s -vv --snapshot-update integration-tests

lint:
	cargo clippy --all-targets -- -D warnings

# ── Dev server ────────────────────────────────────────────────────────────────

server-dev:
	cd server && make run-dev

router-dev:
	cd backends/v3 && cargo run -- --port 8080

# ── Convenience launchers ─────────────────────────────────────────────────────

run-falcon-7b-instruct:
	text-generation-launcher --model-id tiiuae/falcon-7b-instruct --port 8080

run-falcon-7b-instruct-quantize:
	text-generation-launcher --model-id tiiuae/falcon-7b-instruct --quantize bitsandbytes --port 8080

# ── Docs ──────────────────────────────────────────────────────────────────────

preview_doc:
	doc-builder preview text-generation-inference docs/source --not_python_module

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	rm -rf target aml

.PHONY: setup setup-cpu setup-dev \
        install-server install-server-cpu \
        install-router install-router-cpu install-launcher install-benchmark \
        install install-cpu \
        build-kernels build-kernels-cpu \
        rust-tests rust-tests-cpu \
        python-server-tests python-client-tests python-tests \
        install-integration-tests integration-tests update-integration-tests \
        lint server-dev router-dev \
        run-falcon-7b-instruct run-falcon-7b-instruct-quantize \
        preview_doc clean
