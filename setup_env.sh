#!/usr/bin/env bash
#
# setup_env.sh — one-click environment setup for qgf on RTX 5090 / Blackwell (sm_120).
#
# Runs the full JAX 0.6.2 migration/fix end to end:
#   1. preflight   — check `uv` is present and the NVIDIA driver is R570+ (Blackwell needs it)
#   2. uv lock     — resolve deps from pyproject.toml (jax[cuda12]==0.6.2, flax==0.10.2)
#   3. uv sync     — install into .venv (UV_LINK_MODE=copy so the in-place distrax patch is safe)
#   4. patch distrax — shim jax.core.Var/Literal -> jax.extend.core (required on jax>=0.6)
#   5. uv export   — regenerate requirements.txt from the fresh lock
#   6. smoke test  — import jax/flax/distrax, list devices, import utils.networks
#
# Usage:
#   bash setup_env.sh               # full setup (recommended on a fresh checkout)
#   bash setup_env.sh --patch-only  # ONLY re-run the distrax patch (after a distrax reinstall)
#   bash setup_env.sh --skip-lock   # reuse the existing uv.lock (skip re-resolve)
#   bash setup_env.sh --no-smoke    # skip the smoke test
#   bash setup_env.sh -h
#
# NOTE: the distrax patch edits site-packages, which does NOT survive a reinstall — re-run
# `bash setup_env.sh --patch-only` after any `uv sync`/`pip install` that reinstalls distrax.

set -euo pipefail

MIN_DRIVER=570
PATCH_ONLY=0
DO_SMOKE=1
DO_LOCK=1

usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next} {exit}' "$0"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --patch-only) PATCH_ONLY=1; shift ;;
    --no-smoke)   DO_SMOKE=0; shift ;;
    --skip-lock)  DO_LOCK=0; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

# Run from the repo root (this script lives there), so pyproject.toml / scripts/ resolve.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

step() { echo; echo "==> $*"; }

require_uv() {
  command -v uv >/dev/null 2>&1 || {
    echo "Error: 'uv' not found. Install it (https://docs.astral.sh/uv/) or use the pip path in README.md." >&2
    exit 1
  }
}

check_driver() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Warning: nvidia-smi not found; cannot verify GPU driver. Blackwell/sm_120 needs R${MIN_DRIVER}+." >&2
    return 0
  fi
  local ver major
  ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')" || ver=""
  major="${ver%%.*}"
  if [[ -z "$major" || ! "$major" =~ ^[0-9]+$ ]]; then
    echo "Warning: could not parse driver version ('$ver'); make sure it is R${MIN_DRIVER}+." >&2
  elif (( major < MIN_DRIVER )); then
    echo "Warning: driver $ver < R${MIN_DRIVER}. RTX 5090 (sm_120) needs R${MIN_DRIVER}+ / CUDA 12.8; XLA may fail to compile." >&2
  else
    echo "GPU driver $ver OK (>= R${MIN_DRIVER})."
  fi
}

patch_distrax() {
  step "Patching distrax for jax>=0.6 (jax.core.Var/Literal -> jax.extend.core)"
  uv run bash scripts/patch_distrax_jax060.sh
}

smoke_test() {
  step "Smoke test (jax/flax/distrax import, devices, utils.networks)"
  uv run python - <<'PY'
import jax, flax, distrax
print("jax", jax.__version__, "| flax", flax.__version__, "| distrax", distrax.__version__)
print("devices:", jax.devices())
import utils.networks  # pulls distrax + the nn.vmap ensemblize actor -> triggers a real trace
print("import utils.networks OK")
PY
  echo "Smoke test passed."
}

require_uv

# --patch-only: skip install, just re-apply the distrax shim (after a reinstall).
if [[ "$PATCH_ONLY" == 1 ]]; then
  patch_distrax
  echo; echo "Done (patch-only)."
  exit 0
fi

step "Preflight: GPU driver"
check_driver

if [[ "$DO_LOCK" == 1 ]]; then
  step "Resolving dependencies (uv lock)"
  uv lock
else
  echo "(--skip-lock: reusing existing uv.lock)"
fi

step "Installing (UV_LINK_MODE=copy uv sync)"
UV_LINK_MODE=copy uv sync

patch_distrax

step "Regenerating requirements.txt (uv export)"
uv export --no-hashes --no-dev -o requirements.txt

if [[ "$DO_SMOKE" == 1 ]]; then
  smoke_test
else
  echo "(--no-smoke: skipping smoke test)"
fi

echo
echo "=================================================================="
echo " Environment ready: jax[cuda12]==0.6.2, flax==0.10.2, distrax patched."
echo " After any reinstall of distrax, re-run: bash setup_env.sh --patch-only"
echo "=================================================================="
