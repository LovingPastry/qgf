#!/usr/bin/env bash
#
# download_ogbench.sh — helper to fetch OGBench 100M single-task datasets.
#
# The 100M datasets are NOT available through the `ogbench` python API; they
# live as ~100 pre-split .npz slices (each with a matching -val.npz) under
#   https://rail.eecs.berkeley.edu/datasets/ogbench/<env>-100m-v0/
# This script downloads them into the layout main.py expects:
#   $OGBENCH_DATA_DIR/<env>-100m-v0/<env>-v0-000.npz ...
#
# Usage:
#   ./download_ogbench.sh [options] <env> [<env> ...]
#
# Options:
#   -n N        download slices 0..N-1 (train + val).           (default: 5)
#   -a          download ALL slices (~32GB train + ~3GB val per env).
#   -d DIR      target data dir. Overrides $OGBENCH_DATA_DIR.
#   --no-val    skip the -val.npz files (NOTE: main.py loads them; not advised).
#   -l          list the known 100M environments and exit.
#   -h          show this help.
#
# Examples:
#   ./download_ogbench.sh cube-triple-play                 # first 5 slices
#   ./download_ogbench.sh -n 10 puzzle-4x4-play scene-play # first 10 slices, 2 envs
#   ./download_ogbench.sh -a cube-triple-play              # everything
#
# Downloads resume where they left off (wget -c) — safe to re-run.

set -euo pipefail

BASE_URL="https://rail.eecs.berkeley.edu/datasets/ogbench"

# Environments that have a 100M variant on the server.
KNOWN_ENVS=(
  cube-double-play
  cube-triple-play
  cube-quadruple-play
  cube-quadruple-noisy
  cube-octuple-play
  puzzle-3x3-play
  puzzle-4x4-play
  puzzle-4x5-play
  puzzle-4x6-play
  scene-play
  humanoidmaze-giant-navigate
)

NUM_SLICES=5
ALL=false
WITH_VAL=true
DATA_DIR="${OGBENCH_DATA_DIR:-$HOME/.ogbench/data}"

usage() { sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; }

list_envs() {
  echo "Known 100M environments:"
  for e in "${KNOWN_ENVS[@]}"; do echo "  $e"; done
}

# ---- parse args ----------------------------------------------------------
ENVS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) NUM_SLICES="$2"; shift 2 ;;
    -a) ALL=true; shift ;;
    -d) DATA_DIR="$2"; shift 2 ;;
    --no-val) WITH_VAL=false; shift ;;
    -l) list_envs; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    *) ENVS+=("$1"); shift ;;
  esac
done

if [[ ${#ENVS[@]} -eq 0 ]]; then
  echo "Error: no environment specified." >&2
  echo >&2
  list_envs >&2
  echo >&2
  usage >&2
  exit 1
fi

command -v wget >/dev/null 2>&1 || { echo "Error: wget is required but not found." >&2; exit 1; }

echo "Target data dir: $DATA_DIR"
mkdir -p "$DATA_DIR"

# ---- download ------------------------------------------------------------
for env in "${ENVS[@]}"; do
  # Warn (don't fail) on unrecognised names — the server is the source of truth.
  if ! printf '%s\n' "${KNOWN_ENVS[@]}" | grep -qx "$env"; then
    echo "Warning: '$env' is not in the known list; trying anyway." >&2
  fi

  dir="${env}-100m-v0"          # server directory / local subdir
  filebase="${env}-v0"          # slice filename prefix
  dest="$DATA_DIR/$dir"
  mkdir -p "$dest"

  echo
  echo "==> $env  ->  $dest"

  if $ALL; then
    # Mirror the whole directory; no need to know exact slice count/names.
    accept='*.npz'
    $WITH_VAL || accept='*[0-9].npz'   # numbered slices only, skip *-val.npz
    wget -c -r -np -nH --cut-dirs=2 -R "index.html*" -A "$accept" \
      -P "$DATA_DIR" "$BASE_URL/$dir/"
  else
    for i in $(seq 0 $((NUM_SLICES - 1))); do
      iii=$(printf '%03d' "$i")
      wget -c -P "$dest" "$BASE_URL/$dir/${filebase}-${iii}.npz"
      if $WITH_VAL; then
        wget -c -P "$dest" "$BASE_URL/$dir/${filebase}-${iii}-val.npz"
      fi
    done
  fi
done

echo
echo "Done. Point main.py at a dataset with:"
echo "  --ogbench_dataset_dir=$DATA_DIR/<env>-100m-v0/"
