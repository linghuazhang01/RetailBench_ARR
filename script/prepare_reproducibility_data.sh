#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export LC_ALL=C
export LANG=C

CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=1
fi

if ! command -v zstd >/dev/null 2>&1; then
  echo "Missing dependency: zstd. Install zstd before extracting the archives." >&2
  exit 1
fi

reconstruct_archive() {
  local archive="$1"
  if [[ -f "artifacts/${archive}" ]]; then
    return
  fi
  if compgen -G "artifacts/${archive}.part-*" >/dev/null; then
    cat "artifacts/${archive}.part-"* > "artifacts/${archive}"
    return
  fi
  echo "Missing archive or chunks for ${archive}." >&2
  exit 1
}

reconstruct_archive "environment_data.tar.zst"
reconstruct_archive "raw_trajectories_no_checkpoints.tar.zst"

if command -v shasum >/dev/null 2>&1; then
  (cd artifacts && shasum -a 256 -c SHA256SUMS)
elif command -v sha256sum >/dev/null 2>&1; then
  (cd artifacts && sha256sum -c SHA256SUMS)
else
  echo "Warning: no SHA-256 checker found; skipping archive checksum verification." >&2
fi

if [[ "$CHECK_ONLY" == "1" ]]; then
  zstd -t artifacts/environment_data.tar.zst artifacts/raw_trajectories_no_checkpoints.tar.zst
  echo "Archive checks passed."
  exit 0
fi

zstd -dc artifacts/environment_data.tar.zst | tar -xf -
zstd -dc artifacts/raw_trajectories_no_checkpoints.tar.zst | tar -xf -

echo "Restored data/ and paper_submit_data/raw_runs/."
