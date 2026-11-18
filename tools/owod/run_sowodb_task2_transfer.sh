#!/usr/bin/env bash
set -euo pipefail
epochs=3
args=()
for arg in "$@"; do
  case "$arg" in
    --full) epochs=50 ;;
    --check) args+=(--check) ;;
    -h|--help) echo 'Usage: bash tools/owod/run_sowodb_task2_transfer.sh [--full] [--check]'; exit 0 ;;
    *) echo "Unknown transfer option: $arg" >&2; exit 2 ;;
  esac
done
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_sowodb_task2_routing.sh" --transfer --epochs "$epochs" "${args[@]}"
