#!/usr/bin/env bash
# Fresh matched baseline/optimized runs; three epochs each by default.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_sowodb_task2_routing.sh" --unknown --epochs 3 "$@"
