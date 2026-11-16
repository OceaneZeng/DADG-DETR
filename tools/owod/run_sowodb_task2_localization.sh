#!/usr/bin/env bash
# Reuse the verified launcher with a separate paired localization plan/output.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_sowodb_task2_routing.sh" --localization "$@"
