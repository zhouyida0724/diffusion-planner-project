#!/usr/bin/env bash
# Wrapper for backwards-compatible entrypoint.
# Implementation lives in scripts/export/diffusion_planner/pipeline/run_export_sharded_mod.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/export/diffusion_planner/pipeline/run_export_sharded_mod.sh" "$@"
