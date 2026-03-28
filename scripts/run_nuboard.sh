#!/usr/bin/env bash
set -euo pipefail

# run_nuboard.sh
# One-click nuBoard launcher.
# Defaults to the latest simulation output under outputs/sim/exp/simulation/**/<timestamp>/
# and starts nuBoard with simulation_path=[<that_dir>].
#
# This script is a thin wrapper around scripts/run_nuboard_outputs.sh.
# The previous behavior is preserved in scripts/run_nuboard_legacy.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/run_nuboard_outputs.sh" "$@"
