#!/usr/bin/env bash
set -euo pipefail

# run_nuboard.sh
# One-click nuBoard launcher.
# Defaults to the latest simulation output under outputs/sim/exp/simulation/**/<timestamp>/
# and starts nuBoard with simulation_path=[<that_dir>].
#
# This script is a thin wrapper around scripts/nuboard/run_nuboard_outputs.sh.
# Legacy behavior removed (keep using run_nuboard_outputs.sh).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/run_nuboard_outputs.sh" "$@"
