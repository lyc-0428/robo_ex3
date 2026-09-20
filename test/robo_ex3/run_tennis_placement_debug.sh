#!/usr/bin/env bash
set -eo pipefail

DIGITAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEED="${1:-21}"

# Temporary placement-debug entry point: spawn and accept tennis balls only.
exec bash "${DIGITAL_ROOT}/run_vision_sorting_improved.sh" \
  "${SEED}" tennis_only
