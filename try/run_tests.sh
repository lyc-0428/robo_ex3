#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
python3 -m unittest discover -s "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/tests" -v

