#!/usr/bin/env bash
set -euo pipefail

TRY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${TRY_ROOT}/recommended_overlay/digital"
TARGET_ROOT="${1:-$(cd -- "${TRY_ROOT}/.." && pwd)/digital}"
MODE="${2:-dry-run}"

case "$(realpath -m -- "${TARGET_ROOT}")" in
  "$(realpath -- "${TRY_ROOT}/..")"/*) ;;
  *) echo "Refusing to write outside experiment3: ${TARGET_ROOT}" >&2; exit 2 ;;
esac

echo "Overlay source: ${SOURCE_ROOT}"
echo "Digital target: ${TARGET_ROOT}"
if [[ "${MODE}" != "--apply" ]]; then
  echo "Dry run only. Re-run with the second argument --apply to copy it."
  exit 0
fi

cp -a -- "${SOURCE_ROOT}/." "${TARGET_ROOT}/"
echo "Overlay copied. No git commit or push was performed."

