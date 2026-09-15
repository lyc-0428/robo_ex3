#!/bin/bash
set -euo pipefail

ROOT="$HOME/team21_ex3_ws/src/robo_ex3_real_test/robo_ex3-codex-jetson-dji-real-test"
PY="$ROOT/vision/grasp_v16_grid.py"

if [ ! -f "$PY" ]; then
  echo "找不到 $PY；请先上传 Python 文件。"
  exit 1
fi

cd "$ROOT"
python3 -m py_compile "$PY"

echo "实验三：网格自动分类"
echo "1) 选最近目标、靠近并原路退回（不抓）"
echo "2) 抓取并分类 1 件"
echo "3) 连续分类最多 6 件"
echo "0) 退出"
read -r -p "请选择 0-3: " choice

case "$choice" in
  0) exit 0 ;;
  1) max_items=1; mode_args=(--sort-mode --nav-only) ;;
  2) max_items=1; mode_args=(--sort-mode) ;;
  3) max_items=6; mode_args=(--sort-mode) ;;
  *) echo "输入无效"; exit 1 ;;
esac

read -r -p "输入网格左右外边线间距（米）: " grid_width_m
side_clearance_m="${SIDE_CLEARANCE_M:-0.25}"
left_y_sign="${LEFT_Y_SIGN:--1}"

python3 - "$grid_width_m" "$side_clearance_m" "$left_y_sign" <<'PY'
import sys
try:
    width, clearance = map(float, sys.argv[1:3])
    sign = int(sys.argv[3])
except ValueError:
    raise SystemExit("宽度、余量或方向不是有效数字")
if width <= 0 or clearance <= 0 or sign not in (-1, 1):
    raise SystemExit("宽度和余量必须大于0，方向只能是-1或1")
PY

echo "起点：网格下边线外、对准中线。"
echo "左侧方向 y=$left_y_sign；车中心越过边线余量 $side_clearance_m 米。"

python3 "$PY" \
  --grid-width-m "$grid_width_m" \
  --side-clearance-m "$side_clearance_m" \
  --left-y-sign "$left_y_sign" \
  --max-items "$max_items" \
  "${mode_args[@]}"
