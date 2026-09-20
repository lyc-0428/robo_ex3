#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/vision/grasp_v10.py"
LOG_DIR="$ROOT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/grasp_v10_${STAMP}.log"

cd "$ROOT"
mkdir -p "$LOG_DIR"
python3 -m py_compile "$PY"

echo "============================================================"
echo " 实验三真机 V10：恢复动作，仅修正计数退出"
echo " BUILD: RESTORED-MOTION-COUNT-ONLY"
echo "============================================================"
echo "网球放左侧后向右搜索；水瓶放右侧后向左搜索"
echo "离开放置区并重新看到黑色标记后，才接受下一个目标"
echo "直接启动任务；动作次数不等于物体数，往返搜索无目标后结束"
echo "日志：$LOG_FILE"
echo "============================================================"

set +e
python3 "$PY" \
  --target auto \
  --sort-mode \
  --max-items 6 \
  2>&1 | tee "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e

exit "$status"
