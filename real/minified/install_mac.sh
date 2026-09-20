#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
JETSON_HOST="${JETSON_HOST:-adam@10.140.247.161}"
REMOTE_DIR="${REMOTE_DIR:-/home/adam/Team21/yjh/real_grid_sorting_v10}"
REMOTE_STAGE="${REMOTE_DIR}.upload.$$"

echo "上传 V10 定向搜索版到 $JETSON_HOST:$REMOTE_DIR"
echo "本命令只上传文件，不启动机器人。"

ssh "$JETSON_HOST" "mkdir -p '$REMOTE_STAGE'"

COPYFILE_DISABLE=1 tar \
  --format=ustar \
  --exclude='logs' \
  --exclude='__pycache__' \
  --exclude='.DS_Store' \
  -cf - -C "$ROOT" . \
| ssh "$JETSON_HOST" "tar -xf - -C '$REMOTE_STAGE'"

ssh "$JETSON_HOST" \
  "chmod +x '$REMOTE_STAGE/run_grasp_v10.sh' && python3 -m py_compile '$REMOTE_STAGE/vision/grasp_v10.py' && if [ -d '$REMOTE_DIR' ]; then mv '$REMOTE_DIR' \"${REMOTE_DIR}.backup.\$(date +%Y%m%d_%H%M%S)\"; fi && mv '$REMOTE_STAGE' '$REMOTE_DIR'"

echo "上传完成。连接小车 Wi-Fi 后，在 Jetson 终端执行："
echo "cd '$REMOTE_DIR' && bash ./run_grasp_v10.sh"
