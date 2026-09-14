#!/bin/bash
set -u

ROOT="$HOME/team21_ex3_ws/src/robo_ex3_real_test/robo_ex3-codex-jetson-dji-real-test"
PY="$ROOT/vision/vision_pick_place_mobile_async.py"

cd "$ROOT" || exit 1
python3 -m py_compile "$PY" || exit 1

echo "=================================================="
echo " EXP3 异步视频优先版"
echo " 视频读取 : 独立线程，持续取最新帧"
echo " 视频显示 : 30 FPS"
echo " YOLO     : 320x320，最多 6 FPS，独立线程"
echo " 控制动作 : 不再阻塞视频"
echo "=================================================="
echo "1) 持续定位测试：不动车、不抓，按q退出"
echo "2) 自动转向+靠近：动车，但不抓"
echo "3) 网球：自动靠近+抓取+放回"
echo "4) 自动类别：自动靠近+抓取+放回"
echo "5) 水瓶：自动靠近+抓取+放回"
echo "0) 退出"
echo

read -r -p "请输入 0-5: " choice

COMMON="--imgsz 320 --display-fps 30 --infer-fps 6 --yaw-sign -1 --grasp-target 220 --distance-tol 15"

case "$choice" in
  1)
    python3 "$PY" --target auto --dry-run $COMMON
    ;;
  2)
    echo "底盘会自动转向并前后移动。"
    read -r -p "确认安全后输入 YES: " ok
    [ "$ok" = "YES" ] && python3 "$PY" --target auto --nav-only $COMMON
    ;;
  3)
    read -r -p "底盘和机械臂都会运动，确认安全后输入 YES: " ok
    [ "$ok" = "YES" ] && python3 "$PY" --target tennis_ball $COMMON
    ;;
  4)
    read -r -p "底盘和机械臂都会运动，确认安全后输入 YES: " ok
    [ "$ok" = "YES" ] && python3 "$PY" --target auto $COMMON
    ;;
  5)
    echo "注意：bottle夹爪参数仍未专门扫参。"
    read -r -p "确认安全后输入 YES: " ok
    [ "$ok" = "YES" ] && python3 "$PY" --target bottle $COMMON
    ;;
  0)
    exit 0
    ;;
  *)
    echo "输入无效"
    exit 1
    ;;
esac
