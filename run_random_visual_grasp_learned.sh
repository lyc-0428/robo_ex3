#!/bin/bash
set -u

ROOT="$HOME/team21_ex3_ws/src/robo_ex3_real_test/robo_ex3-codex-jetson-dji-real-test"
PY="$ROOT/vision/random_visual_grasp_learned.py"

cd "$ROOT" || exit 1

python3 -m py_compile "$PY" || exit 1

echo "============================================================"
echo " EXP3 学习视觉随机抓取测试"
echo "============================================================"
echo "使用已学习的视觉位置 + 机械臂姿态"
echo "距离控制：bbox 宽度比例，不再使用 d=180/220mm"
echo "最小底盘前后移动：40mm"
echo
echo "1) 自动类别：只自动转向/靠近，不抓"
echo "2) 网球：只自动转向/靠近，不抓"
echo "3) 水瓶：只自动转向/靠近，不抓"
echo "4) 自动类别：自动靠近 + 抓取 + 抬起"
echo "5) 网球：自动靠近 + 抓取 + 抬起"
echo "6) 水瓶：自动靠近 + 抓取 + 抬起"
echo "0) 退出"
echo

read -r -p "请输入 0-6: " choice

case "$choice" in
  1)
    read -r -p "底盘和机械臂会运动，确认安全后输入 YES: " ok
    [ "$ok" = "YES" ] && python3 "$PY" --target auto --nav-only
    ;;
  2)
    read -r -p "底盘和机械臂会运动，确认安全后输入 YES: " ok
    [ "$ok" = "YES" ] && python3 "$PY" --target tennis_ball --nav-only
    ;;
  3)
    read -r -p "底盘和机械臂会运动，确认安全后输入 YES: " ok
    [ "$ok" = "YES" ] && python3 "$PY" --target bottle --nav-only
    ;;
  4)
    read -r -p "会自动抓取，确认安全后输入 YES: " ok
    [ "$ok" = "YES" ] && python3 "$PY" --target auto --keep-object
    ;;
  5)
    read -r -p "会自动抓网球，确认安全后输入 YES: " ok
    [ "$ok" = "YES" ] && python3 "$PY" --target tennis_ball --keep-object
    ;;
  6)
    echo "注意：水瓶夹爪参数目前仍沿用网球。"
    read -r -p "会自动抓水瓶，确认安全后输入 YES: " ok
    [ "$ok" = "YES" ] && python3 "$PY" --target bottle --keep-object
    ;;
  0)
    exit 0
    ;;
  *)
    echo "输入无效"
    exit 1
    ;;
esac
