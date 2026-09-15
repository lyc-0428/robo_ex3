#!/bin/bash
set -u

ROOT="$HOME/team21_ex3_ws/src/robo_ex3_real_test/robo_ex3-codex-jetson-dji-real-test"
PY="$ROOT/vision/random_visual_grasp_bottlefix.py"

cd "$ROOT" || exit 1

python3 -m py_compile "$PY" || exit 1

echo "============================================================"
echo " EXP3 随机抓取 - 水瓶识别增强版"
echo "============================================================"
echo "网球 : YOLO 320 / conf 0.35 / bbox width 判断距离"
echo "水瓶 : YOLO 640 / conf 0.25 / bbox height 判断距离"
echo "稳定 : 最近5次识别中同类出现3次即可"
echo "初始 : 指定类别先恢复对应学习机械臂姿态"
echo "底盘 : 最小前后移动40mm"
echo
echo "1) 网球：只自动转向/靠近，不抓"
echo "2) 水瓶：只自动转向/靠近，不抓"
echo "3) 自动类别：只自动转向/靠近，不抓"
echo "4) 网球：自动靠近 + 抓取 + 抬起"
echo "5) 水瓶：自动靠近 + 抓取 + 抬起"
echo "6) 自动类别：自动靠近 + 抓取 + 抬起"
echo "0) 退出"
echo

read -r -p "请输入 0-6: " choice

case "$choice" in
  1)
    read -r -p "底盘和机械臂会运动，确认安全后输入 YES: " ok
    [ "$ok" = "YES" ] && python3 "$PY" --target tennis_ball --nav-only
    ;;

  2)
    read -r -p "底盘和机械臂会运动，确认安全后输入 YES: " ok
    [ "$ok" = "YES" ] && python3 "$PY" --target bottle --nav-only
    ;;

  3)
    read -r -p "底盘和机械臂会运动，确认安全后输入 YES: " ok
    [ "$ok" = "YES" ] && python3 "$PY" --target auto --nav-only
    ;;

  4)
    read -r -p "会自动抓网球，确认安全后输入 YES: " ok
    [ "$ok" = "YES" ] && python3 "$PY" --target tennis_ball --keep-object
    ;;

  5)
    echo "注意：水瓶夹爪判定参数目前仍沿用网球。"
    read -r -p "会自动抓水瓶，确认安全后输入 YES: " ok
    [ "$ok" = "YES" ] && python3 "$PY" --target bottle --keep-object
    ;;

  6)
    echo "AUTO 会先搜索类别，再使用对应学习姿态和视觉尺寸闭环。"
    read -r -p "会自动抓取，确认安全后输入 YES: " ok
    [ "$ok" = "YES" ] && python3 "$PY" --target auto --keep-object
    ;;

  0)
    exit 0
    ;;

  *)
    echo "输入无效"
    exit 1
    ;;
esac
