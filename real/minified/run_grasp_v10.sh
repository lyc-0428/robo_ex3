#!/bin/bash
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/vision/grasp_v10.py"

cd "$ROOT" || exit 1
python3 -m py_compile "$PY" || exit 1

echo "============================================================"
echo " EXP3 Grasp V10 - ALIGN_PREGRASP 稳定版"
echo "============================================================"
echo "修复：接近阈值后先居中，连续2次满足距离+横向条件才低头"
echo "目标偏差>120px时只转向，不继续前冲"
echo "视觉线程不发送任何底盘/机械臂/夹爪运动命令"
echo
echo "1) 多目标：测试最近目标靠近，不抓"
echo "2) 多目标：自动抓取并连续分拣，最多6个"
echo "3) 单独测试网球完整抓取"
echo "4) 单独测试水瓶完整抓取"
echo "0) 退出"
echo
read -r -p "请输入 0-4: " choice
case "$choice" in
  1)
    read -r -p "底盘和机械臂会运动，确认安全后输入 YES: " ok
    [ "$ok" = "YES" ] && python3 "$PY" --target auto --sort-mode --nav-only --max-items 6
    ;;
  2)
    read -r -p "会连续移动/抓取/分类/扫描，确认安全后输入 YES: " ok
    [ "$ok" = "YES" ] && python3 "$PY" --target auto --sort-mode --max-items 6
    ;;
  3)
    read -r -p "会自动抓网球，确认安全后输入 YES: " ok
    [ "$ok" = "YES" ] && python3 "$PY" --target tennis_ball --keep-object
    ;;
  4)
    read -r -p "会自动抓水瓶，确认安全后输入 YES: " ok
    [ "$ok" = "YES" ] && python3 "$PY" --target bottle --keep-object
    ;;
  0) exit 0 ;;
  *) echo "输入无效"; exit 1 ;;
esac
