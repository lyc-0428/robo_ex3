#!/bin/bash
set -u

ROOT="$HOME/team21_ex3_ws/src/robo_ex3_real_test/robo_ex3-codex-jetson-dji-real-test"
PY="$ROOT/vision/grasp_v10.py"

cd "$ROOT" || exit 1

python3 -m py_compile "$PY" || exit 1

echo "============================================================"
echo " EXP3 Grasp V10 - 网球近距中间观察姿态"
echo "============================================================"
echo "本版针对当前网球问题:"
echo "  远距离能识别和靠近"
echo "  但接近 width≈55~60px 时，夹爪开始遮挡网球"
echo "  V9直到最终PREGRASP才下降机械臂，因此来不及"
echo
echo "V10网球流程:"
echo "  1. recenter 远距离观察/靠近"
echo "  2. 接近预抓取区域(width约>=51px)"
echo "  3. 停车"
echo "  4. 机械臂先相对下降45mm（不是最终抓取姿态）"
echo "  5. 相机姿态变化后重新寻找网球"
echo "  6. 在中间观察姿态慢速靠近 + 横向对准"
echo "  7. 达到最终阈值后再进入学习抓取姿态"
echo "  8. CLOSE_APPROACH -> GRASP"
echo
echo "水瓶流程保持V9，不增加中间下降，避免破坏已验证抓瓶功能。"
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
    [ "$ok" = "YES" ] && \
      python3 "$PY" --target auto --sort-mode --nav-only --max-items 6
    ;;
  2)
    echo "网球放左侧，水瓶放右侧。"
    read -r -p "会连续移动/抓取/分类/扫描，确认场地安全后输入 YES: " ok
    [ "$ok" = "YES" ] && \
      python3 "$PY" --target auto --sort-mode --max-items 6
    ;;
  3)
    read -r -p "会自动抓网球，确认安全后输入 YES: " ok
    [ "$ok" = "YES" ] && \
      python3 "$PY" --target tennis_ball --keep-object
    ;;
  4)
    read -r -p "会自动抓水瓶，确认安全后输入 YES: " ok
    [ "$ok" = "YES" ] && \
      python3 "$PY" --target bottle --keep-object
    ;;
  0)
    exit 0
    ;;
  *)
    echo "输入无效"
    exit 1
    ;;
esac
