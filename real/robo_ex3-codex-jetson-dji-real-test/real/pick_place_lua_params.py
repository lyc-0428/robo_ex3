#!/usr/bin/env python3
import os
import subprocess
import sys
import time

# 注意: 使用 pip 安装的官方 SDK (pip3 install --user ./RoboMaster-SDK)。
# 不要往 sys.path 插 ~/RoboMaster-SDK/src —— 板子上那份源码树版本混乱
# (client.py 引用不存在的 config.DEFAULT_CONN_PROTO), 会导致连接失败。
from robomaster import led
from robomaster import robot

RUN_COUNT = 5

INIT_X_MM = 120
INIT_Y_MM = 120
INIT_SETTLE_TIME = 1.0

EXTRA_FORWARD_MM = 30
COARSE_FORWARD_MM = 60
COARSE_DOWN_MM = -240
FINAL_FORWARD_MM = 6
FINAL_DOWN_MM = -24

TEST_LIFT_MM = 25
MAIN_LIFT_MM = 60
RELEASE_DOWN_MM = -80
FINAL_LIFT_MM = 70

OPEN_POWER = 35
# 判定功率 (实机扫描数据: 空夹 2.5s 到底, 夹网球 1.5s 被挡进闭合区)。
# 更低功率空夹会卡在中间走不到底, 更高功率两案时间差更小, 30 差距最大。
GRIP_POWER = 30
# 闭合速度阈值: closed 在这个时间内到达 = 被物体挡停 = 夹到物体;
# 更慢才到 = 走满全程到机械限位 = 空夹。
GRIP_CLOSED_FAST = 2.0
# 刚性物体(如方块)会把爪子停在中间 normal 不再前进:
# normal 连续保持这么久也判为夹到物体 (球不会走这个分支)。
GRIP_STABLE_TIME = 2.5
# 闭合后等待判定的最长总时间; 超时按失败处理。
GRIP_STATUS_TIMEOUT = 8.0

RIGHT_TURN_DEG = -180
TURN_SPEED_DPS = 20

OPEN_TIME = 1.5
CHASSIS_SETTLE_TIME = 1.2
FORWARD_MOVE_TIME = 1.0
FORWARD_SETTLE_TIME = 1.0
COARSE_MOVE_TIME = 1.0
COARSE_SETTLE_TIME = 1.0
FINAL_MOVE_TIME = 1.5
GRASP_HEIGHT_PAUSE_TIME = 1.2
TEST_LIFT_TIME = 1.5
TEST_SETTLE_TIME = 1.0
MAIN_LIFT_TIME = 1.5
BEFORE_TURN_TIME = 1.5
AFTER_TURN_TIME = 1.5
LOWER_TIME = 1.5
GROUND_PAUSE_TIME = 1.2
RELEASE_TIME = 1.5
FINAL_LIFT_TIME = 1.5

RETRY_PAUSE_TIME = 2.0

gripper_status = {"value": "unknown", "ts": 0.0}

# ROS2 包装节点(real_pick_place_ros2_node.py)会注入发布函数,
# 使 ERROR/SUCCESS 状态同步显示到 /real_pick_place/status 话题。
status_hook = None

# 每次运行的完整输出(测试结果 + 出错日志)自动保存到桌面, 实验报告直接引用。
LOG_DIR = os.path.expanduser("~/Desktop")
LOG_LINES = []


class _Tee:
    """把 stdout 的输出同时收集到 LOG_LINES, 供结束时保存到桌面。"""

    def __init__(self, stream):
        self.stream = stream

    def write(self, text):
        self.stream.write(text)
        if text.strip():
            LOG_LINES.append(text.rstrip("\n"))
        return len(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        # isatty/encoding/fileno 等属性委托给原流, 避免下游代码报错
        return getattr(self.stream, name)


def save_run_log():
    """把本次运行的完整输出保存到桌面 (无论成败, main 的 finally 里调用)。

    文件名: test_<时间戳>.txt, 如 test_20260911_213045.txt (拍视频验收用)。
    """
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LOG_DIR, f"test_{stamp}.txt")
        with open(path, "w") as f:
            f.write("\n".join(LOG_LINES) + "\n")
        print(f"运行日志已保存: {path}")
    except Exception as e:
        print(f"保存运行日志失败: {e}")


def current_wifi_ssid():
    """返回板子当前连接的 WiFi SSID; 读不到返回 ""。"""
    try:
        out = subprocess.run(
            ["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        for line in out.splitlines():
            if line.startswith("yes:"):
                return line.split(":", 1)[1]
    except Exception:
        return "?"
    return ""


def report_status(text):
    print(text)
    if status_hook is not None:
        try:
            status_hook(text)
        except Exception:
            pass


def step(attempt, number, label, sec=0.0):
    print("\n========================================")
    print(f"RUN {attempt}/{RUN_COUNT} | STEP {number}: {label}")
    print("========================================")
    if sec > 0:
        time.sleep(sec)


def on_gripper_status(status):
    if isinstance(status, (list, tuple)) and status:
        status = status[0]
    gripper_status["value"] = status
    gripper_status["ts"] = time.time()


def close_and_judge(gripper):
    """闭合夹爪并判定是否夹到物体, 返回 (success, detail)。

    判据来自实机功率-状态扫描数据 (GRIP_POWER=30):
      1. status 到达 "closed" 用时 <= GRIP_CLOSED_FAST (2.0s)
         -> 物体挡在行程中间, 爪子提前进入闭合区 -> 夹到物体
            (实测: 夹网球 1.5s)
      2. status 到达 "closed" 用时 > GRIP_CLOSED_FAST
         -> 爪子走满全程到机械限位 -> 未夹到 (实测: 空夹 2.5s)
      3. status 稳定在 "normal" 持续 GRIP_STABLE_TIME 且从未 closed
         -> 物体把爪子停在中间 -> 夹到物体 (刚性方块走这个分支)
      4. 超时 / 无推送 -> 无法判定 -> 失败

    注意: 不要在中途 pause() —— 空夹时爪子还没走到限位就被冻结在
    中间位置, 状态会停在 normal 被误判为夹到物体。
    """
    gripper_status["value"] = "unknown"
    gripper_status["ts"] = 0.0
    gripper.close(power=GRIP_POWER)

    t0 = time.time()
    deadline = t0 + GRIP_STATUS_TIMEOUT
    last = None
    stable_since = None

    while True:
        now = time.time()
        v = str(gripper_status["value"]).strip().lower()
        ts = gripper_status.get("ts", 0.0)

        # 判据 1/2: closed 到达时间
        if v == "closed":
            elapsed = now - t0
            if elapsed <= GRIP_CLOSED_FAST:
                return True, (
                    f"快速闭合({elapsed:.1f}s <= {GRIP_CLOSED_FAST}s), "
                    f"判定夹到物体"
                )
            return False, (
                f"缓慢闭合({elapsed:.1f}s > {GRIP_CLOSED_FAST}s), "
                f"走满行程, 未夹到物体"
            )

        # 判据 3: normal 持续稳定 (要求推送仍在流动, ts 新鲜)
        if v != last:
            last = v
            stable_since = now
        elif (v == "normal" and stable_since is not None
              and now - stable_since >= GRIP_STABLE_TIME
              and ts > 0 and now - ts < 1.0):
            return True, (
                f"夹爪稳定停在中间位置 normal({GRIP_STABLE_TIME}s), "
                f"判定夹到物体"
            )

        if now >= deadline:
            return False, (
                f"超时({GRIP_STATUS_TIMEOUT}s)未得到稳定判定, "
                f"最后状态 {gripper_status['value']}"
            )

        time.sleep(0.2)


def move_arm_delta(arm, dx_mm, dy_mm, label, wait_time):
    print(f"{label}: arm.move x={dx_mm} mm, y={dy_mm} mm")
    arm.move(x=dx_mm, y=dy_mm).wait_for_completed()
    time.sleep(wait_time)


def safe_home(arm, gripper):
    print("\n========================================")
    print("SAFE HOME: open gripper and recenter arm")
    print("========================================")
    try:
        gripper.open(power=OPEN_POWER)
        time.sleep(1.0)
    except Exception as e:
        print("Open gripper warning:", e)

    try:
        arm.recenter().wait_for_completed()
        time.sleep(2.0)
    except Exception as e:
        print("Arm recenter warning:", e)


def robot_signal_error(ep):
    """抓取失败报错显示: 全车装甲灯红色闪烁 + 警报音效。"""
    try:
        ep.led.set_led(
            comp=led.COMP_ALL, r=255, g=0, b=0,
            effect=led.EFFECT_FLASH, freq=5,
        )
    except Exception as e:
        print("Set error LED warning:", e)
    try:
        ep.play_sound(robot.SOUND_ID_ATTACK).wait_for_completed(timeout=3)
    except Exception as e:
        print("Play error sound warning:", e)


def robot_signal_success(ep):
    """抓取成功显示: 全车装甲灯绿色常亮 + 成功音效。"""
    try:
        ep.led.set_led(
            comp=led.COMP_ALL, r=0, g=255, b=0,
            effect=led.EFFECT_ON,
        )
    except Exception as e:
        print("Set success LED warning:", e)
    try:
        ep.play_sound(robot.SOUND_ID_RECOGNIZED).wait_for_completed(timeout=3)
    except Exception as e:
        print("Play success sound warning:", e)


def robot_signal_idle(ep):
    """熄灭装甲灯, 恢复默认状态。"""
    try:
        ep.led.set_led(comp=led.COMP_ALL, effect=led.EFFECT_OFF)
    except Exception as e:
        print("Set idle LED warning:", e)


def initialize_arm(attempt, arm, gripper):
    step(attempt, "0A", "RECENTER ARM")
    arm.recenter().wait_for_completed()
    time.sleep(2.0)

    step(attempt, "0B", "MOVE TO INITIAL ARM POSE")
    print(f"initial pose: arm.moveto x={INIT_X_MM} mm, y={INIT_Y_MM} mm")
    arm.moveto(x=INIT_X_MM, y=INIT_Y_MM).wait_for_completed()
    time.sleep(INIT_SETTLE_TIME)

    step(attempt, 1, "OPEN GRIPPER")
    gripper.open(power=OPEN_POWER)
    time.sleep(OPEN_TIME)


def run_once(attempt, ep, arm, gripper, chassis):
    initialize_arm(attempt, arm, gripper)

    step(attempt, 2, "CHASSIS SETTLE")
    time.sleep(CHASSIS_SETTLE_TIME)

    step(attempt, 3, "FIRST FORWARD ALIGNMENT")
    move_arm_delta(arm, EXTRA_FORWARD_MM, 0, "forward alignment", FORWARD_MOVE_TIME)

    step(attempt, 4, "FORWARD SETTLE")
    time.sleep(FORWARD_SETTLE_TIME)

    step(attempt, 5, "FORWARD-LEANING COARSE DESCENT")
    move_arm_delta(arm, COARSE_FORWARD_MM, COARSE_DOWN_MM, "lean forward and down", COARSE_MOVE_TIME)

    step(attempt, 6, "PAUSE BEFORE FINAL APPROACH")
    time.sleep(COARSE_SETTLE_TIME)

    step(attempt, 7, "FINAL FORWARD-LEANING DESCENT")
    move_arm_delta(arm, FINAL_FORWARD_MM, FINAL_DOWN_MM, "final forward and down", FINAL_MOVE_TIME)

    step(attempt, 8, "AT GRASP POSITION")
    time.sleep(GRASP_HEIGHT_PAUSE_TIME)

    step(attempt, 9, "CLOSE GRIPPER")
    step(attempt, 10, "JUDGE GRASP BY CLOSE TIME")
    ok, detail = close_and_judge(gripper)
    print(f"gripper close judgment: ok={ok}, detail={detail}")

    # 关键判定: close_and_judge 按"闭合速度"区分夹到物体与空夹
    # (见函数 docstring)。未夹到 => 报错并退出循环。
    if not ok:
        report_status(
            f"ERROR: RUN {attempt} 抓取失败 - {detail}, 停止循环"
        )
        robot_signal_error(ep)
        safe_home(arm, gripper)
        return False

    report_status(f"SUCCESS: RUN {attempt} {detail}")
    robot_signal_success(ep)

    step(attempt, 11, "TEST LIFT")
    move_arm_delta(arm, 0, TEST_LIFT_MM, "small test lift", TEST_LIFT_TIME)

    step(attempt, 12, "CHECK REAL GRASP")
    print("Check visually whether the cube moved with the gripper.")
    time.sleep(TEST_SETTLE_TIME)

    step(attempt, 13, "MAIN LIFT")
    move_arm_delta(arm, 0, MAIN_LIFT_MM, "main lift", MAIN_LIFT_TIME)

    step(attempt, 14, "SETTLE BEFORE TURN")
    time.sleep(BEFORE_TURN_TIME)

    step(attempt, 15, "RIGHT TURN TO PLACE AREA")
    chassis.move(x=0, y=0, z=RIGHT_TURN_DEG, z_speed=TURN_SPEED_DPS).wait_for_completed()
    time.sleep(AFTER_TURN_TIME)

    step(attempt, 16, "LOWER CUBE AT B POINT")
    move_arm_delta(arm, 0, RELEASE_DOWN_MM, "lower cube", LOWER_TIME)

    step(attempt, 17, "GROUND SETTLE")
    time.sleep(GROUND_PAUSE_TIME)

    step(attempt, 18, "RELEASE CUBE")
    gripper.open(power=OPEN_POWER)
    time.sleep(RELEASE_TIME)

    step(attempt, 19, "LIFT ARM AWAY")
    move_arm_delta(arm, 0, FINAL_LIFT_MM, "lift away", FINAL_LIFT_TIME)

    step(attempt, 20, "DONE - KEEP FINAL POSE")
    print("This run finished. No chassis turn-back. Next run will initialize arm again.")
    report_status(f"SUCCESS: RUN {attempt} 抓取-放置完成")
    robot_signal_idle(ep)
    return True


def main():
    """入口: 收集本次运行的完整输出, 结束时保存运行日志到桌面。"""
    # stdout/stderr 都走 _Tee, 未捕获异常的回溯也会进日志文件
    stdout_orig, stderr_orig = sys.stdout, sys.stderr
    sys.stdout = _Tee(stdout_orig)
    sys.stderr = _Tee(stderr_orig)
    try:
        return _run()
    finally:
        # 无论成功、报错还是异常退出, 都保存本次运行日志
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig
        save_run_log()


def _run():
    print("Connecting RoboMaster...")

    # 预检: 板子必须已连机器人热点, 否则 SDK 静默失败,
    # 最后 close() 时还会炸出 NoneType.is_alive 的晦涩报错。
    ssid = current_wifi_ssid()
    print(f"current wifi: {ssid!r}")
    if not ssid.startswith("RMEP"):
        report_status(
            f"ERROR: 板子当前不在机器人热点上({ssid!r}), 中止。"
            f"先开机机器人, 然后执行: nmcli connection up RMEP-21bbc5"
        )
        return 0, RUN_COUNT

    ep = robot.Robot()
    try:
        initialized = ep.initialize(conn_type="ap")
    except Exception as e:
        # 初始化失败时不能调 ep.close() (SDK 的 stop() 会对未创建的
        # 连接线程调 is_alive, 报 NoneType 错误), 直接返回。
        report_status(f"ERROR: SDK 初始化异常, 中止: {e}")
        return 0, RUN_COUNT

    if not initialized:
        # initialize 可能不抛异常而是返回 False (连不上机器人)。
        # 同样不能调 ep.close(), 直接返回。
        report_status("ERROR: 连不上机器人, 检查机器人是否开机, 中止")
        return 0, RUN_COUNT

    arm = ep.robotic_arm
    gripper = ep.gripper
    chassis = ep.chassis

    success_count = 0
    fail_count = 0

    try:
        try:
            gripper.sub_status(freq=5, callback=on_gripper_status)
            time.sleep(0.5)
        except Exception as e:
            print("Gripper status subscribe warning:", e)

        for attempt in range(1, RUN_COUNT + 1):
            ok = run_once(attempt, ep, arm, gripper, chassis)

            if not ok:
                fail_count += 1
                print("Loop stopped because grasp failed.")
                break

            success_count += 1

            if attempt < RUN_COUNT:
                print(f"Run {attempt} success. Prepare object, then next run starts.")
                time.sleep(RETRY_PAUSE_TIME)

        print("\n========================================")
        print("FINAL RESULT")
        print("========================================")
        print(f"success: {success_count}")
        print(f"failed: {fail_count}")
        print(f"planned runs: {RUN_COUNT}")

        # 验收判定 (与仿真侧一致: 成功率 >= 80%, 即 5 次至少成功 4 次)
        passed = success_count >= max(1, int(0.8 * RUN_COUNT))
        report_status(
            f"验收结果: {'通过' if passed else '未通过'} "
            f"({success_count}/{RUN_COUNT})"
        )

        # 最终判定: 失败保持红灯闪烁报错, 全部成功亮绿灯报成功
        if fail_count > 0:
            report_status(
                f"ERROR: 抓取失败, 循环已停止 "
                f"(success={success_count}, failed={fail_count}, planned={RUN_COUNT})"
            )
            robot_signal_error(ep)
        else:
            report_status(f"SUCCESS: 全部 {success_count}/{RUN_COUNT} 次抓取成功")
            robot_signal_success(ep)

        return success_count, fail_count

    except KeyboardInterrupt:
        print("Interrupted by user.")
        safe_home(arm, gripper)
        return None

    except Exception as e:
        print("Unexpected error:", e)
        safe_home(arm, gripper)
        robot_signal_error(ep)
        raise

    finally:
        try:
            gripper.unsub_status()
        except Exception:
            pass
        ep.close()


if __name__ == "__main__":
    import sys

    # 可选: python3 real/pick_place_lua_params.py <grip_power>
    # 用于在真机上快速测试不同闭合功率 (不经过 ROS2 节点)
    if len(sys.argv) > 1:
        try:
            GRIP_POWER = int(sys.argv[1])
        except ValueError:
            print(f"ERROR: 无效的功率参数: {sys.argv[1]!r}, 应为数字。用法:")
            print("  python3 real/pick_place_lua_params.py        # 默认功率")
            print("  python3 real/pick_place_lua_params.py 30     # 指定功率 30")
            print("  (功率扫描请用: python3 real/gripper_status_sweep.py empty)")
            sys.exit(1)
        print(f"grip power overridden from command line: {GRIP_POWER}")
    main()
