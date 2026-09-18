#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import time
import threading
from robomaster import robot

MOVE_Y = +0.220
MOVE_SPEED = 0.10

YAW_TOL_DEG = 0.6
MAX_CORRECTION_DEG = 8.0
CORRECTION_SPEED = 8

ROOT = "/home/adam/team21_ex3_ws/src/robo_ex3_real_test/robo_ex3-codex-jetson-dji-real-test"
LOG_DIR = os.path.join(ROOT, "diagnostics", "move220_yaw")

lock = threading.Lock()
yaw_value = None


def attitude_cb(attitude):
    global yaw_value
    yaw, pitch, roll = attitude
    with lock:
        yaw_value = float(yaw)


def get_yaw():
    with lock:
        return yaw_value


def wrap_deg(x):
    while x > 180:
        x -= 360
    while x < -180:
        x += 360
    return x


def wait_yaw(timeout=3.0):
    deadline = time.time() + timeout

    while time.time() < deadline:
        y = get_yaw()
        if y is not None:
            return y
        time.sleep(0.05)

    raise RuntimeError("没有收到 IMU yaw")


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    ep = robot.Robot()

    try:
        ep.initialize(conn_type="ap")
        chassis = ep.chassis

        chassis.sub_attitude(
            freq=20,
            callback=attitude_cb,
        )

        time.sleep(0.8)

        yaw0 = wait_yaw()

        print()
        print("============================================")
        print("220 mm 横移 + 停车后航向纠偏测试")
        print("============================================")
        print(f"START yaw = {yaw0:+.2f} deg")
        print(f"MOVE y    = {MOVE_Y:+.3f} m")
        print()

        # ----------------------------------------------------
        # 1. 使用已经验证过的 chassis.move 做横移
        # ----------------------------------------------------

        action = chassis.move(
            x=0,
            y=MOVE_Y,
            z=0,
            xy_speed=MOVE_SPEED,
        )

        action.wait_for_completed()

        try:
            chassis.drive_speed(
                x=0,
                y=0,
                z=0,
                timeout=0.3,
            )
        except Exception:
            pass

        time.sleep(0.6)

        yaw1 = wait_yaw()

        drift = wrap_deg(yaw1 - yaw0)

        print(f"AFTER MOVE yaw = {yaw1:+.2f} deg")
        print(f"YAW DRIFT      = {drift:+.2f} deg")

        # ----------------------------------------------------
        # 2. 停车后只修正航向
        # ----------------------------------------------------

        correction = wrap_deg(yaw0 - yaw1)

        correction = max(
            -MAX_CORRECTION_DEG,
            min(MAX_CORRECTION_DEG, correction)
        )

        corrected = False

        if abs(correction) > YAW_TOL_DEG:

            print()
            print(
                f"[CORRECT] 原地修正 "
                f"z={correction:+.2f} deg"
            )

            chassis.move(
                x=0,
                y=0,
                z=correction,
                z_speed=CORRECTION_SPEED,
            ).wait_for_completed()

            time.sleep(0.5)

            corrected = True

        yaw2 = wait_yaw()

        final_error = wrap_deg(yaw2 - yaw0)

        print()
        print("============================================")
        print(f"FINAL yaw      = {yaw2:+.2f} deg")
        print(f"FINAL ERROR    = {final_error:+.2f} deg")
        print("============================================")

        data = {
            "move_y_m": MOVE_Y,
            "yaw_start_deg": round(yaw0, 3),
            "yaw_after_move_deg": round(yaw1, 3),
            "yaw_drift_deg": round(drift, 3),
            "correction_command_deg": round(correction, 3),
            "correction_applied": corrected,
            "yaw_final_deg": round(yaw2, 3),
            "yaw_final_error_deg": round(final_error, 3),
        }

        stamp = time.strftime("%Y%m%d_%H%M%S")

        path = os.path.join(
            LOG_DIR,
            f"move220_{stamp}.json",
        )

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print()
        print("LOG:", path)

    finally:

        try:
            ep.chassis.drive_speed(
                x=0,
                y=0,
                z=0,
                timeout=0.2,
            )
        except Exception:
            pass

        try:
            ep.chassis.unsub_attitude()
        except Exception:
            pass

        try:
            ep.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
