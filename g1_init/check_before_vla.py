#!/usr/bin/env python3
"""Preflight check before starting infer_server / behavior_tree.

Must run while infer_server is NOT running (G1 SDK is a process singleton).
Empty camera frames trigger one HPU launcher restart, then this script re-execs.
Motion init False triggers one XCU launcher restart, then this script re-execs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from galbot_sdk.g1 import GalbotMotion, GalbotNavigation, GalbotRobot, SensorType


CAMERAS = {
    "head_left": SensorType.HEAD_LEFT_CAMERA,
    "left_arm": SensorType.LEFT_ARM_CAMERA,
    "right_arm": SensorType.RIGHT_ARM_CAMERA,
}
EXPECTED_WH = {
    "head_left": (640, 480),
    "left_arm": (1280, 720),
    "right_arm": (1280, 720),
}
CAPTURE_BINS = (
    ("/data/galbot/bin/front_head_camera_capture", "head"),
    ("/data/galbot/bin/left_arm_camera_capture", "left_arm"),
    ("/data/galbot/bin/right_arm_camera_capture", "right_arm"),
)
WARMUP_SECONDS = 5.0
RGB_RETRIES = 8
RGB_RETRY_DELAY = 0.3
CAPTURE_WAIT_SECONDS = 20.0
XCU_WAIT_SECONDS = 60.0
SUDO_PASSWORD = "gb@2023"
XCU_HOSTS = ("192.168.100.66", "192.168.1.66")
XCU_USER = "root"
XCU_PASSWORD = "12345678"

EXIT_OK = 0
EXIT_CAMERAS = 1
EXIT_MOTION = 2
EXIT_BOTH = 3
EXIT_INFER_SERVER = 4
EXIT_ROBOT = 5
EXIT_LAUNCHER = 6
EXIT_XCU = 7


def decode_rgb(compressed: dict | None):
    if not compressed:
        return None
    raw = compressed.get("data")
    if not raw:
        return None
    nparr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def infer_server_pids() -> list[str]:
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", "infer_server.py"], text=True, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return []
    pids = []
    for line in out.splitlines():
        if "pgrep" in line:
            continue
        pids.append(line.strip())
    return pids


def _bin_pids(binary: str) -> list[tuple[int, str]]:
    marker = (binary + "\x00").encode()
    found: list[tuple[int, str]] = []
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if cmdline.startswith(marker):
            text = cmdline.replace(b"\x00", b" ").decode(errors="replace").strip()
            found.append((int(entry.name), text))
    return found


def capture_para_label(cmdline: str) -> str:
    if "/userdata/user_config/data_collection" in cmdline:
        return "data_collection"
    if "/userdata/user_config/base" in cmdline:
        return "working"
    return "unknown"


def capture_status() -> dict[str, list[tuple[int, str]]]:
    return {name: _bin_pids(binary) for binary, name in CAPTURE_BINS}


def all_captures_running() -> bool:
    return all(bool(pids) for pids in capture_status().values())


def print_capture_processes() -> None:
    print("--- capture processes (diagnostic) ---")
    for binary, name in CAPTURE_BINS:
        found = _bin_pids(binary)
        if not found:
            print(f"WARN {name}: no process ({binary})")
            continue
        for pid, cmd in found:
            print(f"OK   {name}: pid={pid} para={capture_para_label(cmd)}")


def grab_camera(robot: GalbotRobot, name: str, sensor) -> tuple[bool, str]:
    img = None
    for _ in range(RGB_RETRIES):
        img = decode_rgb(robot.get_rgb_data(sensor))
        if img is not None:
            break
        time.sleep(RGB_RETRY_DELAY)
    if img is None:
        return False, f"FAIL {name}: empty get_rgb_data"
    height, width = img.shape[:2]
    expect = EXPECTED_WH[name]
    line = f"OK   {name}  {width}x{height}"
    if (width, height) != expect:
        line += f"  WARN expected {expect[0]}x{expect[1]}"
    return True, line


def shutdown_robot(robot: GalbotRobot | None) -> None:
    if robot is None:
        return
    try:
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
    except Exception as exc:
        print(f"WARN shutdown: {exc}")


def restart_hpu_launcher() -> int:
    print("正在重启launcher", flush=True)
    result = subprocess.run(
        ["sudo", "-S", "systemctl", "restart", "launcher"],
        input=SUDO_PASSWORD + "\n",
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"FAIL sudo systemctl restart launcher returncode={result.returncode}")
        err = (result.stderr or result.stdout or "").strip()
        if err:
            print(err)
        return EXIT_LAUNCHER
    print("launcher restart issued", flush=True)
    return EXIT_OK


def _ssh_xcu_sshpass(host: str, remote_cmd: str, timeout_s: float) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["SSHPASS"] = XCU_PASSWORD
    return subprocess.run(
        [
            "sshpass",
            "-e",
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=8",
            f"{XCU_USER}@{host}",
            remote_cmd,
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )


def _ssh_xcu_paramiko(host: str, remote_cmd: str, timeout_s: float) -> subprocess.CompletedProcess:
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            host,
            username=XCU_USER,
            password=XCU_PASSWORD,
            timeout=min(8.0, timeout_s),
            allow_agent=False,
            look_for_keys=False,
        )
        _stdin, stdout, stderr = client.exec_command(remote_cmd, timeout=timeout_s)
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        code = int(stdout.channel.recv_exit_status())
    finally:
        client.close()
    return subprocess.CompletedProcess(
        args=["ssh", f"{XCU_USER}@{host}", remote_cmd],
        returncode=code,
        stdout=out,
        stderr=err,
    )


def ssh_xcu(remote_cmd: str, timeout_s: float = 90.0) -> subprocess.CompletedProcess:
    last: subprocess.CompletedProcess | None = None
    use_sshpass = shutil.which("sshpass") is not None
    for host in XCU_HOSTS:
        try:
            if use_sshpass:
                last = _ssh_xcu_sshpass(host, remote_cmd, timeout_s)
            else:
                last = _ssh_xcu_paramiko(host, remote_cmd, timeout_s)
        except Exception as exc:
            last = subprocess.CompletedProcess(
                args=["ssh", f"{XCU_USER}@{host}", remote_cmd],
                returncode=255,
                stdout="",
                stderr=str(exc),
            )
        if last.returncode == 0:
            return last
    assert last is not None
    return last


def restart_xcu_launcher() -> int:
    print("正在重启 XCU launcher", flush=True)
    result = ssh_xcu("systemctl restart launcher", timeout_s=120.0)
    if result.returncode != 0:
        print(f"FAIL XCU systemctl restart launcher returncode={result.returncode}")
        err = (result.stderr or result.stdout or "").strip()
        if err:
            print(err)
        return EXIT_XCU
    print("XCU launcher restart issued", flush=True)
    return EXIT_OK


def wait_for_xcu_launcher(timeout_s: float = XCU_WAIT_SECONDS) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        result = ssh_xcu("systemctl is-active launcher", timeout_s=15.0)
        if result.returncode == 0 and (result.stdout or "").strip() == "active":
            print("XCU launcher is active; waiting 10s for WBC", flush=True)
            time.sleep(10.0)
            return
        time.sleep(2.0)
    print("WARN XCU launcher wait timed out; continuing anyway")


def wait_for_captures(timeout_s: float = CAPTURE_WAIT_SECONDS) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if all_captures_running():
            print("capture processes are up", flush=True)
            print_capture_processes()
            return
        time.sleep(1.0)
    print("WARN capture wait timed out; continuing anyway")
    print_capture_processes()


def reexec(args: argparse.Namespace, extra_flags: list[str]) -> None:
    script = str(Path(__file__).resolve())
    flags = list(extra_flags)
    if args.after_launcher and "--after-launcher" not in flags:
        flags.append("--after-launcher")
    if args.after_xcu and "--after-xcu" not in flags:
        flags.append("--after-xcu")
    os.execv(sys.executable, [sys.executable, script, *flags])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check VLA cameras and GalbotMotion before starting infer_server. "
            "Empty camera frames restart HPU launcher once. "
            "Motion init False restarts XCU launcher once."
        ),
    )
    parser.add_argument(
        "--after-launcher",
        action="store_true",
        help="Second pass after HPU launcher restart; do not restart HPU again",
    )
    parser.add_argument(
        "--after-xcu",
        action="store_true",
        help="Second pass after XCU launcher restart; do not restart XCU again",
    )
    return parser.parse_args()


def exit_code(camera_ok: bool, ok_motion: bool) -> int:
    if camera_ok and ok_motion:
        return EXIT_OK
    if (not camera_ok) and (not ok_motion):
        return EXIT_BOTH
    if not camera_ok:
        return EXIT_CAMERAS
    return EXIT_MOTION


def main() -> int:
    args = parse_args()

    running = infer_server_pids()
    if running:
        print("FAIL infer_server already running; stop it before this check")
        for line in running:
            print(f"     {line}")
        return EXIT_INFER_SERVER

    print_capture_processes()

    robot: GalbotRobot | None = None
    ok_robot = False
    ok_motion = False
    camera_ok = False
    try:
        robot = GalbotRobot()
        motion = GalbotMotion()
        nav = GalbotNavigation()
        ok_robot = bool(robot.init(set(CAMERAS.values())))
        ok_motion = bool(motion.init())
        ok_nav = False
        try:
            ok_nav = bool(nav.init())
        except Exception as exc:
            print(f"WARN Nav init raised: {exc}")
        print(f"Robot init: {ok_robot}, Motion init: {ok_motion}, Nav init: {ok_nav}")
        if not ok_robot:
            print("FAIL Robot init: False")
            print("     SDK 没起来，先确认本机 galbot_sdk / EmbOSA，不要开 infer_server")
            return EXIT_ROBOT

        print(f"Waiting {WARMUP_SECONDS:.0f}s for camera streams...")
        time.sleep(WARMUP_SECONDS)

        camera_ok = True
        for name, sensor in CAMERAS.items():
            passed, line = grab_camera(robot, name, sensor)
            print(line)
            camera_ok = camera_ok and passed
    finally:
        shutdown_robot(robot)

    print("--- summary ---")
    print(f"{'OK' if ok_robot and ok_motion else 'FAIL'}   robot={ok_robot}  motion={ok_motion}")
    if not ok_motion:
        if args.after_xcu:
            print("     Motion init: False → 已重启过 XCU，仍失败则平板切到可操作/SDK")
        else:
            print("     Motion init: False → 将重启 XCU launcher（不要重启 HPU）")
    if not camera_ok:
        print("     相机空帧 → 对应 capture 进程没起来或 USB/CSI 没图")

    if camera_ok and ok_motion:
        print("PASS 可以启动 infer_server / behavior_tree")
        return EXIT_OK

    if (not camera_ok) and (not args.after_launcher):
        launcher_status = restart_hpu_launcher()
        if launcher_status != EXIT_OK:
            return launcher_status
        wait_for_captures()
        reexec(args, ["--after-launcher"])

    if (not ok_motion) and (not args.after_xcu):
        xcu_status = restart_xcu_launcher()
        if xcu_status != EXIT_OK:
            return xcu_status
        wait_for_xcu_launcher()
        reexec(args, ["--after-xcu"])

    return exit_code(camera_ok, ok_motion)


if __name__ == "__main__":
    raise SystemExit(main())
