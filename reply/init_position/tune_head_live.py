#!/usr/bin/env python3
"""Live head cameras + keyboard pitch/yaw. Do not run with replay (SDK singleton).

Keys:
  w/s  pitch  (head_joint2, smaller = look down)
  a/d  yaw    (head_joint1)
  [/]  smaller / larger step
  p    print current pose
  enter  write pose into set_init_head_positions.py, README, replay_dataset.py
  q    quit
"""

from __future__ import annotations

import os
import re
import select
import sys
import termios
import threading
import time
import tty
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from galbot_sdk.g1 import ControlStatus, GalbotRobot, SensorType

HERE = Path(__file__).resolve().parent
REPLY_DIR = HERE.parent
HEAD_SCRIPT = HERE / "set_init_head_positions.py"
README = HERE / "README.md"
REPLAY_SCRIPT = REPLY_DIR / "replay_dataset.py"
LIVE_JPG = HERE / "head_live.jpg"

CAMERAS = {
    "head_left": SensorType.HEAD_LEFT_CAMERA,
    "head_right": SensorType.HEAD_RIGHT_CAMERA,
}
J1_LIMIT = (-1.5208, 1.5208)
J2_LIMIT = (-0.2143, 0.4936)
DEFAULT_POSE = [-0.074961, 0.30]
HTTP_PORT = 8765

_jpeg_lock = threading.Lock()
_jpeg_bytes = b""


def decode_bgr(compressed):
    if not compressed or "data" not in compressed:
        return None
    img = cv2.imdecode(np.frombuffer(compressed["data"], np.uint8), cv2.IMREAD_COLOR)
    return img


def clip_pose(j1: float, j2: float):
    return [
        min(max(j1, J1_LIMIT[0]), J1_LIMIT[1]),
        min(max(j2, J2_LIMIT[0]), J2_LIMIT[1]),
    ]


def read_current_head(robot):
    try:
        pos = robot.get_joint_positions(["head"], [])
        if pos and len(pos) >= 2:
            return [float(pos[0]), float(pos[1])]
    except Exception as exc:
        print(f"warning: get_joint_positions failed: {exc}")
    return list(DEFAULT_POSE)


def apply_head(robot, pose, max_speed=0.4):
    status = robot.set_joint_positions(list(pose), ["head"], [], False, max_speed, 5)
    if status != ControlStatus.SUCCESS:
        print(f"warning: set_joint_positions status={status}")


def grab_pair(robot):
    left = decode_bgr(robot.get_rgb_data(CAMERAS["head_left"]))
    right = decode_bgr(robot.get_rgb_data(CAMERAS["head_right"]))
    return left, right


def compose(left, right, pose, step):
    h = 480
    tiles = []
    for name, img in (("head_left", left), ("head_right", right)):
        if img is None:
            tile = np.zeros((h, int(h * 4 / 3), 3), dtype=np.uint8)
            cv2.putText(tile, f"{name}: no frame", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            tile = cv2.resize(img, (int(h * img.shape[1] / img.shape[0]), h))
        tiles.append(tile)
    canvas = np.hstack(tiles)
    hud = (
        f"j1(yaw)={pose[0]:.3f}  j2(pitch)={pose[1]:.3f}  step={step:.4f}   "
        "w/s pitch  a/d yaw  enter=save  q=quit"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(canvas, hud, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return canvas


def publish_jpeg(bgr):
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return
    data = buf.tobytes()
    with _jpeg_lock:
        global _jpeg_bytes
        _jpeg_bytes = data
    LIVE_JPG.write_bytes(data)


class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"<html><body style='margin:0;background:#111;color:#eee;font-family:sans-serif'>"
                b"<p style='padding:8px'>G1 head live &nbsp; w/s pitch &nbsp; a/d yaw &nbsp; enter save</p>"
                b"<img src='/stream' style='width:100%'/>"
                b"</body></html>"
            )
            return
        if self.path != "/stream":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with _jpeg_lock:
                    data = _jpeg_bytes
                if data:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                time.sleep(0.05)
        except BrokenPipeError:
            return


def start_http():
    server = HTTPServer(("0.0.0.0", HTTP_PORT), MJPEGHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def write_pose_files(pose):
    j1, j2 = pose
    text = HEAD_SCRIPT.read_text(encoding="utf-8")
    text, n = re.subn(
        r"joint_pos = \[[^\]]+\]",
        f"joint_pos = [{j1:.6f}, {j2:.6f}]",
        text,
        count=1,
    )
    if n:
        HEAD_SCRIPT.write_text(text, encoding="utf-8")
    if README.exists():
        readme = README.read_text(encoding="utf-8")
        readme = re.sub(
            r"(`set_init_head_positions.py` \| `head` \(2\) \| `)[^`]+(`)",
            rf"\1[{j1:.4f}, {j2:.4f}]\2",
            readme,
            count=1,
        )
        README.write_text(readme, encoding="utf-8")
    if REPLAY_SCRIPT.exists():
        replay = REPLAY_SCRIPT.read_text(encoding="utf-8")
        replay, n = re.subn(
            r"INIT_HEAD = \[[^\]]+\]",
            f"INIT_HEAD = [{j1:.6f}, {j2:.6f}]",
            replay,
            count=1,
        )
        if n:
            REPLAY_SCRIPT.write_text(replay, encoding="utf-8")
    print(f"saved pose [{j1:.6f}, {j2:.6f}] to head script / README / replay restore")


def read_stdin_key(timeout_s: float = 0.03) -> int:
    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not ready:
        return -1
    ch = sys.stdin.read(1)
    if ch == "\x03":
        raise KeyboardInterrupt
    if ch in ("\n", "\r"):
        return 13
    if ch == "\x1b":
        return 27
    return ord(ch)


def handle_key(key: int, robot, pose, step):
    if key in (ord("q"), 27):
        return pose, step, True
    if key in (ord("w"), ord("W")):
        pose = clip_pose(pose[0], pose[1] - step)
        apply_head(robot, pose)
        print(f"pitch down -> {pose}")
    elif key in (ord("s"), ord("S")):
        pose = clip_pose(pose[0], pose[1] + step)
        apply_head(robot, pose)
        print(f"pitch up -> {pose}")
    elif key in (ord("a"), ord("A")):
        pose = clip_pose(pose[0] - step, pose[1])
        apply_head(robot, pose)
        print(f"yaw left -> {pose}")
    elif key in (ord("d"), ord("D")):
        pose = clip_pose(pose[0] + step, pose[1])
        apply_head(robot, pose)
        print(f"yaw right -> {pose}")
    elif key == ord("["):
        step = max(0.001, step * 0.5)
        print(f"step={step:.4f}")
    elif key == ord("]"):
        step = min(0.1, step * 2.0)
        print(f"step={step:.4f}")
    elif key in (ord("p"), ord("P")):
        print(f"current head {pose}")
    elif key in (13, 10):
        write_pose_files(pose)
    return pose, step, False


def main():
    robot = GalbotRobot()
    if not robot.init(set(CAMERAS.values())):
        raise RuntimeError("GalbotRobot init failed")
    print("Initialization succeeded")
    print("Waiting 5s for head cameras...")
    time.sleep(5)

    pose = read_current_head(robot)
    pose = clip_pose(*pose)
    apply_head(robot, pose)
    step = 0.005
    start_http()
    print(f"Browser live view: http://127.0.0.1:{HTTP_PORT}/  (or this machine's IP)")
    print(f"Also writing {LIVE_JPG}")
    print("w/s=pitch  a/d=yaw  [/]=step  enter=save  q=quit")

    show_window = has_display()
    if show_window:
        cv2.namedWindow("g1-head", cv2.WINDOW_NORMAL)

    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while True:
            left, right = grab_pair(robot)
            canvas = compose(left, right, pose, step)
            publish_jpeg(canvas)

            key = -1
            if show_window:
                cv2.imshow("g1-head", canvas)
                key = cv2.waitKey(1) & 0xFF
            if key in (-1, 255):
                key = read_stdin_key(0.03)

            pose, step, done = handle_key(key, robot, pose, step)
            if done:
                break
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        if show_window:
            cv2.destroyAllWindows()
        print(f"final pose {pose}")
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
        print("Resources released")


def has_display() -> bool:
    return bool(os.environ.get("DISPLAY"))


if __name__ == "__main__":
    main()
