#!/usr/bin/env python3
"""Live head + wrist cameras + keyboard arm joints. Do not run with replay (SDK singleton).

Keys:
  l / r  active arm (left / right)
  1-7    active joint
  w/s    selected joint - / +
  [/]    smaller / larger step
  p      print current poses
  enter  write poses into set_init_*_arm_positions.py, README, replay_dataset.py
  q      quit
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

import cv2
import numpy as np
from galbot_sdk.g1 import ControlStatus, GalbotRobot, SensorType

HERE = Path(__file__).resolve().parent
REPLY_DIR = HERE.parent
LEFT_SCRIPT = HERE / "set_init_left_arm_positions.py"
RIGHT_SCRIPT = HERE / "set_init_right_arm_positions.py"
README = HERE / "README.md"
REPLAY_SCRIPT = REPLY_DIR / "replay_dataset.py"
LIVE_JPG = HERE / "arm_live.jpg"

CAMERAS = {
    "head_left": SensorType.HEAD_LEFT_CAMERA,
    "head_right": SensorType.HEAD_RIGHT_CAMERA,
    "left_arm": SensorType.LEFT_ARM_CAMERA,
    "right_arm": SensorType.RIGHT_ARM_CAMERA,
}

# Software limits from vla_config.yaml (arm joints only).
LIMITS = {
    "left_arm": (
        (0.0, 1.45),
        (-1.6081, -0.27),
        (-0.77, 0.74),
        (-2.38, -1.248),
        (-1.16, 0.81),
        (-0.64, 0.7354),
        (-1.26, 1.5382),
    ),
    "right_arm": (
        (-1.83, 0.11),
        (0.29, 1.6081),
        (-0.84, 1.17),
        (1.178, 2.47),
        (-0.76, 1.04),
        (-0.7354, 0.8227),
        (-1.5382, 0.61),
    ),
}
DEFAULTS = {
    "left_arm": [0.758482, -0.625097, -0.105102, -1.601858, -0.266145, 0.051053, 1.184185],
    "right_arm": [-1.149383, 0.602231, 0.748631, 1.483263, 0.356627, -0.218040, -1.422983],
}
HTTP_PORT = 8767
ARM_MAX_SPEED = 0.15
TILE_H = 360
TILE_W = 480

_jpeg_lock = threading.Lock()
_jpeg_bytes = b""


def decode_bgr(compressed):
    if not compressed or "data" not in compressed:
        return None
    img = cv2.imdecode(np.frombuffer(compressed["data"], np.uint8), cv2.IMREAD_COLOR)
    return img


def clip_pose(group: str, pose):
    out = []
    for i, value in enumerate(pose):
        lo, hi = LIMITS[group][i]
        out.append(min(max(float(value), lo), hi))
    return out


def read_current_arm(robot, group: str):
    try:
        pos = robot.get_joint_positions([group], [])
        if pos and len(pos) >= 7:
            return [float(v) for v in pos[:7]]
    except Exception as exc:
        print(f"warning: get_joint_positions({group}) failed: {exc}")
    return list(DEFAULTS[group])


def apply_arm(robot, group: str, pose):
    status = robot.set_joint_positions(list(pose), [group], [], False, ARM_MAX_SPEED, 5)
    if status != ControlStatus.SUCCESS:
        print(f"warning: set_joint_positions {group} status={status}")


def grab_views(robot):
    return {name: decode_bgr(robot.get_rgb_data(sensor)) for name, sensor in CAMERAS.items()}


def make_tile(name, img, active=False):
    canvas = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    if img is None:
        cv2.putText(
            canvas,
            f"{name}: no frame",
            (20, TILE_H // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    else:
        scale = min(TILE_W / img.shape[1], TILE_H / img.shape[0])
        nw = max(1, int(img.shape[1] * scale))
        nh = max(1, int(img.shape[0] * scale))
        resized = cv2.resize(img, (nw, nh))
        y0 = (TILE_H - nh) // 2
        x0 = (TILE_W - nw) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    color = (0, 255, 255) if active else (0, 255, 0)
    cv2.putText(canvas, name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if active:
        cv2.rectangle(canvas, (2, 2), (TILE_W - 3, TILE_H - 3), (0, 255, 255), 2)
    return canvas


def fmt_short(pose):
    return "[" + ",".join(f"{v:.2f}" for v in pose) + "]"


def compose(views, poses, active, joint_idx, step):
    top = np.hstack(
        [
            make_tile("head_left", views.get("head_left")),
            make_tile("head_right", views.get("head_right")),
        ]
    )
    bottom = np.hstack(
        [
            make_tile("left_arm", views.get("left_arm"), active=active == "left_arm"),
            make_tile("right_arm", views.get("right_arm"), active=active == "right_arm"),
        ]
    )
    canvas = np.vstack([top, bottom])
    hud_h = 64
    hud = np.zeros((hud_h, canvas.shape[1], 3), dtype=np.uint8)
    pose = poses[active]
    line1 = (
        f"{active} j{joint_idx + 1}={pose[joint_idx]:.3f}  step={step:.4f}   "
        "l/r arm  1-7 joint  w/s nudge  enter=save  q=quit"
    )
    line2 = f"L {fmt_short(poses['left_arm'])}   R {fmt_short(poses['right_arm'])}"
    cv2.putText(hud, line1, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(hud, line2, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    return np.vstack([hud, canvas])


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
                b"<p style='padding:8px'>G1 arm live &nbsp; l/r arm &nbsp; 1-7 joint &nbsp; w/s nudge &nbsp; enter save</p>"
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


def fmt_pose(pose):
    return ", ".join(f"{v:.6f}" for v in pose)


def _replace_joint_pos(path: Path, pose):
    text = path.read_text(encoding="utf-8")
    text, n = re.subn(
        r"^joint_pos = \[[^\]]+\]",
        f"joint_pos = [{fmt_pose(pose)}]",
        text,
        count=1,
        flags=re.M,
    )
    if n:
        path.write_text(text, encoding="utf-8")
    return n


def write_pose_files(poses):
    n_left = _replace_joint_pos(LEFT_SCRIPT, poses["left_arm"])
    n_right = _replace_joint_pos(RIGHT_SCRIPT, poses["right_arm"])
    if README.exists():
        readme = README.read_text(encoding="utf-8")
        left_short = "[" + ", ".join(f"{v:.4f}" for v in poses["left_arm"]) + "]"
        right_short = "[" + ", ".join(f"{v:.4f}" for v in poses["right_arm"]) + "]"
        readme = re.sub(
            r"(`set_init_left_arm_positions.py`\s+\| `left_arm` \(7\)\s+\| `)[^`]+(`)",
            rf"\1{left_short}\2",
            readme,
            count=1,
        )
        readme = re.sub(
            r"(`set_init_right_arm_positions.py`\s+\| `right_arm` \(7\)\s+\| `)[^`]+(`)",
            rf"\1{right_short}\2",
            readme,
            count=1,
        )
        README.write_text(readme, encoding="utf-8")
    if REPLAY_SCRIPT.exists():
        replay = REPLAY_SCRIPT.read_text(encoding="utf-8")
        replay, _ = re.subn(
            r"INIT_LEFT_ARM = \[[^\]]+\]",
            f"INIT_LEFT_ARM = [{fmt_pose(poses['left_arm'])}]",
            replay,
            count=1,
        )
        replay, _ = re.subn(
            r"INIT_RIGHT_ARM = \[[^\]]+\]",
            f"INIT_RIGHT_ARM = [{fmt_pose(poses['right_arm'])}]",
            replay,
            count=1,
        )
        REPLAY_SCRIPT.write_text(replay, encoding="utf-8")
    print(
        f"saved L [{fmt_pose(poses['left_arm'])}] R [{fmt_pose(poses['right_arm'])}] "
        f"(left_script={n_left} right_script={n_right})"
    )


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


def handle_key(key: int, robot, poses, active, joint_idx, step):
    if key in (ord("q"), 27):
        return poses, active, joint_idx, step, True
    if key in (ord("l"), ord("L")):
        active = "left_arm"
        print(f"active={active} j{joint_idx + 1}")
        return poses, active, joint_idx, step, False
    if key in (ord("r"), ord("R")):
        active = "right_arm"
        print(f"active={active} j{joint_idx + 1}")
        return poses, active, joint_idx, step, False
    if ord("1") <= key <= ord("7"):
        joint_idx = key - ord("1")
        print(f"active={active} j{joint_idx + 1}={poses[active][joint_idx]:.4f}")
        return poses, active, joint_idx, step, False
    if key in (ord("w"), ord("W"), ord("s"), ord("S")):
        delta = -step if key in (ord("w"), ord("W")) else step
        nxt = list(poses[active])
        nxt[joint_idx] += delta
        poses[active] = clip_pose(active, nxt)
        apply_arm(robot, active, poses[active])
        print(f"{active} j{joint_idx + 1} -> {poses[active][joint_idx]:.4f}  {poses[active]}")
        return poses, active, joint_idx, step, False
    if key == ord("["):
        step = max(0.001, step * 0.5)
        print(f"step={step:.4f}")
        return poses, active, joint_idx, step, False
    if key == ord("]"):
        step = min(0.05, step * 2.0)
        print(f"step={step:.4f}")
        return poses, active, joint_idx, step, False
    if key in (ord("p"), ord("P")):
        print(f"left  {poses['left_arm']}")
        print(f"right {poses['right_arm']}")
        return poses, active, joint_idx, step, False
    if key in (13, 10):
        write_pose_files(poses)
        return poses, active, joint_idx, step, False
    return poses, active, joint_idx, step, False


def has_display() -> bool:
    return bool(os.environ.get("DISPLAY"))


def main():
    robot = GalbotRobot()
    if not robot.init(set(CAMERAS.values())):
        raise RuntimeError("GalbotRobot init failed")
    print("Initialization succeeded")
    print("Waiting 5s for head/wrist cameras...")
    time.sleep(5)

    poses = {
        "left_arm": clip_pose("left_arm", read_current_arm(robot, "left_arm")),
        "right_arm": clip_pose("right_arm", read_current_arm(robot, "right_arm")),
    }
    active = "right_arm"
    joint_idx = 3
    step = 0.01
    start_http()
    print(f"Browser live view: http://127.0.0.1:{HTTP_PORT}/  (or this machine's IP)")
    print(f"Also writing {LIVE_JPG}")
    print("l/r=arm  1-7=joint  w/s=nudge  [/]=step  enter=save  q=quit")
    print("Around: keep workspace clear; e-stop ready. Arms move slowly.")
    print(f"start L {poses['left_arm']}")
    print(f"start R {poses['right_arm']}")

    show_window = has_display()
    if show_window:
        cv2.namedWindow("g1-arm", cv2.WINDOW_NORMAL)

    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while True:
            views = grab_views(robot)
            canvas = compose(views, poses, active, joint_idx, step)
            publish_jpeg(canvas)

            key = -1
            if show_window:
                cv2.imshow("g1-arm", canvas)
                key = cv2.waitKey(1) & 0xFF
            if key in (-1, 255):
                key = read_stdin_key(0.03)

            poses, active, joint_idx, step, done = handle_key(
                key, robot, poses, active, joint_idx, step
            )
            if done:
                break
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        if show_window:
            cv2.destroyAllWindows()
        print(f"final L {poses['left_arm']}")
        print(f"final R {poses['right_arm']}")
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
        print("Resources released")


if __name__ == "__main__":
    main()
