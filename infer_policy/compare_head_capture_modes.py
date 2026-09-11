#!/usr/bin/env python3
"""Switch the REAL front_head_camera_capture driver and compare SDK frames.

This does not undistort or crop in Python. It restarts the vendor binary with
the same para_dir DATA_COLLECTION_MODE uses.

  working          /userdata/user_config/base            expected 1280x960
  data_collection  /userdata/user_config/data_collection    expected 640x480

  python3 compare_head_capture_modes.py --compare
  python3 compare_head_capture_modes.py --to working
  python3 compare_head_capture_modes.py --to data_collection
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from head_capture_switch import CAPTURE_BIN, EXPECTED_WH, switch_to

OUT_DIR = Path(__file__).resolve().parent / "logs" / "head_fov_preview"
SCRIPT = Path(__file__).resolve()


def grab_once(out_path: Path, expect_wh: tuple[int, int], timeout_s: float = 20.0) -> None:
    """Fresh process so SDK does not keep the previous capture subscription."""
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--grab-once",
        str(out_path),
        "--expect-w",
        str(expect_wh[0]),
        "--expect-h",
        str(expect_wh[1]),
        "--grab-timeout",
        str(int(timeout_s)),
    ]
    print("grab", " ".join(cmd))
    subprocess.check_call(cmd)


def grab_once_main(out_path: Path, expect_w: int, expect_h: int, timeout_s: float) -> int:
    import cv2
    import numpy as np
    from galbot_sdk.g1 import GalbotRobot, SensorType

    robot = GalbotRobot()
    ok = robot.init({SensorType.HEAD_LEFT_CAMERA})
    print(f"SDK init={ok}")
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        data = robot.get_rgb_data(SensorType.HEAD_LEFT_CAMERA)
        raw = (data or {}).get("data") or b""
        if raw:
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                last = img
                h, w = img.shape[:2]
                print(f"  got {w}x{h}")
                if w == expect_w and h == expect_h:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_path), img)
                    print(f"saved {out_path}")
                    os._exit(0)
        time.sleep(0.3)
    raise SystemExit(
        f"did not get {expect_w}x{expect_h}; last={None if last is None else last.shape[1::-1]}"
    )


def letterbox(img, tw: int, th: int):
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def label(img, text: str):
    import cv2

    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def compare() -> Path:
    import cv2
    import numpy as np

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    working_path = OUT_DIR / "driver_working_head_left.jpg"
    dc_path = OUT_DIR / "driver_datacollection_head_left.jpg"

    switch_to("working")
    grab_once(working_path, EXPECTED_WH["working"])

    switch_to("data_collection")
    grab_once(dc_path, EXPECTED_WH["data_collection"])

    switch_to("working")

    working = cv2.imread(str(working_path))
    dc = cv2.imread(str(dc_path))
    if working is None or dc is None:
        raise RuntimeError("missing saved frames")
    pair = np.hstack(
        [
            label(letterbox(working, 640, 480), f"WORKING driver  {working.shape[1]}x{working.shape[0]}"),
            label(letterbox(dc, 640, 480), f"DATA-COLLECTION driver  {dc.shape[1]}x{dc.shape[0]}"),
        ]
    )
    out = OUT_DIR / "compare_real_drivers.jpg"
    cv2.imwrite(str(out), pair, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"wrote {out}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--to", choices=("working", "data_collection"))
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--grab-once", type=Path)
    parser.add_argument("--expect-w", type=int, default=0)
    parser.add_argument("--expect-h", type=int, default=0)
    parser.add_argument("--grab-timeout", type=int, default=20)
    args = parser.parse_args()

    if args.grab_once:
        return grab_once_main(args.grab_once, args.expect_w, args.expect_h, args.grab_timeout)
    if not os.path.isfile(CAPTURE_BIN):
        print(f"missing {CAPTURE_BIN}", file=sys.stderr)
        return 2
    if args.to and not args.compare:
        switch_to(args.to)
        return 0
    if not args.compare:
        parser.error("pass --compare or --to working|data_collection")
    compare()
    return 0


if __name__ == "__main__":
    sys.exit(main())
