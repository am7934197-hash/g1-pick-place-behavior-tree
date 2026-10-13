"""Capture one frame from each VLA camera and save PNGs for viewing."""

import os
import time

import cv2
import numpy as np
from galbot_sdk.g1 import GalbotRobot, SensorType

CAMERAS = {
    "head_left": SensorType.HEAD_LEFT_CAMERA,
    "head_right": SensorType.HEAD_RIGHT_CAMERA,
    "left_arm": SensorType.LEFT_ARM_CAMERA,
    "right_arm": SensorType.RIGHT_ARM_CAMERA,
}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_preview")


def decode_rgb(compressed):
    if not compressed:
        return None
    nparr = np.frombuffer(compressed["data"], np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    robot = GalbotRobot()
    if not robot.init(set(CAMERAS.values())):
        raise RuntimeError("GalbotRobot initialization failed")
    print("Initialization succeeded")
    print("Waiting 5s for camera streams...")
    time.sleep(5)

    for name, sensor in CAMERAS.items():
        img = None
        for attempt in range(8):
            data = robot.get_rgb_data(sensor)
            img = decode_rgb(data)
            if img is not None:
                break
            time.sleep(0.3)
        path = os.path.join(OUT_DIR, f"{name}.png")
        if img is None:
            print(f"{name}: no frame")
            continue
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"{name}: {img.shape} -> {path}")

    robot.request_shutdown()
    robot.wait_for_shutdown()
    robot.destroy()
    print(f"Saved under {OUT_DIR}")
    print("Open the four PNG files in Cursor, or: eog camera_preview/*.png")


if __name__ == "__main__":
    main()
