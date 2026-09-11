#!/usr/bin/env python3
"""Grab working-mode frames and compare via_640 vs crop_only preprocess.

Does not switch capture modes. Writes labeled contact sheets under
logs/preprocess_compare/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from emulate_data_collection import emulate_fov
from image_preprocessor import (
    ImagePreprocessor,
    count_letterbox_black_bar_rows,
    crop_to_training_aspect,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "logs" / "preprocess_compare"
CAMERAS = ("head_left", "head_right", "left_arm", "right_arm")
WRIST_CAMERAS = ("left_arm", "right_arm")


def _letterbox(img: np.ndarray) -> np.ndarray:
    pp = ImagePreprocessor({
        "enabled": True,
        "target_size": [224, 224],
        "keep_aspect_ratio": True,
        "crop_to_training_aspect": False,
    })
    return pp._resize(img)


def _fov_crop(name: str, native: np.ndarray) -> np.ndarray:
    zoom = 1.0
    img = crop_to_training_aspect(native) if name in WRIST_CAMERAS else native
    return emulate_fov(img, zoom, (640, 480), "crop_only")


def _scheme_frames(name: str, native: np.ndarray) -> dict[str, np.ndarray]:
    zoom = 1.0
    img = crop_to_training_aspect(native) if name in WRIST_CAMERAS else native
    cropped = _fov_crop(name, native)
    via = emulate_fov(img, zoom, (640, 480), "via_640")
    only = emulate_fov(img, zoom, (640, 480), "crop_only")
    a224 = _letterbox(via)
    b224 = _letterbox(only)
    return {
        "native": native,
        "crop": cropped,
        "via_640": via,
        "a224": a224,
        "b224": b224,
    }


def _fit(img: np.ndarray, tw: int, th: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _label(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22 * len(lines) + 8), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(
            out, text, (8, 22 * (i + 1)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
    return out


def _panel(img: np.ndarray, title: str, tw: int = 480, th: int = 400) -> np.ndarray:
    h, w = img.shape[:2]
    bars = count_letterbox_black_bar_rows(img)
    fitted = _fit(img, tw, th - 52)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    canvas[52:, :] = fitted
    return _label(canvas, [title, f"{w}x{h}  bars {bars[0]}/{bars[1]}"])


def contact_sheet(name: str, frames: dict[str, np.ndarray]) -> np.ndarray:
    a224 = frames["a224"]
    b224 = frames["b224"]
    diff = np.abs(a224.astype(np.int16) - b224.astype(np.int16)).astype(np.uint8)
    diff3 = cv2.resize(diff, (224 * 3, 224 * 3), interpolation=cv2.INTER_NEAREST)
    a3 = cv2.resize(a224, (224 * 3, 224 * 3), interpolation=cv2.INTER_NEAREST)
    b3 = cv2.resize(b224, (224 * 3, 224 * 3), interpolation=cv2.INTER_NEAREST)
    panels = [
        _panel(frames["native"], f"{name} native"),
        _panel(frames["crop"], "crop only"),
        _panel(frames["via_640"], "A via_640"),
        _panel(a3, "A 224 x3"),
        _panel(b3, "B crop_only 224 x3"),
        _panel(diff3, f"|A-B| max={int(diff.max())} x3"),
    ]
    return np.hstack(panels)


def decode_rgb(compressed) -> np.ndarray | None:
    if not compressed:
        return None
    raw = compressed.get("data")
    if not raw:
        return None
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def grab_sdk(names: tuple[str, ...], warmup_s: float = 5.0) -> dict[str, np.ndarray]:
    from galbot_sdk.g1 import GalbotRobot, SensorType

    sensor_map = {
        "head_left": SensorType.HEAD_LEFT_CAMERA,
        "head_right": SensorType.HEAD_RIGHT_CAMERA,
        "left_arm": SensorType.LEFT_ARM_CAMERA,
        "right_arm": SensorType.RIGHT_ARM_CAMERA,
    }
    wanted = {name: sensor_map[name] for name in names if name in sensor_map}
    robot = GalbotRobot()
    frames: dict[str, np.ndarray] = {}
    try:
        if not robot.init(set(wanted.values())):
            raise RuntimeError("GalbotRobot.init failed")
        print(f"Waiting {warmup_s:.0f}s for camera streams...")
        time.sleep(warmup_s)
        for name, sensor in wanted.items():
            img = None
            for _ in range(8):
                img = decode_rgb(robot.get_rgb_data(sensor))
                if img is not None:
                    break
                time.sleep(0.3)
            if img is None:
                print(f"FAIL {name}: empty frame")
                continue
            print(f"OK   {name} {img.shape[1]}x{img.shape[0]}")
            frames[name] = img
    finally:
        try:
            robot.request_shutdown()
            robot.wait_for_shutdown()
            robot.destroy()
        except Exception as exc:
            print(f"WARN shutdown: {exc}")
    return frames


def load_dir(directory: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    frames: dict[str, np.ndarray] = {}
    for name in names:
        path = None
        for candidate in (
            directory / f"{name}.png",
            directory / f"{name}.jpg",
            directory / f"{name}.jpeg",
            directory / f"{name}_native.png",
            directory / f"{name}_native.jpg",
        ):
            if candidate.is_file():
                path = candidate
                break
        if path is None:
            print(f"FAIL {name}: no file in {directory}")
            continue
        bgr = cv2.imread(str(path))
        if bgr is None:
            print(f"FAIL {name}: could not read {path}")
            continue
        frames[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        print(f"OK   {name} {frames[name].shape[1]}x{frames[name].shape[0]} from {path}")
    return frames


def write_sheets(frames: dict[str, np.ndarray], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, native in frames.items():
        h, w = native.shape[:2]
        if name.startswith("head") and (w, h) != (1280, 960):
            print(f"WARN {name} is {w}x{h}, expected working 1280x960; A/B may look identical")
        if name in WRIST_CAMERAS and (w, h) != (1280, 720):
            print(f"WARN {name} is {w}x{h}, expected working 1280x720")
        parts = _scheme_frames(name, native)
        sheet_bgr = cv2.cvtColor(contact_sheet(name, parts), cv2.COLOR_RGB2BGR)
        path = out_dir / f"{name}_compare.jpg"
        cv2.imwrite(str(path), sheet_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        native_path = out_dir / f"{name}_native.png"
        cv2.imwrite(str(native_path), cv2.cvtColor(native, cv2.COLOR_RGB2BGR))
        print(f"wrote {path}")
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-dir", type=Path, help="directory of name.png/jpg working frames")
    parser.add_argument("--cameras", nargs="+", default=list(CAMERAS))
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--no-sdk", action="store_true")
    args = parser.parse_args()
    if cv2 is None:
        print("opencv is required", file=sys.stderr)
        return 2
    names = tuple(args.cameras)
    frames: dict[str, np.ndarray] = {}
    if args.from_dir:
        frames = load_dir(args.from_dir, names)
    elif not args.no_sdk:
        try:
            frames = grab_sdk(names)
        except Exception as exc:
            print(f"SDK grab failed: {exc}")
    if not frames:
        print("no working-mode frames; not writing synthetic images")
        return 1
    write_sheets(frames, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
