"""Classify ready_224 wrist geometry as A/B/C/D using actual video frames.

A: 1280x720 center-crop 960x720 then letterbox -> ~28px top/bottom bars, content 168x224
B: 1280x720 native letterbox -> ~49px top/bottom bars, content 126x224
C: stretch to 224x224 -> ~0 bars
D: other (including stretch-to-480x640 then letterbox, which also has ~28px bars)

A vs D both have 28px bars; FOV differs. This script therefore also compares
horizontal content against a live/native 720x1280 frame when provided.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from image_preprocessor import (
    ImagePreprocessor,
    count_letterbox_black_bar_rows,
    crop_to_training_aspect,
)
from pi05_image_preprocess import preprocess_images_pi05, resize_with_pad_uint8

try:
    from PIL import Image
except ImportError:
    Image = None


def _bars(img: np.ndarray) -> Tuple[int, int]:
    return count_letterbox_black_bar_rows(img)


def make_native_wrist(h: int = 720, w: int = 1280) -> np.ndarray:
    """Synthetic 16:9 wrist frame with left/right FOV markers.

    A 4:3 center-crop drops the outer 160px on each side; those markers
    disappear under path A/D but remain under path B.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, :] = 40
    img[:, :80, 0] = 255
    img[:, -80:, 1] = 255
    img[h // 2 - 20:h // 2 + 20, w // 2 - 40:w // 2 + 40, 2] = 255
    return img


def make_native_head(h: int = 960, w: int = 1280) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, :] = 50
    img[:, :40, 0] = 255
    img[:, -40:, 1] = 255
    return img


def apply_mode(native: np.ndarray, mode: str) -> np.ndarray:
    pp = ImagePreprocessor({
        "enabled": True,
        "target_size": [224, 224],
        "keep_aspect_ratio": True,
        "crop_to_training_aspect": mode == "A",
        "training_hw": [480, 640],
    })
    if mode == "A":
        return pp._resize(native)
    if mode == "B":
        pp._crop_to_training = False
        return pp._resize(native)
    if mode == "C":
        pp._keep_ratio = False
        pp._crop_to_training = False
        return pp._resize(native)
    if mode == "D":
        # Stretch native to 480x640 then letterbox. Common cv2.resize mistake/path.
        if native.shape[0] != 480 or native.shape[1] != 640:
            import cv2
            stretched = cv2.resize(native, (640, 480), interpolation=cv2.INTER_LINEAR)
        else:
            stretched = native
        pp._crop_to_training = False
        return pp._resize(stretched)
    raise ValueError(mode)


def classify_from_bars(top: int, bottom: int) -> str:
    bars = (top + bottom) / 2.0
    if bars <= 4:
        return "C"
    # H.264/yuv420 decoding softens the zero/content boundary by several rows;
    # ready_224's nominal 28px bars often measure 22-28px at threshold=2.
    if 18 <= bars <= 36:
        return "A_or_D_4_3_letterbox"
    if 44 <= bars <= 56:
        return "B"
    return "D_other"


def load_ready224_frame(dataset_dir: str, camera: str, episode: int = 0) -> Optional[np.ndarray]:
    if Image is None:
        return None
    videos = os.path.join(dataset_dir, "videos")
    if not os.path.isdir(videos):
        return None
    # LeRobot layout varies; search for mp4 containing the camera key.
    matches = []
    for root, _, files in os.walk(videos):
        for name in files:
            full = os.path.join(root, name)
            # LeRobot stores the camera name in the parent directory and uses
            # generic file-000.mp4 names, so matching only the basename finds
            # nothing.
            if name.endswith(".mp4") and camera in full:
                matches.append(full)
    if not matches:
        return None
    matches.sort()
    path = matches[min(episode, len(matches) - 1)]
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def save_side_by_side(path: str, left: np.ndarray, right: np.ndarray, blend: Optional[np.ndarray] = None) -> None:
    if Image is None:
        return
    tiles = [left, right]
    if blend is not None:
        tiles.append(blend)
    canvas = np.concatenate(tiles, axis=1)
    Image.fromarray(canvas).save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=os.environ.get("TRAIN_DATASET_DIR", ""))
    parser.add_argument("--native-dir", default="", help="Directory with native_{cam}.png from a robot dump")
    parser.add_argument("--synthetic", action="store_true", help="Generate 720x1280 FOV-marker natives")
    parser.add_argument("--out", default="logs/wrist_geometry")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    report: Dict[str, Any] = {"dataset": args.dataset or None, "cameras": {}}
    for cam in ("head_left", "left_arm", "right_arm"):
        train = load_ready224_frame(args.dataset, cam) if args.dataset else None
        native_path = os.path.join(args.native_dir, f"native_{cam}.png") if args.native_dir else ""
        native = np.array(Image.open(native_path)) if native_path and os.path.isfile(native_path) else None
        if native is None and args.synthetic:
            native = make_native_head() if cam == "head_left" else make_native_wrist()
            if Image is not None:
                Image.fromarray(native).save(os.path.join(args.out, f"native_{cam}.png"))
        cam_rep: Dict[str, Any] = {}
        if train is not None:
            top, bottom = _bars(train)
            cam_rep["train_shape"] = list(train.shape)
            cam_rep["train_bars"] = {"top": top, "bottom": bottom}
            cam_rep["train_class"] = classify_from_bars(top, bottom)
            cam_rep["train_preprocess"] = preprocess_images_pi05({cam: train})["cameras"][cam]
            Image.fromarray(train).save(os.path.join(args.out, f"train_{cam}.png"))
        if native is not None:
            modes = {}
            for mode in ("A", "B", "C", "D"):
                out = apply_mode(native, mode)
                top, bottom = _bars(out)
                modes[mode] = {
                    "bars": {"top": top, "bottom": bottom},
                    "shape": list(out.shape),
                    "preprocess": preprocess_images_pi05({cam: out})["cameras"][cam],
                    "left_strip_r": float(out[:, :16, 0].mean()),
                    "right_strip_g": float(out[:, -16:, 1].mean()),
                }
                Image.fromarray(out).save(os.path.join(args.out, f"robot_{mode}_{cam}.png"))
                if train is not None and train.shape == out.shape:
                    blend = (0.5 * train.astype(np.float32) + 0.5 * out.astype(np.float32)).astype(np.uint8)
                    save_side_by_side(
                        os.path.join(args.out, f"compare_{mode}_{cam}.png"),
                        train, out, blend,
                    )
            cam_rep["native_shape"] = list(native.shape)
            cam_rep["modes"] = modes
            # FOV marker check: left 80px is red. After A-crop 1280→960 the
            # left 160px are gone, so the 80px marker cannot survive.
            if native.shape[1] == 1280 and native.shape[0] == 720:
                cam_rep["fov_marker"] = {
                    "A_keeps_left_red": bool(modes["A"]["left_strip_r"] > 80),
                    "B_keeps_left_red": bool(modes["B"]["left_strip_r"] > 80),
                    "C_keeps_left_red": bool(modes["C"]["left_strip_r"] > 80),
                    "D_keeps_left_red": bool(modes["D"]["left_strip_r"] > 80),
                    "note": "A drops 25% horizontal FOV. D keeps sides but squeezes 16:9 into 4:3. Bars cannot separate A vs D.",
                }
        report["cameras"][cam] = cam_rep

    # Dataset-level verdict uses wrist cameras.
    wrist_classes = [
        report["cameras"][cam].get("train_class")
        for cam in ("left_arm", "right_arm")
        if report["cameras"][cam].get("train_class")
    ]
    if wrist_classes and all(c == wrist_classes[0] for c in wrist_classes):
        report["wrist_verdict"] = wrist_classes[0]
    elif not args.dataset or not os.path.isdir(args.dataset):
        report["wrist_verdict"] = "DATASET_UNAVAILABLE"
        report["note"] = (
            "ready_224 is not on this machine, so do not infer production geometry "
            "from the public decoder alone. The pinned production contract uses "
            "4:3 content (mode A for a 16:9 wrist) based on the training origin/ready_224 evidence."
        )
    else:
        report["wrist_verdict"] = "MIXED_OR_UNKNOWN"
    with open(os.path.join(args.out, "wrist_geometry.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("wrist_verdict") != "MIXED_OR_UNKNOWN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
