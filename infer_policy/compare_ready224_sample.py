"""Compare one ready_224 sample against the pinned inference schema."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from alignment_schema import (
    ACTION_NAMES,
    CAMERA_KEYS,
    READY224_DIR,
    STATE_NAMES,
    TRAINING_FPS,
)


def load_info(dataset_dir: str) -> dict:
    path = os.path.join(dataset_dir, "meta", "info.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _feature_names(info: dict, key: str) -> List[str]:
    feat = (info.get("features") or {}).get(key) or {}
    names = feat.get("names") or []
    if names and isinstance(names[0], list):
        names = names[0]
    return list(names)


def compare(dataset_dir: str) -> Dict[str, Any]:
    info = load_info(dataset_dir)
    features = info.get("features") or {}
    state_names = _feature_names(info, "observation.state")
    action_names = _feature_names(info, "action")
    image_keys = [k for k, v in features.items() if (v or {}).get("dtype") in ("image", "video")]
    fps = float(info.get("fps") or 0)
    report = {
        "dataset_dir": os.path.abspath(dataset_dir),
        "fps": {"dataset": fps, "expected": TRAINING_FPS, "pass": abs(fps - TRAINING_FPS) < 0.01},
        "state_names": {
            "dataset": state_names,
            "expected": STATE_NAMES,
            "pass": state_names == STATE_NAMES,
        },
        "action_names": {
            "dataset": action_names[:16] if action_names else action_names,
            "expected": ACTION_NAMES,
            "pass": (action_names[:16] == ACTION_NAMES) if action_names else False,
        },
        "images": {
            "dataset": sorted(image_keys),
            "expected": [f"observation.images.{k}" for k in CAMERA_KEYS],
            "pass": all(f"observation.images.{k}" in image_keys for k in CAMERA_KEYS),
        },
    }
    report["pass"] = all(
        report[k]["pass"] for k in ("fps", "state_names", "action_names", "images")
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=os.environ.get("TRAIN_DATASET_DIR", READY224_DIR))
    args = parser.parse_args()
    if not os.path.isdir(args.dataset):
        print(json.dumps({
            "pass": False,
            "error": f"dataset not readable on this machine: {args.dataset}",
        }, indent=2))
        return 2
    report = compare(args.dataset)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
