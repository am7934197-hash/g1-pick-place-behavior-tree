#!/usr/bin/env python3
"""In-place pick-model test: infer and execute without a behavior tree."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from image_preprocessor import ImagePreprocessor, validate_image_shape
from robot_io import StationaryRobot
from vla_client import VLAClient


def parse_args():
    parser = argparse.ArgumentParser(
        description="Restore the pick pose, then infer and execute pick actions in place."
    )
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--instruction", help="override the instruction in config.yaml")
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="infer cycles; 0 means until Ctrl+C; default is config max_steps",
    )
    parser.add_argument("--interval", type=float, default=1.0,
                        help="seconds between dry-run requests")
    parser.add_argument("--skip-policy-setup", action="store_true",
                        help="use a policy already loaded on PolicyServer")
    parser.add_argument("--skip-initial-pose", action="store_true",
                        help="do not move to the configured training-start pose")
    parser.add_argument("--dry-run", action="store_true",
                        help="predict only; do not execute returned actions")
    parser.add_argument("--no-save-images", action="store_true")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    for section in ("robot", "vla", "image_preprocess"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing config section: {section}")
    return config


def validate_observation(observation: dict, features: dict) -> None:
    for feature_key, feature in features.items():
        if feature.get("dtype") == "image":
            camera = feature_key.rsplit(".", 1)[-1]
            if camera not in observation:
                raise ValueError(f"missing camera: {camera}")
            validate_image_shape(observation[camera], feature["shape"])
            continue
        if feature_key == "observation.state":
            names = list(feature.get("names", []))
            missing = [name for name in names if name not in observation]
            if missing:
                raise ValueError(f"missing state values: {missing}")
            values = np.asarray([observation[name] for name in names], dtype=np.float32)
            expected = int(feature["shape"][0])
            if values.size != expected or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid state vector: shape={values.shape}, expected ({expected},)")


def save_images(observation: dict, directory: Path, step: int) -> list:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for camera in ("head_right", "left_arm", "right_arm"):
        image = observation[camera]
        path = directory / f"{stamp}_step{step}_{camera}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"failed to save {path}")
        paths.append(str(path))
    return paths


def append_result(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def action_vectors(actions: list, count: int) -> list:
    vectors = []
    for item in actions:
        value = item["action"] if isinstance(item, dict) else item
        vectors.append(value)
        if len(vectors) >= count:
            break
    return vectors


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    robot_cfg, vla_cfg = config["robot"], config["vla"]
    instruction = args.instruction or vla_cfg.get("instruction", "")
    if not instruction:
        raise ValueError("instruction cannot be empty")

    execute = bool(config.get("execute", True)) and not args.dry_run
    if args.count is not None:
        max_steps = args.count
    elif execute:
        max_steps = int(config.get("max_steps", 400))
    else:
        max_steps = 1
    if max_steps < 0:
        raise ValueError("count must be >= 0")
    execute_count = max(1, int(config.get("execute_actions_per_chunk", 20)))

    processor = ImagePreprocessor(config["image_preprocess"])
    features = vla_cfg["observation_features"]
    output_dir = Path(config.get("output", {}).get("directory", "outputs"))
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    result_file = output_dir / config.get("output", {}).get("result_file", "predictions.jsonl")

    robot = StationaryRobot(robot_cfg)
    client = VLAClient(vla_cfg)
    try:
        robot.initialize()
        if not args.skip_initial_pose:
            robot.set_initial_pose()
        client.connect()
        if not args.skip_policy_setup and vla_cfg.get("setup_policy", True):
            client.setup_policy()

        print(
            f"[infer] execute={execute} max_steps={max_steps or 'until Ctrl+C'} "
            f"chunk={execute_count} instruction={instruction!r}"
        )
        step = 0
        while max_steps == 0 or step < max_steps:
            started = time.perf_counter()
            raw = robot.capture_observation()
            raw_shapes = {
                key: list(value.shape) for key, value in raw.items()
                if isinstance(value, np.ndarray) and value.ndim == 3
            }
            observation = processor.process(raw)
            validate_observation(observation, features)
            sent_shapes = {key: list(observation[key].shape) for key in raw_shapes}
            print(f"[infer] step={step} raw_shapes={raw_shapes} sent_shapes={sent_shapes}")

            image_paths = [] if args.no_save_images else save_images(observation, output_dir / "images", step)
            actions, timing = client.infer(observation, instruction, timestep=step)
            chunk = action_vectors(actions, execute_count)
            executed = False
            if execute:
                if not chunk:
                    raise RuntimeError("policy returned no actions")
                executed = robot.execute_action_chunk(chunk)
                if not executed:
                    raise RuntimeError("failed to execute action chunk")
            record = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "step": step,
                "instruction": instruction,
                "raw_image_shapes": raw_shapes,
                "sent_image_shapes": sent_shapes,
                "state": {
                    name: observation[name]
                    for name in (features.get("observation.state") or {}).get("names", [])
                },
                "image_files": image_paths,
                "timing": timing,
                "num_actions": len(actions),
                "executed_actions": len(chunk) if execute else 0,
                "executed": executed,
                "actions": actions,
            }
            append_result(result_file, record)
            print(
                f"[infer] predicted {len(actions)} actions; "
                f"{'executed ' + str(len(chunk)) if execute else 'NOT EXECUTED'}; "
                f"result={result_file} timing={timing}"
            )
            step += 1
            if not execute:
                remaining = args.interval - (time.perf_counter() - started)
                if remaining > 0 and (max_steps == 0 or step < max_steps):
                    time.sleep(remaining)
    except KeyboardInterrupt:
        print("\n[infer] stopped by user")
    finally:
        client.close()
        robot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
