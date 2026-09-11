#!/usr/bin/env python3
"""A -> B G1 task behaviour tree.

The tree deliberately has only one robot owner: ``infer_server.py``.  This
process sends commands to that server over HTTP, so two VLA stages cannot
compete for the G1 SDK connection.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


LOG = logging.getLogger("g1_ab_tree")


class Status(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


@dataclass
class Context:
    config: dict[str, Any]
    client: "InferServerClient"
    error: str = ""
    completed_goals: list[str] = field(default_factory=list)


class Action:
    def __init__(self, name: str, fn: Callable[[Context], bool]) -> None:
        self.name = name
        self.fn = fn

    def tick(self, context: Context) -> Status:
        LOG.info("[BT] %s", self.name)
        try:
            if self.fn(context):
                return Status.SUCCESS
        except Exception as exc:  # preserve the actual remote error in the log
            context.error = f"{self.name}: {exc}"
            LOG.exception("[BT] %s failed", self.name)
            return Status.FAILURE
        if not context.error:
            context.error = f"{self.name} failed"
        LOG.error("[BT] %s", context.error)
        return Status.FAILURE


class Sequence:
    def __init__(self, name: str, *children: Action) -> None:
        self.name = name
        self.children = children

    def run(self, context: Context) -> Status:
        LOG.info("[BT] start sequence: %s", self.name)
        for child in self.children:
            status = child.tick(context)
            if status is Status.FAILURE:
                LOG.error("[BT] sequence stopped at %s", child.name)
                return status
        LOG.info("[BT] sequence completed: %s", self.name)
        return Status.SUCCESS


class InferServerClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}", body, method="POST" if body is not None else "GET",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} {path}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Cannot reach infer_server at {self.base_url}: {exc.reason}") from exc
        if not result.get("success", False):
            raise RuntimeError(result.get("message", f"{path} returned success=false"))
        return result

    def health(self) -> dict[str, Any]:
        return self.request("/health")

    def execute(self, profile: str, instruction: str, max_steps: int) -> dict[str, Any]:
        return self.request("/execute", {
            "profile": profile, "instruction": instruction, "max_steps": max_steps,
        })

    def set_head_camera_mode(self, mode: str) -> dict[str, Any]:
        return self.request("/head_camera_mode", {"mode": mode})

    def togoal(
        self, goal_name: str, points_file: str, timeout_seconds: float, retries: int,
    ) -> dict[str, Any]:
        return self.request("/togoal", {
            "goal_name": goal_name,
            "points_file": points_file,
            "timeout_seconds": timeout_seconds,
            "retries": retries,
        })

    def restore_initial_pose(self) -> dict[str, Any]:
        return self.request("/restore_initial_pose", {})


def _load_nav_points(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    points = data.get("points")
    if not isinstance(points, dict) or not points:
        raise ValueError(f"no points in {path}")
    return points


def _navigation_config(config: dict[str, Any], config_path: Path | None = None) -> dict[str, Any]:
    nav = config.get("navigation")
    if not isinstance(nav, dict):
        raise ValueError("Missing mapping 'navigation'")
    points_file = nav.get("points_file")
    pick = nav.get("pick")
    place = nav.get("place")
    if not points_file or not pick or not place:
        raise ValueError("navigation.points_file, pick, and place are required")
    path = Path(str(points_file))
    if not path.is_absolute() and config_path is not None:
        path = (config_path.parent / path).resolve()
    if not path.is_file():
        raise ValueError(f"navigation.points_file does not exist: {path}")
    points = _load_nav_points(path)
    for name in (pick, place):
        rec = points.get(name)
        if rec is None:
            raise ValueError(f"navigation point '{name}' not in {path}")
        pose = rec.get("pose") if isinstance(rec, dict) else None
        if not isinstance(pose, list) or len(pose) != 7:
            raise ValueError(f"point '{name}' pose must be 7 numbers")
    nav["points_file"] = str(path)
    nav["pick"] = str(pick)
    nav["place"] = str(place)
    nav["timeout_seconds"] = float(nav.get("timeout_seconds", 60.0))
    nav["retries"] = int(nav.get("retries", 3))
    if nav["timeout_seconds"] <= 0:
        raise ValueError("navigation.timeout_seconds must be > 0")
    if nav["retries"] < 1:
        raise ValueError("navigation.retries must be >= 1")
    return nav


def _run_vla(
    ctx: Context, profile: str, instruction: str, max_steps: int,
) -> bool:
    """Run one VLA stage. Head capture stays on working; FOV is emulated in preprocess."""
    ctx.client.execute(profile, instruction, max_steps)
    return True


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for key in ("infer_server", "point_a", "point_b"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Missing mapping '{key}' in {path}")
    _navigation_config(config, path)
    for point in ("point_a", "point_b"):
        stage = config[point]
        if not stage.get("profile") or not stage.get("instruction"):
            raise ValueError(f"{point}.profile and {point}.instruction are required")
    return config


def build_tree(context: Context) -> Sequence:
    config = context.config
    point_a = config["point_a"]
    point_b = config["point_b"]
    nav = _navigation_config(config)
    points_file = nav["points_file"]
    pick_name = nav["pick"]
    place_name = nav["place"]
    timeout_seconds = float(nav["timeout_seconds"])
    retries = int(nav["retries"])

    def healthcheck(ctx: Context) -> bool:
        ctx.client.health()
        return True

    def run_at_a(ctx: Context) -> bool:
        return _run_vla(
            ctx, point_a["profile"], point_a["instruction"], int(point_a.get("max_steps", 2000)),
        )

    def move_to_place(ctx: Context) -> bool:
        ctx.client.togoal(place_name, points_file, timeout_seconds, retries)
        ctx.completed_goals.append(place_name)
        return True

    def run_at_b(ctx: Context) -> bool:
        return _run_vla(
            ctx, point_b["profile"], point_b["instruction"], int(point_b.get("max_steps", 2000)),
        )

    def return_to_pick(ctx: Context) -> bool:
        if not ctx.completed_goals:
            return True
        ctx.client.togoal(pick_name, points_file, timeout_seconds, retries)
        ctx.completed_goals.clear()
        return True

    def restore_initial_pose(ctx: Context) -> bool:
        ctx.client.restore_initial_pose()
        return True

    return Sequence(
        "G1_A_to_B",
        Action("Check infer_server", healthcheck),
        # The policy was trained from a tightly controlled ready_224 start pose.
        # Do not rely on a prior manual positioning script or on server startup:
        # this tree may be started while the robot is in any safe pose at pick.
        Action("Restore the training pose before point A", restore_initial_pose),
        Action("Run VLA model at pick", run_at_a),
        Action("Navigate from pick to place", move_to_place),
        Action("Run VLA model at place", run_at_b),
        Action("Navigate back to pick", return_to_pick),
        Action("Restore the configured initial joint pose", restore_initial_pose),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the G1 A -> B VLA behaviour tree")
    parser.add_argument("--config", default=Path(__file__).with_name("task.yaml"), type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_config(args.config)
        server = config["infer_server"]
        client = InferServerClient(
            str(server.get("url", "http://127.0.0.1:8006")), float(server.get("timeout_seconds", 3600)),
        )
        context = Context(config=config, client=client)
        status = build_tree(context).run(context)
    except Exception as exc:
        LOG.error("Configuration or startup failure: %s", exc)
        return 2
    if status is Status.SUCCESS:
        return 0
    LOG.error("Task failed: %s", context.error)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
