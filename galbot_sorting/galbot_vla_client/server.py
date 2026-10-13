"""
FastAPI task server for VLA robot control (Python 3.8).
Direct gRPC connection to remote PolicyServer.
"""

import os
import time
import json
import yaml
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from robot_controller import RobotController
from vla_client import VLAClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    import cv2
except ImportError:
    cv2 = None

# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_info(path: str = "info.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_config(os.environ.get("CONFIG_PATH", os.path.join(BASE_DIR, "config.yaml")))
INFO = load_info(os.environ.get("INFO_PATH", os.path.join(BASE_DIR, "info.json")))

# Merge static robot info into runtime config for convenience. Runtime config
# wins on overlapping fields such as sensor_set.
ROBOT_CFG = {**INFO.get("robot", {}), **CONFIG.get("robot", {})}
VLA_CFG = CONFIG.get("vla", {})
SERVER_CFG = CONFIG.get("server", {})

# ------------------------------------------------------------------
# Global state
# ------------------------------------------------------------------

robot: Optional[RobotController] = None
vla: Optional[VLAClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global robot, vla
    robot = RobotController(ROBOT_CFG)
    vla = VLAClient(VLA_CFG)

    if robot.is_available():
        robot.init()
        time.sleep(2.0)  # Wait for robot/motion controllers to stabilize

        init_pos = ROBOT_CFG.get("init_arm_position")
        if init_pos is not None:
            print("[Server] Setting initial arm position...")
            ok = robot.set_arm_position(init_pos, timeout=8.0)
            if not ok:
                print("[Server] Warning: set_arm_position failed, retrying once...")
                time.sleep(1.0)
                robot.set_arm_position(init_pos, timeout=8.0)

        init_gripper = ROBOT_CFG.get("init_gripper_position")
        if init_gripper is not None and len(init_gripper) >= 2:
            print("[Server] Setting initial gripper positions...")
            robot.set_gripper_positions(init_gripper[0], init_gripper[1])
    else:
        print("[Server] Robot SDK not available, running in simulation mode")

    # Connect to PolicyServer (always connect; setup policy only if configured)
    if vla.connect():
        if VLA_CFG.get("auto_setup_policy", True):
            vla.setup_policy()
        print("[Server] success to connect to PolicyServer")
    else:
        print("[Server] Warning: failed to connect to PolicyServer")

    yield

    if robot is not None:
        robot.shutdown()
    if vla is not None:
        vla.disconnect()


app = FastAPI(title="Galbot VLA Task Server", lifespan=lifespan)

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------

class ExecuteTaskRequest(BaseModel):
    instruction: str
    enable_navigation: bool = True
    enable_vla: bool = True
    enable_placement: bool = True


class TaskResponse(BaseModel):
    success: bool
    message: str
    details: Optional[dict] = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _save_observation_images(obs: dict, step: int, output_dir: str = "vla_debug_images"):
    """Save RGB images from observation dict to disk for debugging."""
    if cv2 is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for key in ("head_right", "left_arm", "right_arm"):
        img = obs.get(key)
        if img is None:
            continue
        if not hasattr(img, "shape"):
            continue
        # obs images are RGB; cv2.imwrite expects BGR
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_to_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_to_save = img
        filename = os.path.join(output_dir, f"step{step:04d}_{key}_{timestamp}.png")
        cv2.imwrite(filename, img_to_save)
        print(f"[Server] Saved debug image: {filename}")


def _resolve_server_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _append_vla_inference_log(
    run_id: str,
    instruction: str,
    step: int,
    actions,
    executed_actions: Optional[int] = None,
    obs: Optional[dict] = None,
) -> Optional[str]:
    output_file = SERVER_CFG.get("vla_inference_log_file", "vla_inference_results.jsonl")
    if not output_file:
        return None

    output_path = _resolve_server_path(output_file)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    record = {
        "event": "vla_inference",
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "run_id": run_id,
        "step": step,
        "instruction": instruction,
        "observation_gripper": {
            "left": obs.get("left_arm_gripper") if obs else None,
            "right": obs.get("right_arm_gripper") if obs else None,
        },
        "num_actions": len(actions) if actions else 0,
        "executed_actions": executed_actions,
        "actions": actions or [],
    }
    with open(output_path, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, default=_json_default)
        f.write("\n")

    return output_path


def _navigate_to(tag: str) -> bool:
    pose = ROBOT_CFG.get("navigation", {}).get(tag)
    if pose is None:
        print(f"[Server] No pose configured for '{tag}'")
        return False
    print(f"[Server] Navigating to {tag}: {pose}")
    return robot.navigate_to_goal(pose, timeout=30.0, retry=3)


def _execute_vla(instruction: str, max_steps: int = 2000) -> dict:
    """Run VLA inference loop and execute actions."""
    dt = ROBOT_CFG.get("control", {}).get("dt", 1.0 / 30.0)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    inference_log_file = None

    # Update instruction on policy server
    vla.update_instruction(instruction)
    time.sleep(0.2)

    step = 0
    total_actions_executed = 0

    while step < max_steps:
        # Get observation (head, left, right cameras + 16-DOF state)
        obs = robot.get_observation()
        if obs is None:
            print(f"[Server] Failed to get observation at step {step}, retrying...")
            time.sleep(dt)
            continue

        # Save images used for VLA inference
        _save_observation_images(obs, step)

        # Infer action chunk via gRPC
        actions = vla.infer(obs, instruction, timestep=step)
        execute_count = 0
        actions_to_execute = []
        if actions:
            execute_count = int(VLA_CFG.get("execute_actions_per_chunk", len(actions)))
            if execute_count <= 0 or execute_count > len(actions):
                execute_count = len(actions)
            actions_to_execute = actions[:execute_count]

        inference_log_file = _append_vla_inference_log(
            run_id,
            instruction,
            step,
            actions,
            executed_actions=len(actions_to_execute),
            obs=obs,
        )
        if inference_log_file:
            print(f"[Server] Saved VLA inference result: {inference_log_file}")
        if not actions:
            print(f"[Server] No actions returned at step {step}, ending VLA loop")
            break

        print(
            f"[Server] Received {len(actions)} actions at step {step}, "
            f"executing first {len(actions_to_execute)}"
        )

        # Extract joint angles from actions
        action_vectors = [a["action"] for a in actions_to_execute if "action" in a]

        # Execute actions
        robot.execute_action_chunk(action_vectors, dt=dt)
        total_actions_executed += len(action_vectors)

        # Advance step by number of actions consumed
        step += len(actions_to_execute)

    return {
        "run_id": run_id,
        "steps": step,
        "total_actions_executed": total_actions_executed,
        "inference_log_file": inference_log_file,
    }


def _place_object() -> bool:
    """Move arms to fixed placement joint angles and open gripper."""
    placement = ROBOT_CFG.get("placement", {})
    target_joints = placement.get("fixed_placement_joints")
    if target_joints is None:
        print("[Server] No fixed_placement_joints configured")
        return False

    print(f"[Server] Moving to placement joint angles: {target_joints}")
    ok = robot.set_arm_position(
        target_joints,
        timeout=placement.get("set_ee_timeout", 5.0)
    )
    if not ok:
        print("[Server] Failed to reach placement joints")
        return False

    time.sleep(0.5)
    robot.open_gripper()
    time.sleep(0.5)
    return True


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "robot_available": robot.is_available() if robot else False,
        "policy_server_connected": vla.connected if vla else False,
    }


@app.post("/execute_task", response_model=TaskResponse)
def execute_task(req: ExecuteTaskRequest):
    """
    Execute the full pick-and-place task:
    1. Navigate to shelf
    2. Run VLA to grasp object
    3. Navigate to table
    4. Place object
    """
    if robot is None or vla is None:
        return TaskResponse(success=False, message="Server not initialized")

    details = {}

    try:
        # Step 1: Navigate to shelf
        if req.enable_navigation:
            ok = _navigate_to("shelf_pose")
            details["nav_to_shelf"] = ok
            if not ok:
                return TaskResponse(
                    success=False,
                    message="Failed to navigate to shelf",
                    details=details,
                )
            time.sleep(1.0)

        # Step 2: VLA grasp
        if req.enable_vla:
            vla_summary = _execute_vla(req.instruction)
            details["vla"] = vla_summary

        # Step 3: Navigate to table
        if req.enable_navigation:
            ok = _navigate_to("table_pose")
            details["nav_to_table"] = ok
            if not ok:
                return TaskResponse(
                    success=False,
                    message="Failed to navigate to table",
                    details=details,
                )
            time.sleep(1.0)

        # Step 4: Place object
        if req.enable_placement:
            ok = _place_object()
            details["placement"] = ok
            if not ok:
                return TaskResponse(
                    success=False,
                    message="Failed to place object",
                    details=details,
                )

        return TaskResponse(
            success=True,
            message="Task completed successfully",
            details=details,
        )

    except Exception as e:
        return TaskResponse(
            success=False,
            message=f"Task execution error: {e}",
            details=details,
        )


@app.post("/vla_only", response_model=TaskResponse)
def vla_only(req: ExecuteTaskRequest):
    """
    Execute VLA inference only (no navigation, no placement).
    Useful for debugging or pure manipulation tasks.
    """
    if robot is None or vla is None:
        return TaskResponse(success=False, message="Server not initialized")

    try:
        vla_summary = _execute_vla(req.instruction)
        return TaskResponse(
            success=True,
            message="VLA execution completed",
            details={"vla": vla_summary},
        )
    except Exception as e:
        return TaskResponse(success=False, message=f"VLA error: {e}")


@app.post("/navigate")
def navigate_to(tag: str):
    """Direct navigation endpoint for testing."""
    if robot is None:
        return {"success": False, "message": "Robot not initialized"}
    ok = _navigate_to(tag)
    return {"success": ok, "tag": tag}


@app.post("/set_gripper")
def set_gripper(width: float, velocity: Optional[float] = None):
    """Direct gripper control for testing."""
    if robot is None:
        return {"success": False, "message": "Robot not initialized"}
    ok = robot.set_gripper(width, velocity=velocity)
    reported_velocity = velocity if velocity is not None else ROBOT_CFG.get("gripper_velocity")
    return {"success": ok, "width": width, "velocity": reported_velocity}


@app.post("/reset_vla")
def reset_vla():
    """Reset VLA policy state."""
    if vla is None:
        return {"success": False, "message": "VLA not initialized"}
    ok = vla.reset()
    return {"success": ok}


@app.post("/reconnect_vla")
def reconnect_vla():
    """Reconnect to PolicyServer (useful after server restart)."""
    if vla is None:
        return {"success": False, "message": "VLA client not initialized"}
    if vla.connected:
        vla.disconnect()
    ok = vla.connect()
    if ok:
        vla.setup_policy()
    return {"success": ok, "connected": vla.connected}


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    host = SERVER_CFG.get("host", "0.0.0.0")
    port = SERVER_CFG.get("port", 8005)
    uvicorn.run(app, host=host, port=port)
