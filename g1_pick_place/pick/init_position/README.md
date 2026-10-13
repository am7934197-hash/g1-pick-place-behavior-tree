# 抓取任务初始位姿

把 G1 的腿、头、左臂、右臂调到抓取/分拣任务的初始关节角。  
每个脚本只控制一组关节，互不影响。

## 前置条件

- 本机已安装 Galbot SDK（`python3` 能 `import galbot_sdk.g1`）
- 机器人已上电、无急停
- **不要**与正在占用 SDK 的进程同时跑（例如 `infer_policy/server.py`）
- 周围无碰撞物，急停随时可按

## 推荐执行顺序

先抬腿（躯干高度），再转头，最后动双臂：

```bash
cd /home/galbot/project_yuqiz/pick/init_position

python3 set_init_leg_positions.py
python3 set_init_head_positions.py
python3 set_init_left_arm_positions.py
python3 set_init_right_arm_positions.py
python3 set_init_gripper_positions.py
```

也可以只跑其中某一个，例如只调右臂：

```bash
python3 set_init_right_arm_positions.py
```

每个脚本会：初始化 SDK → 阻塞等待该组关节到位 → 释放资源。  
限速 `0.4 rad/s`（夹爪 `0.1`），超时 `45s`。成功会打印 `Joint angle setting succeeded`。

## 目标关节角

| 脚本 | 关节组 | 目标位置（rad） |
|------|--------|-----------------|
| `set_init_leg_positions.py` | `leg` (5) | `[0.5770, 1.4614, 0.9260, 0.0401, 0.0010]` |
| `set_init_head_positions.py` | `head` (2) | `[-0.0750, 0.4300]` |
| `set_init_left_arm_positions.py` | `left_arm` (7) | `[0.7585, -0.6251, -0.1051, -1.6019, -0.2661, 0.0511, 1.1842]` |
| `set_init_right_arm_positions.py` | `right_arm` (7) | `[-1.1494, 0.6022, 0.7486, 1.4833, 0.3566, -0.2180, -1.4230]` |
| `set_init_gripper_positions.py` | 双夹爪 (2) | `[0.0, 0.0]`（物理宽度，和训练起始值一致） |

## 注意

- 双夹爪由 `set_init_gripper_positions.py` 单独控制，训练起始值均为 `0.0`。
- 四个脚本各自 `init` / `destroy`，不能并行执行。
- 失败时打印 `Joint angle setting failed`，常见原因：急停、SDK 被占用、超时未到位。
- 改目标角度：编辑对应脚本里的 `joint_pos`。
