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
限速 `0.4 rad/s`。夹爪走 `set_gripper_command`（速度 `0.1 m/s`，力矩 `10 N`），目标 `0.0 m` 完全闭合时 SDK 经常报不到位，所以不阻塞等编码器，下发后等待 2s。成功会打印 `Joint angle setting succeeded` 或 `Gripper position setting succeeded`。

## 目标关节角

下列数值是当前 `020000` checkpoint 所用 `ready_224` 数据集中，
52 条抓取 episode 第一帧的逐关节中位数。这些值与
`infer_policy/vla_config.yaml` 的 `robot.init_joint_positions` 保持一致。

| 脚本 | 关节组 | 目标位置（rad） |
|------|--------|-----------------|
| `set_init_leg_positions.py` | `leg` (5) | `[0.576609, 1.462627, 0.925278, 0.040399, 0.000090]` |
| `set_init_head_positions.py` | `head` (2) | `[-0.074925, 0.430006]` |
| `set_init_left_arm_positions.py` | `left_arm` (7) | `[0.767878, -0.583967, -0.088312, -1.620591, -0.284889, 0.026508, 1.193521]` |
| `set_init_right_arm_positions.py` | `right_arm` (7) | `[-1.074697, 0.587419, 0.683460, 1.414618, 0.352612, -0.195067, -1.538162]` |
| `set_init_gripper_positions.py` | 双夹爪 (2) | `[0.0, 0.0]`（物理宽度，和训练起始值一致） |

## 注意

- 双夹爪由 `set_init_gripper_positions.py` 单独控制，训练起始值均为 `0.0`。
- 四个脚本各自 `init` / `destroy`，不能并行执行。
- 失败时打印 `Joint angle setting failed`，常见原因：急停、SDK 被占用、超时未到位。
- 改目标角度：编辑对应脚本里的 `joint_pos`。
