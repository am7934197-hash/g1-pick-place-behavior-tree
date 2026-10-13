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
cd /home/galbot/project_yuqiz/reply/init_position

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

边看头相机边调头（不要和 replay 同时跑）：

```bash
python3 tune_head_live.py
```

浏览器打开 `http://127.0.0.1:8765/` 看左右头相机实时画面。键盘：`w/s` 低头/抬头，`a/d` 左右转，回车把当前角度写进脚本。

边看头相机边调腿高（不要和 replay / `tune_head_live.py` 同时跑）：

```bash
python3 tune_leg_live.py
```

浏览器打开 `http://127.0.0.1:8766/`。键盘：`w/s` 降低/升高（j1/j2/j3 按 1:2.3:1.2 一起动），`a/d` 腰转，回车保存。

只把身体转正（腰 `leg_joint4`/`leg_joint5` 回 0，身高不动）：

```bash
python3 set_body_straight.py
```

边看头相机 + 左右腕部相机调双臂（不要和 replay / 其它 `tune_*_live.py` 同时跑）：

```bash
python3 tune_arm_live.py
```

浏览器打开 `http://127.0.0.1:8767/`。键盘：`l/r` 选左/右臂，`1-7` 选关节，`w/s` 减小/增大，回车保存。

每个脚本会：初始化 SDK → 阻塞等待该组关节到位 → 释放资源。  
限速 `0.4 rad/s`（夹爪 `0.1`），超时 `45s`。成功会打印 `Joint angle setting succeeded`。

## 目标关节角


| 脚本                                | 关节组             | 目标位置（rad）                                                      |
| --------------------------------- | --------------- | -------------------------------------------------------------- |
| `set_init_leg_positions.py`       | `leg` (5)       | `[0.5369, 1.3694, 0.8780, 0.0001, 0.0009]`                     |
| `set_init_head_positions.py`      | `head` (2)      | `[-0.1350, 0.2000]`                                            |
| `set_init_left_arm_positions.py`  | `left_arm` (7)  | `[0.8356, -0.8958, -0.3661, -1.9185, 0.4191, 0.4797, 1.5382]` |
| `set_init_right_arm_positions.py` | `right_arm` (7) | `[-0.5729, 0.7911, 0.1447, 2.0235, -0.4703, -0.7324, -1.5381]`  |
| `set_init_gripper_positions.py`   | 双夹爪 (2)         | `[0.0, 0.0]`（物理宽度，和训练起始值一致）                                    |




## 注意

- 双夹爪由 `set_init_gripper_positions.py` 单独控制，训练起始值均为 `0.0`。
- 四个脚本各自 `init` / `destroy`，不能并行执行。
- 失败时打印 `Joint angle setting failed`，常见原因：急停、SDK 被占用、超时未到位。
- 改目标角度：编辑对应脚本里的 `joint_pos`。

