# G1 pick → place 双 VLA 行为树

执行顺序：`infer_server 保持工作模式头相机 → pick 点模型 → 地图导航到 place → place 点模型 → 导航回 pick → 恢复初始姿态`。

头/腕都保持工作模式，**不要**重启 capture 切数采 para_dir（会把 XCU WBC 的 `target_server` 打掉）。新 pick checkpoint 使用三路相机（`head_right`、`left_arm`、`right_arm`）、23 维状态和 16 维动作；旧 place checkpoint 使用四路相机、23 维状态和 23 维动作。两个 profile 的网络输入都是 224×224。

`infer_server.py` 复用 `../infer_policy` 的 VLA 推理循环、`RobotController`
和 `navigate_to_goal`（地图坐标系），但只初始化一次 G1。服务启动时
并行连接两个 PolicyServer 并预加载 A/B 模型；每次 `/execute` 只切换已就绪的
客户端，不再重新加载模型。两个 PolicyServer 只负责 GPU 推理，仍由唯一的
`infer_server.py` 控制机器人。

底盘移动依赖 SLAM：地图必须在 `/var/maps/cur/`，`localization_server` 已启动，定位 score ≥ 0.8。

## 运行前配置

1. `infer_server.yaml` 已填写 pick/place 两个 `040000/pretrained_model` 路径。每个 PolicyServer 只能持有一个模型，
   因此 A/B 必须使用不同端口（默认本地 SSH 隧道端口为 `30017`、`30018`）。两个
   PolicyServer 可以在同一张 GPU 7 上运行。
2. 在 `task.yaml` 的 `navigation` 填写示教点文件和点名：
   - `points_file`：`nav_points.json` 路径
   - `pick`：取料点名（地图原点 `[0,0,0,0,0,0,1]`）
   - `place`：放置点名（示教记下的地图位姿）
   并将两个 instruction 改为各自训练时的原始文本。
3. 服务启动前会分别读取两个 checkpoint 的 `config.json`，按 profile 校验相机、23 维状态、16/23 维动作、关节顺序和图像尺寸。新 pick 模型缺少的腿/头维度会用实时关节状态补齐；所有 profile 的 `leg`、`head` joint groups 都保持 disabled，当前只下发双臂和双夹爪。

示教新点可用 `project_yuqiz/g1_move/slam/record_nav_points.py`，再把名字写进 `task.yaml`。

## 启动

先在推理机的 GPU 7 上启动两个 PolicyServer（下面的模块路径请以推理机环境为准）：

```bash
CUDA_VISIBLE_DEVICES=7 python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=30017
CUDA_VISIBLE_DEVICES=7 python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=30018
```

再分别建立 SSH 隧道，将远端 `30017/30018` 映射为真机侧的 `30017/30018`。然后在
真机的两个终端执行：

```bash
cd /home/zhangyuqi/zhangyuqi/G1-20260911/sorting_task/behavior_tree2.0/behavior_tree
python3 infer_server.py --config infer_server.yaml
python3 behavior_tree.py --config task.yaml
```

启动 `infer_server` 时**不切**头相机模式。启动日志应出现 `head camera startup switch disabled`，随后 `Motion init: True`。日志还应显示 `model_at_a` 为三路 224×224、16 维动作，`model_at_b` 为四路 224×224、23 维动作。

`infer_server.py` 会并行连接两个模型。可请求 `GET /health` 查看
`profile_status` 和 `all_profiles_ready`；两个模型都 ready 后再启动行为树。

行为树只在 pick 点调用模型，不会先导航到 pick；启动前请将机器人放在建图原点。
`/togoal` 使用 `RobotController.navigate_to_goal()`：切到底盘位姿控制器，按
`nav_points.json` 里的地图位姿发目标，轮询到达后再返回。碰撞检测关闭；急停必须旋开。

Place 成功且回到 pick 后，行为树调用 `POST /restore_initial_pose`。该接口复用唯一
`RobotController` 的 `set_init_pose()`，目标值来自 `../infer_policy/vla_config.yaml`
中的 `robot.init_joint_positions`，与 `../pick/init_position` 中保存的腿、头、双臂、
双夹爪初始值一致。不会另外运行会重复占用 G1 SDK 的独立姿态脚本。
