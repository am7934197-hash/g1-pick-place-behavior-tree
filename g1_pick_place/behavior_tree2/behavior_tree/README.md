# G1 pick → place VLA 行为树

执行顺序：`infer_server 保持工作模式头相机 → pick 点模型 → 地图导航到 place → place 点模型 → 导航回 pick → 恢复初始姿态`。

头/腕都保持工作模式，**不要**重启 capture 切数采 para_dir（会把 XCU WBC 的 `target_server` 打掉）。新 pick checkpoint 使用三路相机（`head_right`、`left_arm`、`right_arm`）、23 维状态和 16 维动作；旧 place checkpoint 使用四路相机、23 维状态和 23 维动作。两个 profile 的网络输入都是 224×224。

`infer_server.py` 复用 `../infer_policy` 的 VLA 推理循环、`RobotController`
和 `navigate_to_goal`（地图坐标系），但只初始化一次 G1。当前默认为
`single_model` 模式：A/B 两个逻辑 profile 共用 `shared_profile` 指定的一个模型和
一个 PolicyServer 端口，到 pick/place 时分别把对应 `instruction` 作为 task 发给同一模型。
也保留 `multi_model` 双模型双端口模式。

机器人端直接复用项目内固定版本的 LeRobot async-inference 协议模块：
`lerobot.transport.services_pb2`、`services_pb2_grpc`、`transport.utils`，以及
`lerobot.async_inference.helpers` 中的 wire dataclass。不要在机器人端直接升级为
PyPI 最新 LeRobot；当前控制机是 Python 3.8，而最新版 LeRobot 要求更高版本，且
部署中的 PolicyServer 还包含 `UpdateInstruction`、`ResetPolicy` 和 PNG transport
扩展。若升级，机器人端与推理端必须同时更新并重新跑协议测试。

当前 `rtc_controller.py` 是远端 PolicyServer 的执行适配层，不等同于 LeRobot
本地 `RTCInferenceEngine`。官方 engine 需要直接持有 policy、processor 和 robot，
不能跨现有 gRPC 接口运行；因此这里只复用官方传输模块，保留 G1 的异步 buffer、
欠流保护、停止会话和动作融合逻辑。

底盘移动依赖 SLAM：地图必须在 `/var/maps/cur/`，`localization_server` 已启动，定位 score ≥ 0.8。

## 运行前配置

1. `config.yaml` 中 `vla_runtime.mode: single_model` 会只加载
   `vla_runtime.shared_profile`（当前为 `model_at_a`）及其端口。若改为
   `multi_model`，则会加载 `vla_profiles` 中的全部模型，此时每个模型必须使用不同端口。
   `checkpoint_source: remote` 表示 checkpoint 只存在于推理机，本机不会扫描或逐项打印
   checkpoint 文件缺失信息。
2. 在同一份 `config.yaml` 的 `navigation` 填写示教点文件和点名：
   - `points_file`：`nav_points.json` 路径
   - `pick`：取料点名（地图原点 `[0,0,0,0,0,0,1]`）
   - `place`：放置点名（示教记下的地图位姿）
   并将两个 instruction 改为各自训练时的原始文本。
3. 服务启动前会读取实际要加载的 checkpoint 的 `config.json`，按 profile 校验相机、23 维状态、16/23 维动作、关节顺序和图像尺寸。新 pick 模型缺少的腿/头维度会用实时关节状态补齐；所有 profile 的 `leg`、`head` joint groups 都保持 disabled，当前只下发双臂和双夹爪。

示教新点可用 `project_yuqiz/g1_move/slam/record_nav_points.py`，再把名字写进 `config.yaml`。

## 启动

默认单模型模式只需在推理机启动一个 PolicyServer，端口要与
`vla_profiles.model_at_a.policy_server_address` 一致（下面的模块路径以推理机环境为准）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=30018
```

`0.0.0.0` 是 PolicyServer 的监听地址，不能填为客户端目标地址。真机客户端使用
`config.yaml` 中共享 profile 的 `policy_server_address`；推理机的系统防火墙和路由器/NAT 必须允许该 TCP 端口。
然后在真机的两个终端执行：

```bash
cd /home/zhangyuqi/zhangyuqi/G1-20260911/sorting_task/behavior_tree2.0/behavior_tree
python3 infer_server.py --config config.yaml
python3 behavior_tree.py --config config.yaml
```

启动 `infer_server` 时默认不切头相机（`head_camera.startup_mode` 缺省为 skip）。启动日志应出现 `head camera startup switch disabled`，随后 `Motion init: True`。默认单模型配置只会预加载 `model_at_a`；切回 `multi_model` 后才会同时预加载 `model_at_b`。

`infer_server.py` 在 `single_model` 模式只连接并加载一个模型。可请求
`GET /health` 查看 `vla_runtime_mode`、`shared_profile`、`profile_status` 和
`all_profiles_ready`；共享模型 ready 后再启动行为树。

耗时报告中的“模型A/B任务准备阶段”包含 warmup 推理和夹爪起始状态准备，不是重新加载权重。
默认 `rtc.warmup_once_per_model: true`，因此共享模型只在第一次 pick task 前预热一次；
切换到 place task 不再重复预热。Cold Start 也只按物理 `shared_profile` 统计一次。

行为树只在 pick 点调用模型，不会先导航到 pick；启动前请将机器人放在建图原点。
`/togoal` 使用 `RobotController.navigate_to_goal()`：切到底盘位姿控制器，按
`nav_points.json` 里的地图位姿发目标，轮询到达后再返回。碰撞检测关闭；急停必须旋开。

Place 成功且回到 pick 后，行为树调用 `POST /restore_initial_pose`。该接口复用唯一
`RobotController` 的 `set_init_pose()`，目标值来自 `config.yaml`
中的 `robot.init_joint_positions`，与 `../pick/init_position` 中保存的腿、头、双臂、
双夹爪初始值一致。不会另外运行会重复占用 G1 SDK 的独立姿态脚本。
