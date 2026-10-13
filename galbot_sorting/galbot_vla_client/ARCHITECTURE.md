# VLA 机器人任务服务 — 架构结论

## 问题背景

- **galbot SDK** 绑定 Python 3.8，运行在机器人本体上
- **lerobot** 要求 Python >= 3.12，PolicyServer 部署在远端 GPU 服务器上
- 两个库无法在同一 Python 进程中运行，必须跨进程/跨机器通信

## 最终架构（gRPC 直连）

```
机器人端 (Python 3.8)              远端服务器 (Python 3.12)
+------------------+              +---------------------------+
|  galbot_vla_client|  gRPC       |  PolicyServer (gRPC:8080) |
|  (FastAPI:8000)  |  protobuf   |     protobuf 6.31         |
|                  | <----------> |     lerobot, VLA model    |
|  robot_controller|              +---------------------------+
|  galbot SDK      |
|  set_joint_cmds  |
+------------------+
```

### 各组件职责

| 组件 | 运行环境 | 核心职责 |
|------|---------|---------|
| **PolicyServer** | 远端 Python 3.12 | 已有服务，加载 VLA 策略模型，gRPC 接收 observation，返回 action chunk |
| **galbot_vla_client** | 机器人端 Python 3.8 | FastAPI 服务，编排任务流程，调用 galbot SDK 控制机器人，gRPC 直连远端 PolicyServer |

## 为什么 gRPC 直连可行（之前认为不可行）

### 关键发现

lerobot 仓库里预生成的 `services_pb2.py` 是用 **protobuf 6.31** 的 `protoc` 生成的，包含 `runtime_version` 版本检查，导致 **protobuf 4.x runtime 无法 import**。

但用 **grpcio-tools 自带的 protoc（5.26.1）重新生成** `services_pb2.py` 后：
- **不包含 `runtime_version` 检查**
- **protobuf 4.25.4 runtime 可以正常 import 和使用**
- **wire format 与 protobuf 6.31 完全兼容**，两端可以正常通信

### protobuf 版本兼容性原理

| 层面 | 是否需要对齐 | 原因 |
|------|-------------|------|
| Wire Format（二进制编码） | 不需要 | protobuf 2~6 共享同一套 wire format，是协议级承诺 |
| Runtime API（`google.protobuf`） | 各自独立 | 只要本机 generated code 和 runtime 匹配即可 |

### 类比

- 就像 **TCP/IP 协议**：Windows 10 和 Windows 11 的 TCP 实现细节不同，但包结构一样，能互通
- `protoc` 是编译器，`protobuf` runtime 是标准库，只要 generated code 只调用 runtime 里已有的 API，版本可以不完全一致

## 文件清单

```
galbot_vla_client/
  server.py              # FastAPI 主服务，编排任务流程
  robot_controller.py    # galbot SDK 封装：导航、摄像头、机械臂、夹爪、set_joint_commands
  vla_client.py          # gRPC 客户端，直连 PolicyServer（不依赖 lerobot）
  config.yaml            # 导航位姿、放置位姿、VLA 配置、控制参数
  info.json              # 机器人关节维度、摄像头信息
  requirements.txt       # Python 3.8 依赖
  lerobot/
    transport/
      services_pb2.py      # 由 protoc 5.26.1 重新生成，兼容 protobuf 4.25
      services_pb2_grpc.py # gRPC Stub/Servicer，兼容 protobuf 4.25
```

## 部署步骤

### 1. 远端服务器启动 PolicyServer

```bash
cd /home/galbot/VLA_PROJECTS/lerobotpiper
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080
```

### 2. 机器人端安装依赖

```bash
cd /home/galbot/VLA_PROJECTS/galbot_vla_client
pip3 install -r requirements.txt
```

### 3. 机器人端启动 Task Server

```bash
cd /home/galbot/VLA_PROJECTS/galbot_vla_client
python3 server.py   # 监听 0.0.0.0:8000
```

## 需要你调整的配置

编辑 `galbot_vla_client/config.yaml`：

```yaml
vla:
  policy_server_address: "<远端服务器IP>:8080"   # PolicyServer gRPC 地址
  pretrained_name_or_path: "/远端服务器上模型路径"
  device: "cuda"

robot:
  navigation:
    shelf_pose: [x, y, z, qx, qy, qz, qw]    # 货架前导航位姿
    table_pose: [x, y, z, qx, qy, qz, qw]    # 桌子前导航位姿
  placement:
    fixed_target_pose: [x, y, z, qx, qy, qz, qw]  # 放下水瓶的固定目标位姿
```

## 接口调用示例

```bash
# 执行完整任务
curl -X POST http://localhost:8000/execute_task \
  -H "Content-Type: application/json" \
  -d '{"instruction": "pick up the water bottle"}'

# 只执行 VLA 部分（不导航、不放置）
curl -X POST http://localhost:8005/vla_only \
  -H "Content-Type: application/json" \
  -d '{"instruction": "pick up the water bottle"}'
```

## 安全设计

- `set_joint_commands` 不插值，直接驱动到目标位置。因此在 `execute_action_chunk` 中加入了**每步最大关节变化限制**（`max_joint_delta_per_step`），防止目标与当前位置差距过大导致运动过快
- 导航带碰撞检测和重试机制
- VLA 执行有最大步数上限，防止无限循环
