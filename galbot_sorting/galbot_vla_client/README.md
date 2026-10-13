# Galbot VLA Client

机器人端 VLA 任务服务（Python 3.8），通过 gRPC 直连远端 PolicyServer。

## 启动

```bash
cd /home/galbot/VLA_PROJECTS/galbot_vla_client
python3 server.py
```

## 接口

- `POST /execute_task` - 执行完整任务（导航 + VLA + 放置）
- `POST /vla_only` - 仅执行 VLA 推理
- `GET /health` - 健康检查

## 配置

编辑 `config.yaml`：
- `vla.policy_server_address` - 远端 PolicyServer IP:端口
- `vla.pretrained_name_or_path` - 远端模型路径
- `robot.navigation.shelf_pose/table_pose` - 导航位姿
- `robot.placement.fixed_target_pose` - 固定放置位姿

## Proto 文件与重新生成

如果远端 PolicyServer 的 proto 协议有变更，需在机器人端重新生成 `pb2` 文件：

```bash
cd /home/galbot/VLA_PROJECTS/galbot_vla_client

# 使用 grpcio-tools 自带的 protoc（5.26.1）生成，兼容 protobuf 4.25.4 runtime
# -I . 表示以当前目录为 include 根，protoc 会找到 ./lerobot/transport/services.proto
python3 -m grpc_tools.protoc \
  -I . \
  --python_out=. \
  --grpc_python_out=. \
  ./lerobot/transport/services.proto
```

生成后会覆盖：
- `lerobot/transport/services_pb2.py`
- `lerobot/transport/services_pb2_grpc.py`

> 注意：必须使用 grpcio-tools 自带的 protoc（当前环境为 5.26.1），不能用 protobuf 6.x 的 protoc，否则生成的代码会包含 `runtime_version` 检查，导致 Python 3.8 + protobuf 4.25.4 无法导入。
