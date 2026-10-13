# Base Infer

原地测试 pick 模型的最小 client：恢复训练起始姿态、采图、请求 PolicyServer，
并把返回的双臂动作下发给机器人。不导航、不走行为树。Ctrl+C 停止。

图像处理按数采相机 para（640×480）对齐训练 origin：

- SDK 压缩帧解码为 RGB `uint8`；
- 三路相机原生已是 640×480（4:3），不再裁工作模式 16:9 腕部；
- 只等比例 letterbox 一次到 `pick_place_balence` 要求的 `224x224x3`（上下各约 28 px）；
- 原生尺寸必须是 640×480，否则直接报错，避免误用工作模式 1280×720；
- 推理前严格检查图像尺寸、类型、非全黑，以及 23 维全身关节状态。

动作执行：

- pick 模型输出 16 维（双臂 + 夹爪）；腿和头保持当前实测值，不下发；
- 每个 chunk 先执行前 `execute_actions_per_chunk` 帧（默认 20），再采下一帧图推理；
- 30Hz 轨迹线性插值到 250Hz，带关节限位和步长限制。

## 使用

先编辑 `config.yaml` 中的 PolicyServer 地址、checkpoint 和 instruction，然后运行：

```bash
cd /home/galbot/project_yuqiz/base_infer
python3 infer.py
```

默认最多跑 `max_steps` 次推理循环（当前 400）。只推理、不下发：

```bash
python3 infer.py --dry-run
```

如果 PolicyServer 已经加载好模型：

```bash
python3 infer.py --skip-policy-setup
```

机器人已经在正确位置时，可跳过起始姿态：

```bash
python3 infer.py --skip-initial-pose
```

预测写入 `outputs/predictions.jsonl`，对应的三路实际发送图像写入
`outputs/images/`。终端会同时打印 SDK 原始尺寸和发送尺寸，便于检查是否发生了
非预期的图像处理。

当前 checkpoint 固定为：

```text
/home/zhangyuqi/zhangyuqi/G1-20260911/g1/0914/pick_place_balence/checkpoints/040000/pretrained_model
```
