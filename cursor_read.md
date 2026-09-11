# PI05 推理图片预处理：必须与训练 forward 完全一致

请修改推理服务器的图片输入处理，使其与 PI05 训练时送入 `policy.forward()` 的图片完全一致。
本服务器无法访问训练服务器代码，因此以下规则是完整规范，请不要依赖其他本地文件推断处理方式。

## 训练时的实际流程

训练数据来自 4 路相机视频，顺序和 key 固定为：

1. `observation.images.head_left`
2. `observation.images.head_right`
3. `observation.images.left_arm`
4. `observation.images.right_arm`

每路视频帧原始尺寸为 `640x480x3`，RGB，`uint8`，像素范围 `[0, 255]`。

处理顺序如下：

```python
# frame: uint8, shape [H, W, 3] 或 tensor [3, H, W]
frame = frame.to(torch.float32) / 255.0       # [0, 255] -> [0, 1]
frame = resize_with_pad(frame, 224, 224)     # 保持比例缩放，不裁剪，黑边补齐
frame = frame * 2.0 - 1.0                    # [0, 1] -> [-1, 1]
```

由于原图比例为 4:3，`640x480` 会先缩放为约 `224x168`，再在上下补黑边，最终为 `224x224`。

## 必须遵守

- 不要做任何 crop（包括 `head_camera_roi`、center crop）、horizontal flip、旋转或颜色增强。
- 当前不需要相机视场对齐；请禁用或绕过 `head_camera_roi`，四路相机统一按下面的训练流程处理。
- 不要改变相机顺序。
- 不要交换 RGB/BGR 通道；训练使用 RGB。
- 不要使用 ImageNet mean/std 做额外归一化。训练配置中的 `VISUAL: IDENTITY` 不做 mean/std 标准化。
- 最终送入 PI05 vision encoder 的数值范围必须是 `[-1, 1]`。
- 最终 tensor 形状应为 `[B, 3, 224, 224]`；4 路相机应作为 4 个 image tensor 传给 PI05。

## 避免重复处理

请先确认推理调用链：

- 如果推理调用 `PI05Policy.select_action()` / `predict_action_chunk()`，这些函数内部会调用 `_preprocess_images()`，此时外部只应提供 RGB 图片，并完成 `uint8 -> float32 / 255`（若调用链已经完成这一步，则不要再除一次）。不要在外部再次做 `*2-1` 或 resize。
- 如果推理服务器直接调用底层模型 `model.forward()`，则服务器必须自己完成上面的完整流程，包括 resize-with-pad 和 `*2-1`。
- 最终只能做一次 `/255`、一次 resize-with-pad、一次 `*2-1`，避免重复归一化。

## 当前推理端的处理顺序

当前不使用 `head_camera_roi`。四路相机都按同一流程处理：

```text
SDK 解码原图（不裁剪）
    → 如分辨率不是 640x480，仅 resize 到训练特征尺寸 640x480（不裁剪）
    → /255
    → PI05 resize-with-pad（224x224）
    → *2-1
```

如果推理代码中存在 `_apply_head_camera_roi()`，本次请禁用其调用；不要在头相机上使用任何 ROI。不要为了适配分辨率而额外裁剪，交给 PI05 的 `resize_with_pad` 保持比例并补黑边。

## 参考训练代码

- 训练循环中的 `/255`：
  `/home/zhangyuqi/zhangyuqi/lerobot/src/lerobot/scripts/lerobot_train.py`
- PI05 的 resize-with-pad、`[-1,1]` 归一化：
  `/home/zhangyuqi/zhangyuqi/lerobot/src/lerobot/policies/pi05/modeling_pi05.py`

请在修改后打印一次推理输入的 shape、dtype、min、max，并确认类似：

```text
shape=[1, 3, 224, 224], dtype=float32, min=-1.0, max=1.0
```

## 可直接实现的独立伪代码

```python
for camera_name in ["head_left", "head_right", "left_arm", "right_arm"]:
    image = get_rgb_image(camera_name)  # RGB, uint8, HxWx3

    # 不做 ROI、不做 crop；若 SDK 分辨率不是训练特征尺寸，只做普通 resize。
    if image.shape[:2] != (480, 640):
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)

    image = image.astype(np.float32) / 255.0  # [0,255] -> [0,1]

    # 等比例 resize + 居中黑边，得到 224x224（不要裁剪）
    h, w = image.shape[:2]
    ratio = max(w / 224.0, h / 224.0)
    new_w, new_h = int(w / ratio), int(h / ratio)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded = np.zeros((224, 224, 3), dtype=np.float32)
    top, left = (224 - new_h) // 2, (224 - new_w) // 2
    padded[top:top + new_h, left:left + new_w] = resized

    image = padded * 2.0 - 1.0  # [0,1] -> [-1,1]
    image = torch.from_numpy(image).permute(2, 0, 1)  # [3,224,224]
```

四路图片分别传给 PI05，不能拼接成一张图片。若 policy API 自己已经做了 `resize_with_pad` 和 `*2-1`，外部不要重复做；请根据实际调用链保留唯一一套预处理。
