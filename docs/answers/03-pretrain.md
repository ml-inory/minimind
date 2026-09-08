# Assignment 03 参考答案

## 学习率

### 1. 后期为什么要降低学习率？

训练后期参数已经接近较优区域，大步长容易来回震荡甚至发散；
小步长允许更精细地收敛。

### 2. cosine 衰减的形状

学习率从峰值按 cosine 曲线平滑下降，到训练结束前达到一个最小值。

### 3. 是否衰减到 0？

通常不衰减到 0，而是保留一个下限（例如最大学习率的 10%）：

```text
lr_min = 0.1 * lr
```

MiniMind 使用的公式等价于：

```python
lr_min + 0.5 * (lr - lr_min) * (1 + cos(pi * current_step / total_steps))
```

## 训练 step

### 4. 输入与 loss

前向输入是 `(input_ids, labels)`。返回的 loss 由两部分组成：

```text
logits_loss + aux_loss
```

Dense 模型没有 MoE 辅助 loss，`aux_loss` 为 0。

### 5. autocast 与 loss scaling

autocast 让矩阵乘等在低精度（FP16/BF16）下计算，节省显存和加速；
FP16 梯度容易下溢，loss scaler 先把 loss 放大再 backward，更新前缩放回原尺度。

### 6. 梯度累积

累积 K 个 micro-batch 的梯度后再更新，等价于一个更大的 batch。
每个 micro-batch 的 loss 先除以 K，否则 K 次梯度相加会让总更新幅度放大 K 倍。

### 7. step 与 zero_grad 时机

每累积满 `accumulation_steps` 次才 `optimizer.step()`；
`zero_grad(set_to_none=True)` 应在 step 之后调用，准备下一轮累积。

### 8. 梯度裁剪

裁剪的是参数梯度的全局范数（`clip_grad_norm_`），
防止个别 batch 产生超大梯度导致参数一步“飞”出正常区域。

## 完整流程

### 9. loss 不下降先查什么

先查“数据 → forward → loss → backward”链路是否能跑通且数值合理：

- 初始 loss 是否接近 `log(vocab_size)`；
- labels 是否正确 shift；
- 学习率是否真的在更新；
- 有没有 accidentally `model.eval()` 或梯度被清零。

不要一上来就调大学习率。

### 10. eval/train

保存 checkpoint 时 `eval()` 会关闭 dropout，保证保存的是确定性的推理状态；
保存完必须 `train()` 恢复训练模式。

## 完成自检

1. 去掉 `loss / accumulation_steps` 会让一次更新相当于放大了 `accumulation_steps` 倍的学习率。
2. `zero_grad` 放在 step 之前会清掉 step 正要使用的梯度，更新变成空操作；
   放在 step 之后才能正确开始下一轮累积。
3. 常见 NaN 来源：数据含 NaN、学习率过大、loss 计算错误、AMP 下 FP16 溢出、
   梯度爆炸、logits/labels 错位。可在 forward/backward 后加 `torch.isfinite` 断言定位。
