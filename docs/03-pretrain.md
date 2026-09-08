# Assignment 03：预训练训练循环

网络结构完成后，模型还只是“随机初始化”。预训练的目的是让它学会**根据前文预测下一个 token**。

## 1. 一个训练 step 的完整流程

```text
DataLoader 取出 (input_ids, labels)
  → 计算 cosine 学习率并写入 optimizer
  → autocast 下 forward：model(input_ids, labels=labels)
  → loss = logits_loss + aux_loss（MoE 辅助 loss，dense 时为 0）
  → loss /= accumulation_steps
  → loss.backward()
  → 每 accumulation_steps 次：
       scaler.unscale_(optimizer)
       clip_grad_norm_(model.parameters(), grad_clip)
       scaler.step(optimizer)
       scaler.update()
       optimizer.zero_grad(set_to_none=True)
```

## 2. 任务 A：cosine 学习率 `get_lr`

文件：`trainer/trainer_utils.py`

原实现使用 cosine 衰减，但不会衰减到 0，而是到一个最小值：

```text
lr_min = 0.1 * lr
progress = current_step / total_steps
lr(current) = lr_min + 0.5 * (lr - lr_min) * (1 + cos(pi * progress))
```

验证：

```bash
python3 -m pytest tests/test_trainer_utils.py -v
```

## 3. 任务 B：`train_pretrain.py::train_epoch`

文件：`trainer/train_pretrain.py`

脚本已经把数据、模型、优化器、混合精度 scaler 都准备好，`train_epoch` 里每一行都标了顺序。
你需要补：

1. 当前全局 step 的学习率，并写入 `optimizer.param_groups`；
2. 在 `autocast_ctx` 内调用 `model(input_ids, labels=labels)`；
3. 把 `res.loss + res.aux_loss` 除以 `args.accumulation_steps`；
4. `scaler.scale(loss).backward()`；
5. 每 `accumulation_steps` 步做 unscale、梯度裁剪、`scaler.step/update`、清零梯度。

日志、checkpoint、尾部残差更新已经保留。

为什么要梯度累积？小显存时无法直接放大 batch，就先用小 batch 累积多次梯度再更新一次，等效于更大的 batch。
为什么要 grad clip？防止 loss spike 时梯度爆炸，把梯度范数限制到 1.0。

## 4. 本地冒烟运行

用很小的模型和几条文本快速验证整个脚本：

```bash
cd trainer
python3 train_pretrain.py \
  --epochs 1 \
  --batch_size 2 \
  --max_seq_len 16 \
  --hidden_size 16 \
  --num_hidden_layers 2 \
  --num_workers 0 \
  --accumulation_steps 1 \
  --learning_rate 1e-3 \
  --save_interval 1 \
  --log_interval 1 \
  --device cpu \
  --data_path ../tests/data/pretrain_toy.jsonl \
  --save_dir /tmp/mm-homework-out \
  --from_weight none
```

看到 loss 并成功保存 `pretrain_16.pth` 即通过。

自动版（标记为 slow，会花几十秒）：

```bash
python3 -m pytest tests/test_pretrain_smoke.py -v
```

## 5. 真正的预训练

下载 MiniMind 数据后（见 [README_original.md](../README_original.md)）：

```bash
cd trainer
python3 train_pretrain.py
```

对于 768 维模型，这是 64M 参数级别，单张 3090 上约 1~2 小时。
建议先用 `--hidden_size 64 --num_hidden_layers 2` 验证流程，再放大。
