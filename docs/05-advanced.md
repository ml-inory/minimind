# Assignment 05（可选）：进阶方向

主线作业之外的扩展，按推荐顺序排列。

## 1. LoRA

文件：`model/model_lora.py`、`trainer/train_lora.py`

全参数微调成本高，LoRA 冻结原权重，只训练两个低秩矩阵：

```text
W_new = W + B @ A,   A: (rank, out), B: (out, rank)
```

建议练习：

1. 自己实现 `LoRA.forward`；
2. 解释 `apply_lora` 为什么用闭包保留 `original_forward`；
3. 理解 `save_lora` 为什么只保存 `*.lora.*` 参数；
4. 写一个 `merge_lora`：把 `B @ A` 合并回原 `weight`。

## 2. MoE

文件：`model/model_minimind.py::MOEFeedForward`

`MOEFeedForward` 已保留完整实现，是很好的“对照阅读材料”。观察：

- router gate 如何输出 top-k 专家；
- `norm_topk_prob` 为什么做归一化；
- 训练时如何用 `load * scores.mean` 计算辅助均衡 loss。

可以把它手工重写一遍并关闭原实现进行 A/B 测试。

## 3. KV Cache 与增量生成

`Attention.forward` 的 `past_key_value` 参数就是 KV cache：已经算过的 key/value 不再重复计算。
`MiniMindModel.forward` 中 `start_pos` 决定本次 RoPE 从哪个位置开始。

建议：在 `generate` 上做实验，比较 `use_cache=True/False` 的时间和输出一致性。

## 4. YaRN 长度外推

`precompute_freqs_cis` 的 `rope_scaling` 分支对应 YaRN。读完论文后，尝试解释 `beta_fast`、`beta_slow`、`factor` 的作用。

## 5. RLHF / RLAIF

仓库中的 `train_ppo.py`、`train_grpo.py`、`train_dpo.py` 是完整实现。主线作业完成后，可以按 DPO → GRPO → PPO 的顺序阅读：

1. 奖励从哪来；
2. reference model 为什么 stop-gradient；
3. KL 惩罚为什么能防止策略漂移。

这部分不设自动测试，属于开放阅读任务。
