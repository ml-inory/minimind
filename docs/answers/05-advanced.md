# Assignment 05 参考答案（答题框架）

## LoRA

设原始权重为 `W`，低秩矩阵为 `A`、`B`：

```text
W_new = W + B @ A
```

- `A` 用随机初始化，`B` 用 0 初始化，这样训练开始时增量 `B @ A = 0`，不破坏原权重；
- `apply_lora` 通过闭包保留 `original_forward`，在前向输出上叠加 `lora(x)`，避免直接改写权重；
- 训练时只更新 `A/B`，因此保存时只需要 `*.lora.*`；
- 推理前可把 `B @ A` 合并回 `W`，得到等价于完整微调结果的稠密权重。

## MoE

- router 的 `top-k` 决定每个 token 激活哪些专家；k 越大表达容量越高、计算开销越大，
  k=1 最稀疏。现代实现（如 Mixtral）常用 top-2；
- 若所有 token 总选同一批专家，其余专家得不到训练，模型容量被浪费；
  辅助均衡 loss 鼓励 router 让各专家负载接近；
- MoE 的总参数量 = shared/dense 参数 + 全部专家参数，
  而激活参数量 = 每 token 实际经过的专家参数，所以两者不同。

## KV Cache

- cache 保存每个注意力层的 key/value 历史，不保存 query；
- 增量生成时新 token 的 query 仍要与所有历史 key 做 attention，
  因此每一层都要前向，只是 key/value 不用从头重算；
- 可以对比 `use_cache=True/False` 的输出是否一致、耗时差异。

## YaRN

- `original_max_position_embeddings`：模型原始训练长度；
- `factor`：目标外推倍率；
- `beta_fast` / `beta_slow`：决定哪些频率进入插值过渡区；
- `attention_factor`：补偿长上下文下注意力熵变化，稳定注意力分布。

建议以 YaRN 论文第 3 节为主，结合代码逐行对照。

## RLHF / RLAIF

三个算法的统一问题是：

1. **奖励从哪来**：RM 打分、规则、AI 反馈或同组样本相对比较；
2. **策略如何更新**：DPO 直接优化偏好损失；PPO/GRPO 用重要性采样与裁剪更新策略；
3. **如何防止漂移**：KL 惩罚（相对 reference policy）、裁剪、组内相对奖励。

答案没有唯一模板；能用自己的话讲清“奖励来源 → 更新方式 → 约束机制”即可。
