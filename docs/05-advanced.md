# Assignment 05（可选）：进阶方向

主线作业完成后，下面每个方向都可以独立成为一个研究型小项目。

## 1. LoRA

文件：`model/model_lora.py`（当前保留完整实现）

练习方式：

1. 阅读 LoRA 论文，回答：为什么两个低秩矩阵的初始化方式不同？
2. 不看原实现，在 `model_lora_exercise.py` 中自己写出 `LoRA.forward` 与 `apply_lora`；
3. 比较你的实现与原始实现，找出你漏掉的工程细节；
4. 回答：为什么保存 LoRA 时只需要保存低秩矩阵，不需要保存整个模型？

论文：[Hu et al., LoRA](https://arxiv.org/abs/2106.09685)

## 2. MoE

文件：`model/model_minimind.py::MOEFeedForward`

阅读 Switch Transformer 与 Mixtral 后回答：

1. router 为什么需要 top-k 而不是只选最大的专家？
2. “专家负载不均衡”会导致什么问题？辅助 loss 为什么能缓解？
3. MoE 的参数量和实际激活参数量为什么不同？

论文：[Switch Transformers](https://arxiv.org/abs/2101.03961)、[Mixtral](https://arxiv.org/abs/2401.04088)

## 3. KV Cache

自己实现一个实验：

1. 关闭 cache 逐 token 生成；
2. 打开 cache 逐 token 生成；
3. 对比速度与输出是否一致。

思考：cache 里存的是什么？为什么“只生成一个 token”时前向仍然要跑全部层？

## 4. YaRN

阅读 YaRN 论文后，在 `precompute_freqs_cis` 中解释每个超参数的几何含义：

1. `original_max_position_embeddings`；
2. `factor`；
3. `beta_fast` / `beta_slow`；
4. `attention_factor`。

论文：[Peng et al., YaRN](https://arxiv.org/abs/2309.00071)

## 5. RLHF / RLAIF

阅读顺序建议：DPO → GRPO → PPO。

每个算法回答同一个问题：奖励从哪来、策略如何更新、如何防止策略漂移。

论文：[DPO](https://arxiv.org/abs/2305.18290)、[GRPO](https://arxiv.org/abs/2402.03300)、[PPO](https://arxiv.org/abs/1707.06347)

## 6. 参考答案

进阶题的答案偏开放性，提供答题框架而非唯一答案：

```bash
git show solutions:docs/answers/05-advanced.md
```
