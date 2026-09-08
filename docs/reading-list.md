# 论文阅读清单

按作业阶段分类。每篇都标注“读到哪里”，请带着问题阅读，不要只看摘要。

## 阶段 0：前置背景

| 资料 | 建议阅读内容 |
|------|--------------|
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | 整体注意力机制和数据流 |
| [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) | 完整实现视角，对照 PyTorch 阅读 |

## 阶段 1：数据与语言建模目标

| 论文/资料 | 建议阅读内容 |
|-----------|--------------|
| Sennrich et al., [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) | BPE 分词：为什么需要 subword |
| Radford et al., [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | GPT-2 的语言建模目标、input/label 错位关系 |
| Ouyang et al., [InstructGPT](https://arxiv.org/abs/2203.02155) | SFT 为什么只让模型学习“回答” |
| Zhou et al., [LIMA](https://arxiv.org/abs/2305.11206) | 少而精的对齐数据 |

## 阶段 2：模型结构

| 论文/资料 | 建议阅读内容 |
|-----------|--------------|
| Ba et al., [Layer Normalization](https://arxiv.org/abs/1607.06450) | LayerNorm 的动机与公式 |
| Zhang & Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | RMSNorm：去掉 mean 后发生了什么 |
| Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | RoPE 的核心章节与推导 |
| Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Scaled Dot-Product Attention、多头注意力 |
| Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) | GQA：为什么共享 KV head |
| Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | SwiGLU 的来源 |
| Xiong et al., [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) | Pre-Norm vs Post-Norm |
| Touvron et al., [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | 现代实现细节：RMSNorm、RoPE、SwiGLU 组合方式 |

## 阶段 3：训练

| 论文/资料 | 建议阅读内容 |
|-----------|--------------|
| Loshchilov & Hutter, [AdamW](https://arxiv.org/abs/1711.05101) | AdamW 与权重衰减 |
| Loshchilov & Hutter, [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) | cosine 学习率衰减 |
| Brown et al., [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | GPT-3 训练配方：batch、梯度裁剪、学习率 |
| Micikevicius et al., [Mixed Precision Training](https://arxiv.org/abs/1710.03740) | FP16/BF16、loss scaling |
| Karpathy, [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) | 训练调试的一般方法 |

## 阶段 4：生成解码

| 论文/资料 | 建议阅读内容 |
|-----------|--------------|
| Fan et al., [Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833) | top-k 采样 |
| Holtzman et al., [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) | top-p（nucleus）采样、退化问题 |

## 阶段 5（可选）：进阶

| 论文/资料 | 建议阅读内容 |
|-----------|--------------|
| Hu et al., [LoRA](https://arxiv.org/abs/2106.09685) | 低秩适配：初始化与合并 |
| Fedus et al., [Switch Transformers](https://arxiv.org/abs/2101.03961) | MoE 路由与负载均衡 |
| Jiang et al., [Mixtral of Experts](https://arxiv.org/abs/2401.04088) | 现代 MoE 实践 |
| Peng et al., [YaRN](https://arxiv.org/abs/2309.00071) | RoPE 长度外推 |
| Schulman et al., [PPO](https://arxiv.org/abs/1707.06347) | 强化学习策略优化 |
| Rafailov et al., [DPO](https://arxiv.org/abs/2305.18290) | 直接偏好优化 |
| Shao et al., [GRPO](https://arxiv.org/abs/2402.03300) | 组相对策略优化 |

> 阅读提示：不要只找“公式”，先回答每个阶段文档里的“为什么”问题，
> 再带着问题去论文里找作者当时解决的是什么缺陷。
