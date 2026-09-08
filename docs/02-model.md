# Assignment 02：实现 MiniMind 网络结构

## 1. 本阶段目标

从零实现一个可训练的最小 Decoder-Only 语言模型。
完成后，给定 `input_ids`，模型应能输出 logits，并在提供 labels 时返回 loss。

## 2. 任务总览

| 编号 | 位置 | 内容 |
|------|------|------|
| Task A | `RMSNorm.norm` | 归一化层 |
| Task B | `precompute_freqs_cis` | RoPE 频率表 |
| Task C | `precompute_freqs_cis` | 位置相关的 cos/sin |
| Task D | `apply_rotary_pos_emb` | 旋转位置编码应用 |
| Task E | `Attention.forward` | 自注意力 |
| Task F | `FeedForward.forward` | 前馈网络 |
| Task G | `MiniMindBlock.forward` | Transformer Block |
| Task H | `MiniMindForCausalLM.forward` | 语言建模 loss |

## 3. 引导问题

请先通过阅读回答这些问题，再开始实现：

### 关于归一化

1. LayerNorm 做了什么？RMSNorm 去掉了哪一步？为什么可以去掉？
2. 归一化应该对哪一个维度做？权重参数的形状是什么？

### 关于位置编码

3. 原始 self-attention 为什么对位置不敏感？
4. RoPE 的核心思想是“旋转”而不是“相加”，它旋转的是什么？
5. 频率推导只给出了 `dim/2` 个角度，而最终张量需要 `dim` 列；结合“旋转半区”的几何意义思考如何补齐。
6. 频率公式里 `rope_base` 的指数为什么只取偶数维？它和复数旋转有什么关系？

### 关于注意力

7. Scaled Dot-Product Attention 的 scale 从哪来？不 scale 会怎样？
8. causal mask 保证什么性质？实现时 mask 应加到 score 的哪些位置？
9. GQA 和 MHA 的 key/value head 数量关系是什么？`repeat_kv` 为什么要复制？
10. KV cache 里存的是哪两个张量？增量生成时 RoPE 的位置从哪里开始？

### 关于前馈网络与残差

11. SwiGLU 里的“门控”指什么？为什么它比普通 ReLU 更常用？
12. Pre-Norm 的 residual 连接应该加在哪里？norm 应该放在 residual 之前还是之后？

### 关于目标函数

13. “预测下一个 token”具体指什么？logits 和 labels 在时间维上如何对齐？
14. 为什么要用 `ignore_index=-100`？

## 4. 参考资料

见 [论文阅读清单](reading-list.md) 的“阶段 2：模型结构”。

建议阅读顺序：

1. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)——建立整体直觉；
2. RMSNorm 论文——Task A；
3. RoFormer——Task B/C/D；
4. Attention Is All You Need——Task E；
5. GLU Variants Improve Transformer——Task F；
6. On Layer Normalization in the Transformer Architecture——Task G；
7. GPT-2——Task H 的语言建模目标。

## 5. 验收

```bash
python3 -m pytest tests/test_model_components.py -v
```

测试会验证：

- RMSNorm 的数值结果；
- RoPE 频率表与旋转结果；
- FeedForward 的数值结果；
- Attention 与标准 PyTorch 实现的等价性；
- 完整模型的 logits 形状和 loss 对齐方式。

## 6. 完成自检

实现完成后，不看代码回答：

1. 每个张量在进入 attention 前后的形状变化是什么？
2. causal mask 为什么能让第 i 个位置只看前 i 个位置？
3. 如果去掉 q/k norm、RoPE、residual 中的任意一个，模型训练会发生什么？

回答完后再对照答案：

```bash
git show solutions:docs/answers/02-model.md
```
