# Assignment 02：实现 MiniMind 网络结构

完成本作业后，你应该能从输入 token 自己推出 hidden states、logits 和 loss。

## 1. 整体数据流

```text
input_ids (B, S)
  → Embedding (B, S, H)
  → Dropout
  → [MiniMindBlock] × N
      → Pre-Norm → Attention（含 RoPE、GQA、causal mask）
      → residual add
      → Pre-Norm → FeedForward（SwiGLU）
      → residual add
  → final RMSNorm
  → lm_head → logits (B, S, V)
  → CrossEntropyLoss（shift 一个 token）
```

变量含义：`B=batch size`，`S=seq len`，`H=hidden_size`，`V=vocab_size`。

## 2. 任务 A：`RMSNorm.norm`

文件：`model/model_minimind.py`

RMSNorm 与 LayerNorm 类似，但没有减均值：

```text
rms(x) = sqrt(mean(x^2, dim=-1) + eps)
y = weight * (x / rms(x))
```

`norm` 只负责归一化部分（不含 `weight`），`forward` 负责乘 weight。

为什么需要归一化？Pre-Norm 让每一层输入的方差稳定，深层 Transformer 才能稳定训练。

## 3. 任务 B/C：RoPE 频率表 `precompute_freqs_cis`

Transformer 本身不感知 token 顺序，需要把位置信息注入 attention。
RoPE 的做法是：对 query/key 的相邻维度做旋转，旋转角度随位置线性增加。

先计算基础频率：

```text
freqs[i] = 1 / rope_base^(2i / dim),  i = 0, 1, ..., dim/2-1
```

然后用位置下标做外积：

```text
angles[t, i] = t * freqs[i]
```

返回的 `freqs_cos` / `freqs_sin` 形状都是 `(end, dim)`：

```text
cos = cos(angles) 拼接成 dim 维
sin = sin(angles) 拼接成 dim 维
```

`rope_scaling`（YaRN）和最终 `cos/sin` 的拼接代码已经给你，不要改动；你只需要补基础频率的计算。读完函数后，分清哪几行是已经给出的工程逻辑。

## 4. 任务 D：`apply_rotary_pos_emb`

`rotate_half(x)` 把张量沿最后一维切成前后两半，交换并取负前半：

```text
rotate_half([a, b]) = [-b, a]
```

位置旋转：

```text
q_embed = q * cos + rotate_half(q) * sin
k_embed = k * cos + rotate_half(k) * sin
```

`cos`、`sin` 通过 `unsqueeze(unsqueeze_dim)` 变成可广播形状。

## 5. 任务 E：`Attention.forward`

MiniMind 使用 **GQA（Grouped-Query Attention）**：

```text
q_heads = 8
kv_heads = 4
每个 q head 组共享同一份 key/value head（n_rep = 8/4 = 2）
```

`repeat_kv` 函数已经给出，用于把 kv head 复制到 q head 数量。
`Attention.forward` 中已给出投影、reshape、norm、RoPE、past KV 拼接、flash attention 分支，你需要实现的是 **非 flash 的经典 attention 路径**：

1. 计算 scaled scores：

```text
scores = q @ k^T / sqrt(head_dim)
形状: (B, n_local_heads, S_q, S_kv)
```

2. 加 causal mask：`scores[:, :, :, -seq_len:]` 上三角位置设为 `-inf`，使位置 `t` 看不到 `t` 之后的 token；
3. 可选加 attention mask（`1` 表示可见，`0` 表示不可见），用 `-1e9` 或 `-inf` 屏蔽；
4. `softmax(dim=-1)` → dropout → 乘 `v`。

为什么 scale 是 `1/sqrt(head_dim)`？防止点积随维度增大而过大、softmax 过早饱和。

## 6. 任务 F：`FeedForward.forward`

MiniMind 使用 SwiGLU：

```text
FFN(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
```

`ACT2FN[config.hidden_act]` 已经选好激活函数，三个线性层也已定义。

## 7. 任务 G：`MiniMindBlock.forward`

标准 Pre-Norm + residual：

```text
residual = x
x = self_attn( input_layernorm(x) ) + residual
residual = x
x = mlp( post_attention_layernorm(x) ) + residual
```

`self_attn` 返回 `(hidden_states, present_key_value)`，需要正确解包。

## 8. 任务 H：`MiniMindForCausalLM.forward` 的 loss

训练目标是“预测下一个 token”，所以 logits 和 labels 要错开一位：

```text
pred  = logits[:, :-1]      # 用 t 位置预测 t+1
target= labels[:, 1:]
loss  = CrossEntropyLoss(pred, target, ignore_index=-100)
```

不要忘记把 logits reshape 成 `(num_tokens, vocab_size)`。

## 9. 自测

```bash
python3 -m pytest tests/test_model_components.py -v
```

测试覆盖：RMSNorm 数值、SwiGLU 数值、causal attention 与标准 SDPA 一致性、最终 logits/loss。

## 10. 检查你的理解

- 为什么 `q` 要乘以 `cos` 并加上旋转后的部分？
- 为什么 attention 需要 causal mask？
- GQA 相比 MHA 省了多少 KV cache？
- `-100` 的 label 为什么不会贡献梯度？
