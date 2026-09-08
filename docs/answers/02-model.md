# Assignment 02 参考答案

## 归一化

### 1. LayerNorm 与 RMSNorm

LayerNorm 对最后一维做：

```text
(x - mean) / sqrt(var + eps) * weight + bias
```

RMSNorm 去掉减均值和 bias，只做：

```text
x / sqrt(mean(x^2, dim=-1) + eps) * weight
```

在 Transformer 中，残差连接已经承担了部分“稳定信号中心”的作用，
去掉均值项在实证中影响很小，却能省去统计均值的计算。

### 2. 归一化维度与权重形状

对最后一维（hidden features）归一化；`weight` 形状为 `(hidden_size,)`。

## 位置编码

### 3. 为什么 self-attention 对位置不敏感？

attention score 只取决于 query 与 key 的内容点积，交换任意两个 token 的输入顺序，
对应的 score 集合不变，因此模型看不到“先后关系”。

### 4. RoPE 旋转的是什么？

RoPE 把每个位置的 query/key 向量按“位置编号 × 频率”旋转一个角度，
使两个位置的 attention score 只依赖它们的**相对位置差**。

### 5. `dim/2` 个角度如何变成 `dim` 列？

把向量看成前后两个半区 `[a, b]`，第 i 对由 `a[i]` 与 `b[i]` 组成。
每个频率只对应一对坐标，所以只有 `dim/2` 个频率。
实现频率表时把每个 `cos/sin` 半段重复一次，使前半区和后半区都能乘上相同角度的旋转量。

### 6. 为什么指数只取偶数维？

RoFormer 用二维旋转矩阵表示一个复数乘法：

```text
theta_i 对应复数平面上的一个旋转
```

每个频率需要两个相邻通道（实部/虚部）。偶数下标 `2i` 产生频率 i，
等价于把 `dim/2` 个 2D 旋转铺满整个向量。

## 注意力

### 7. scale 从哪来？

如果 q、k 每个分量是单位方差随机变量，点积的方差约等于 `head_dim`，
标准差约 `sqrt(head_dim)`。不缩放会让点积过大、softmax 进入饱和区、梯度变小。
因此除以 `sqrt(head_dim)`。

### 8. causal mask

位置 i 只能看到 j <= i。mask 加到当前 query/key 长度对应的 score 矩阵上三角：

```text
scores[:, :, :, -seq_len:] 的上三角位置 += -inf
```

softmax 后这些位置的权重为 0。

### 9. GQA

多个 query head 共享一组 key/value head：

```text
n_rep = num_attention_heads / num_key_value_heads
```

`repeat_kv` 把 KV head 复制 `n_rep` 份，使形状与 query head 数量一致，才能做 batched matmul。

### 10. KV cache

cache 保存历史 `key` 与 `value`。增量生成时只输入最后一个 token，
它对应的 RoPE 起始位置应等于 `past_len`，而不是 0。

## 前馈与残差

### 11. SwiGLU

门控指用一个经过激活的投影（gate）去乘另一个投影（up），
让每一维的通过量由输入决定，而不是固定激活：

```text
down_proj(act(gate_proj(x)) * up_proj(x))
```

### 12. Pre-Norm residual

```text
h = x + self_attn(input_layernorm(x))
x = h + mlp(post_attention_layernorm(h))
```

norm 先作用于子层输入，residual 加在子层输出上。

## 目标函数

### 13. 下一个 token

`logits` 形状为 `(B, S, V)`。第 t 个位置的 logits 预测 `labels[t+1]`，
所以 loss 计算为：

```python
cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=-100)
```

### 14. `-100`

`ignore_index=-100` 让这些位置不进入 loss。padding、system/user 等位置都应使用它屏蔽。

## 完成自检

1. `x` 在进入 attention 前是 `(B, S, H)`；投影并 reshape 后 q/k/v 是 `(B, S, heads, head_dim)`，
   转置后是 `(B, heads, S, head_dim)`；输出再转置并 reshape 回 `(B, S, H)`。
2. 上三角 mask 使位置 i 的 softmax 只能分配到 j <= i 的 key。
3. q/k norm 去掉会损失训练稳定性；RoPE 去掉会失去顺序信息；
   residual 去掉后深层网络容易出现梯度消失/训练不稳定。
