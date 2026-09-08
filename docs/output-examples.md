# 输出/结果示例

下面用 `tests/data/` 里的小型 fixture 和 toy 模型展示各阶段的“正常输出长什么样”。

> 注意：
> - tokenizer id 是稳定的，可以和你的实现逐位对比；
> - 模型随机初始化后的 loss / logits 数值会随 PyTorch 版本变化，主要看形状和量级；
> - 自动判分以 `tests/` 为准，这里只是帮你建立直觉。

## 1. PretrainDataset 输出

数据：`tests/data/pretrain_toy.jsonl`

第一行文本是 `hello minimind`，设置 `max_length=16`。

```python
input_ids = [1, 6170, 2995, 467, 916, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
labels    = [1, 6170, 2995, 467, 916, 2, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
```

可以验证：

```python
tokenizer.decode(input_ids)
# '<|im_start|>hello minimind<|im_end|><|endoftext|>...'
```

规律：

- `bos_token_id = 1` 在开头；
- 文本 token 跟在 BOS 后；
- `eos_token_id = 2` 在真实文本末尾；
- 右侧是 `pad_token_id = 0`；
- `labels` 与 `input_ids` 等长，padding 位置是 `-100`。

## 2. SFTDataset 输出

第一行对话内容是：

```text
system: You are a helpful assistant.
user: 1+1=?
assistant: 2
```

`SFTDataset` 的 label 可读成：

```text
<|im_start|>system...     IGNORE
<|im_start|>user\n1+1=?   IGNORE
<|im_start|>assistant\n   IGNORE
2                         LEARN
<|im_end|>\n              LEARN
<|endoftext|>...          IGNORE
```

也就是说：问题、system、角色标记和 padding 都不学习；
只有 assistant 的回答内容与结束符参与 loss。

## 3. 模型输出形状

使用 toy 配置（等价于测试里的配置）：

```text
hidden_size=16
num_hidden_layers=2
vocab_size=64
num_attention_heads=4
num_key_value_heads=2
head_dim=4
```

输入 `input_ids: (2, 17)`，输出：

```text
param count : 4,192
logits      : (2, 17, 64)
loss        : ≈ 4.17（随机初始化，数值会变）
```

单独看各组件：

```text
Attention input        : (B, S, hidden_size)
Attention output       : (B, S, hidden_size)
KV cache (keys/values) : (B, S, num_kv_heads, head_dim)
FeedForward input/out  : (B, S, hidden_size)
Decoder Block output   : (B, S, hidden_size)
```

如果输出形状和上面不一致，说明 reshape / transpose 或投影维度写错了。

## 4. get_lr 示例

`lr = 1e-3, total_steps = 100` 时，cosine 调度的关键节点：

```text
step   0  -> 1.000e-3
step  50  -> 5.500e-4
step 100  -> 1.000e-4
```

## 5. 训练日志示例

### 预训练冒烟

用 3 条 toy 文本、`hidden_size=16`、`num_hidden_layers=2`、
`accumulation_steps=2`、`max_seq_len=24` 跑出来的日志：

```text
Model Params: 0.11M
Trainable Params: 0.110M
Epoch:[1/1](1/2), loss: 8.7479, logits_loss: 8.7479, aux_loss: 0.0000, lr: 0.00055000, epoch_time: 0.0min
Epoch:[1/1](2/2), loss: 8.7547, logits_loss: 8.7547, aux_loss: 0.0000, lr: 0.00010000, epoch_time: 0.0min
```

### SFT 冒烟

```text
Model Params: 0.11M
Trainable Params: 0.110M
Epoch:[1/1](1/1), loss: 8.7500, logits_loss: 8.7500, aux_loss: 0.0000, lr: 0.00010000, epoch_time: 0.0min
```

日志字段含义：

```text
loss        : 当前 micro-batch 的平均 loss（已乘回 accumulation_steps）
logits_loss : 语言建模 loss
aux_loss    : MoE 辅助 loss（dense 模型恒为 0）
lr          : 当前 step 实际使用的学习率
```

## 6. 如何复现

在仓库根目录打开 Python，直接构造小数据集即可查看：

```python
from transformers import AutoTokenizer
from dataset.lm_dataset import PretrainDataset, SFTDataset

tokenizer = AutoTokenizer.from_pretrained("model")
ds = PretrainDataset("tests/data/pretrain_toy.jsonl", tokenizer, max_length=16)
print(ds[0])
```

> 当前 master 在 TODO 完成前会抛出 `NotImplementedError`，
> 这是预期的；先用 `solutions` 分支看输出，再回来对照自己的实现。
