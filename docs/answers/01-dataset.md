# Assignment 01 参考答案

## 引导问题

### 1. 模型预测“当前 token”还是“下一个 token”？

预测下一个 token。语言建模的目标是在给定前文 `x_1 ... x_t` 时预测 `x_{t+1}`。
因此 logits 与 labels 在时间维上错开一位：

```text
logits[t] 预测 labels[t+1]
```

### 2. 如何观察 tokenizer？

可以直接打印特殊 token 的 id：

```python
tokenizer.bos_token_id  # 1
tokenizer.eos_token_id  # 2
tokenizer.pad_token_id  # 0
```

MiniMind 的 BOS 是 `<|im_start|>`，EOS 是 `<|im_end|>`，PAD 是 `<|endoftext|>`。

### 3. padding 参与 loss 会怎样？

模型会学到“看到 padding 就预测 padding”这种无意义规律；padding token 数量还可能远多于真实文本，
从而稀释正常文本的梯度。因此 padding 位置的 label 应设为 `-100`。

### 4. SFT 对话中，模型该学哪些文本？

`apply_chat_template` 会把 system/user/assistant 都渲染成一段文本。
如果全部参与 loss，模型会学会把用户的提问也“复述”出来。
正确做法是只把 assistant 的**回答内容**以及表示回答结束的 EOS 作为目标，
system/user 与 `assistant\n` 角色标记都屏蔽。

### 5. 为什么不能把 assistant content 单独 tokenize？

因为训练样本必须保持完整对话上下文：模型是在前文（system/user/前几轮）条件下预测回答的。
单独对回答 tokenize 会丢失条件关系，也无法让模型学会“何时开始回答、何时结束”。

## 实现要点

### Pretrain

```text
tokens = tokenizer(text, add_special_tokens=False, ...).input_ids
input_ids = [bos] + tokens[:max_length - 2] + [eos] + [pad] * ...
labels = input_ids.copy()
labels[input_ids == pad] = -100
```

### SFT 边界

回答起点可用：

```python
tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
```

回答终点可用：

```python
tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
```

从起点标记之后到终点标记结束的区间保留 label，其余位置设为 `-100`。

## 完成自检

- input 和 label 等长，因为每个输入位置都对应一个“下一个 token”目标；
- `-100` 是 PyTorch `CrossEntropyLoss` 的 `ignore_index`，该位置不产生梯度；
- SFT 只覆盖回答内容与 EOS，不覆盖用户问题、system、角色标记。
