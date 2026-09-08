# Assignment 01：数据准备

大模型训练的第一个环节不是“搭网络”，而是把文本变成模型能吃的整数序列，并决定**哪些位置应该学习、哪些位置应该忽略**。

## 1. 概念：tokenization

MiniMind 使用 BPE tokenizer（`model/tokenizer.json`）。文本会先被切成 token，每个 token 对应一个整数 id：

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("model")
print(tokenizer("hello minimind")["input_ids"])
```

常用特殊 token：

- `bos_token_id=1`：句子开始 `<|im_start|>`
- `eos_token_id=2`：句子结束 `<|im_end|>`
- `pad_token_id=0`：把 batch 内短样本补到相同长度 `<|endoftext|>`
- label 里的 `-100`：`CrossEntropyLoss(ignore_index=-100)` 会跳过该位置，不计 loss

## 2. 任务 1A：`PretrainDataset.__getitem__`

文件：`dataset/lm_dataset.py`

预训练数据是纯文本，每一行类似：

```json
{"text": "深度学习很有意思"}
```

`PretrainDataset` 已经把样本读入 `self.samples`，你需要实现：

1. 用 `self.tokenizer` 把 `sample["text"]` tokenize；
2. 截断到 `max_length - 2`，预留 BOS / EOS；
3. 在开头加 `bos_token_id`、结尾加 `eos_token_id`；
4. 右侧 padding 到 `self.max_length`；
5. `labels` 初始等于 `input_ids`，但 **padding 位置改成 `-100`**；
6. 返回两个 `torch.long` 张量 `(input_ids, labels)`。

提示：

```python
tokenizer(text, add_special_tokens=False, max_length=..., truncation=True).input_ids
```

padding 不需要 attention mask，因为后续 loss 用 `-100` 屏蔽，padding 本身不会被模型学习。

验证：

```bash
python3 -m pytest tests/test_dataset.py::TestPretrainDataset -v
```

## 3. 概念：chat template 与 SFT loss mask

SFT 对话长这样：

```json
{
  "conversations": [
    {"role": "system", "content": "你是助手"},
    {"role": "user", "content": "1+1=?"},
    {"role": "assistant", "content": "2"}
  ]
}
```

`tokenizer.apply_chat_template(...)` 会把整段对话变成：

```text
<|im_start|>system
你是助手<|im_end|>
<|im_start|>user
1+1=?<|im_end|>
<|im_start|>assistant
2<|im_end|>
```

SFT 的原则是：**问题部分不要学（label=-100），只学 assistant 的回答**。
否则模型会把“用户问题”也当成自己该生成的话。

MiniMind 的模板把回答起点编码为：

```python
bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
```

把回答终点编码为：

```python
eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
```

## 4. 任务 1B：`SFTDataset.generate_labels`

文件：`dataset/lm_dataset.py`

`__getitem__` 已经负责加载对话、套用 chat template、截断和 padding，最后调用你实现的 `generate_labels(input_ids)`。

实现思路：

1. 初始化 `labels = [-100] * len(input_ids)`；
2. 用下标 `i` 扫描 `input_ids`；
3. 如果 `input_ids[i:i+len(bos_id)] == bos_id`，说明一个 assistant 回答开始；
4. 从回答起点向后找到下一个 `eos_id`，把这段开区间内的 label 设为原始 token id；
5. 跳过整个回答，继续扫描下一轮。

`SFTDataset` 的构造器里已经帮你算好了 `self.bos_id`、`self.eos_id`。
注意 `SFTDataset.__getitem__` 中的调试代码是注释，正式提交不需要打开。

验证：

```bash
python3 -m pytest tests/test_dataset.py::TestSFTDataset -v
```

## 5. 为什么 padding 也要放进 input_ids？

DataLoader 会把多个样本叠成 batch，所以同一 batch 的序列必须等长。短的样本在右侧补 `pad_token_id`。
模型仍然会对 padding 位置做 forward，但那些位置的预测目标被设为 `-100`，梯度为 0，因此不影响训练。

## 6. 完成标准

```bash
python3 -m pytest tests/test_dataset.py -v
```

全部通过后，你已经能回答：一个 token 序列的“正确答案”到底是什么。
