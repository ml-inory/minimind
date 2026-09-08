# Assignment 01：数据准备

## 1. 本阶段目标

理解下面三个问题，并在代码中把它们实现出来：

1. 一段文本经过 tokenizer 后，如何变成模型的输入和“标准答案”？
2. 批量训练要求等长序列，padding 应该出现在哪里？它会不会影响梯度？
3. SFT 中哪些 token 应该参与 loss？为什么不是整段对话都参与？

## 2. 任务

### Task A：`PretrainDataset.__getitem__`

文件：`dataset/lm_dataset.py`

输入是一个预训练样本，输出应该是一个可直接用于语言建模的 `(input_ids, labels)` 对。

### Task B：`SFTDataset.generate_labels`

文件：`dataset/lm_dataset.py`

输入是已经套好 chat template 的完整对话 token 序列，输出是等长的 label 序列。
它决定模型“应该模仿哪些 token、忽略哪些 token”。

该任务还包含 `SFTDataset.__init__` 中的 TODO（Task B-0）：
先决定用什么 token 序列标记一段回答的“起点”和“终点”，
再在 `generate_labels`（Task B-1）里使用它们。

## 3. 引导问题

在写代码之前，先回答下面这些问题：

1. 语言模型在每个位置预测的是“当前 token”还是“下一个 token”？这决定了 input 和 label 是否错位。
2. 你在 `__getitem__` 里已经能看到 `sample`，可以先用 tokenizer 打印它，观察 BOS/EOS/PAD 的真实 id。
3. 如果 padding 位置也参与 loss，模型会学到什么奇怪的规律？
4. 用 `tokenizer.apply_chat_template(...)` 打印一段 SFT 对话，观察：
   - 一段回答从哪里开始；
   - 一段回答到哪里结束；
   - 哪些文本是模型应该“背下来”的，哪些只是背景。
5. 为什么不能简单地把“assistant 的 content”单独 tokenize？你从第 4 题的字符串中能发现什么边界标记？

## 4. 参考资料

见 [论文阅读清单](reading-list.md) 的“阶段 1：数据与语言建模目标”。

重点关注：

- GPT-2 论文中关于语言建模目标的描述；
- InstructGPT / LIMA 中关于“让模型学习什么”的讨论；
- BPE 论文中 subword 划分的基本思想。

## 5. 验收

```bash
python3 -m pytest tests/test_dataset.py -v
```

测试会检查：BOS/EOS/Padding 是否正确、padding 是否被 mask、
SFT 的 label 是否只覆盖 assistant 回答。

## 6. 完成自检

不看代码，向自己解释：

- 为什么 `PretrainDataset` 返回的 `labels` 和 `input_ids` 长度相同；
- `-100` 在交叉熵中的语义；
- SFT 的 label 应该覆盖哪些 token、不覆盖哪些 token？你的判断依据是什么？
