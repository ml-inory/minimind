# Assignment 04：SFT 与生成

## 1. 本阶段目标

预训练让模型学会“续写”，SFT 让模型学会“按指令回答”。本阶段有两个任务：

1. 完成 SFT 训练循环；
2. 完成自回归生成中的采样逻辑。

## 2. 任务

### Task A：`train_full_sft.py::train_epoch`

文件：`trainer/train_full_sft.py`

SFT 与预训练的训练循环高度相似，但请独立推导一遍，不要直接复制 Assignment 03 的代码。
差异应该体现在数据与 label 上，而不是优化器机制上。

### Task B：`MiniMindForCausalLM.generate`

文件：`model/model_minimind.py`

函数已经负责管理自回归循环、KV cache 和 EOS 停止条件；需要完成的是“如何从最后一组 logits 选出下一个 token”。

## 3. 引导问题

### 关于 SFT

1. SFT 和预训练在 label 上的区别是什么？为什么？
2. SFT 的 loss 应该覆盖 system/user 文本吗？
3. SFT 的学习率为什么通常比预训练小？

### 关于生成

4. 每次生成“一个 token”和一次前向算出整段 logits 有什么关系？
5. 温度参数如何改变概率分布的形状？温度趋近 0 时会发生什么？
6. top-k 截断是在概率上做还是在 logits 上做？为什么？
7. top-p 与 top-k 想解决什么问题？
8. repetition penalty 为什么要区分正分和负分？
9. 训练时模型见过“重复文本”，生成时为什么还会退化？这个问题催生了哪些采样方法？

## 4. 参考资料

见 [论文阅读清单](reading-list.md) 的“阶段 4：生成解码”。

SFT 的背景可回看“阶段 1”中的 InstructGPT / LIMA。

## 5. 验收

SFT 训练循环：

```bash
python3 -m pytest tests/test_sft_smoke.py -v
```

生成部分目前没有独立单元测试，验收方式是：完成全部阶段后运行 `eval_llm.py`，
模型应能正常生成直到 EOS，不会因为采样代码崩溃。

## 6. 完成自检

不看代码回答：

1. 如果 `temperature=0`，你的实现会发生什么？应不应该支持？
2. top-k 和 top-p 同时使用时的执行顺序重要吗？
3. 生成时为什么只在最后一个位置取 logits，而不是整段都取？
