# Assignment 03：预训练训练循环

## 1. 本阶段目标

模型结构完成后，它只是随机初始化。预训练的目标是让模型学会“根据前文预测下一个 token”。
本阶段要完成的是：让数据、模型、优化器真正转起来。

## 2. 任务

### Task A：`get_lr`

文件：`trainer/trainer_utils.py`

实现一个随训练进度变化的学习率函数。

### Task B：`train_pretrain.py::train_epoch`

文件：`trainer/train_pretrain.py`

完成一个训练 step 的核心逻辑：forward、loss、backward、梯度累积、梯度裁剪、参数更新。

## 3. 引导问题

### 关于学习率

1. 训练后期学习率为什么要下降？
2. “cosine 衰减”的形状是什么？起点和终点分别应该是什么？
3. 学习率是否需要衰减到 0？如果不是，通常会保留多少？

### 关于一个训练 step

4. 一个 step 里，模型前向需要的输入是什么？返回的 loss 由哪几部分组成？
5. `autocast` 的作用是什么？loss scaling 解决什么问题？
6. 梯度累积的数学含义是什么？为什么 loss 要先除以累积步数再 backward？
7. 什么时候应该真正调用 `optimizer.step()`？什么时候应该清零梯度？
8. 梯度裁剪裁剪的是什么范数？它保护模型的哪一部分？

### 关于完整流程

9. 如果你在一个 batch 上观察 loss 不下降，应该先检查 forward、loss、还是学习率？
10. 为什么要用 `model.eval()` 保存 checkpoint，再切回 `model.train()`？

## 4. 参考资料

见 [论文阅读清单](reading-list.md) 的“阶段 3：训练”。

重点关注：

- AdamW 论文：优化器与权重衰减；
- SGDR 论文：cosine 学习率调度；
- GPT-3 论文：完整训练配方；
- Mixed Precision Training：autocast 与 scaler；
- Karpathy 的训练调试建议。

## 5. 验收

单元测试：

```bash
python3 -m pytest tests/test_trainer_utils.py -v
```

冒烟测试（会真实启动 `train_pretrain.py`，CPU 约几十秒）：

```bash
python3 -m pytest tests/test_pretrain_smoke.py -v
```

## 6. 完成自检

不看代码回答：

1. 如果把 `loss / accumulation_steps` 去掉，训练结果会有什么问题？
2. `optimizer.zero_grad` 放在 step 之前和之后有什么区别？
3. 什么情况下 loss 是 NaN？你设计的训练循环能定位到是哪一步出的问题吗？

回答完后再对照答案：

```bash
git show solutions:docs/answers/03-pretrain.md
```
