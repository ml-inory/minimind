# 第 0 讲：仓库地图与学习方法

## 1. 这门作业的规则

这是一门“引导式”的作业，不是代码填空题。
每篇文档只说明三件事：

1. 这个阶段要**理解什么**；
2. 要完成哪些函数；
3. 用什么测试验收。

文档不会直接给出实现公式或关键推导。当你不知道怎么做时，应该先回到问题本身，
查阅 [论文阅读清单](reading-list.md)，再对着 `solutions` 分支中的原始实现验证你的结论。

建议的执行顺序：

```text
读问题 → 查资料/读论文 → 写出自己的推导 → 实现 → 跑测试 → 与参考答案对比
```

## 2. 仓库主链路

```text
dataset/lm_dataset.py
  → model/model_minimind.py
  → trainer/train_pretrain.py
  → trainer/train_full_sft.py
  → generate()
```

每篇作业文档都会标注它处于主链路的哪一环。

## 3. 作业阶段

| 阶段 | 文档 | 主要文件 | 测试 |
|------|------|----------|------|
| 1 | [01-dataset.md](01-dataset.md) | `dataset/lm_dataset.py` | `tests/test_dataset.py` |
| 2 | [02-model.md](02-model.md) | `model/model_minimind.py` | `tests/test_model_components.py` |
| 3 | [03-pretrain.md](03-pretrain.md) | `trainer/train_pretrain.py`、`trainer/trainer_utils.py` | `tests/test_trainer_utils.py`、`tests/test_pretrain_smoke.py` |
| 4 | [04-sft-generation.md](04-sft-generation.md) | `trainer/train_full_sft.py`、`model/model_minimind.py` | `tests/test_sft_smoke.py` |
| 5（可选） | [05-advanced.md](05-advanced.md) | 各进阶文件 | — |

## 4. 如何定位 TODO

代码中的 TODO 使用统一编号：

```bash
rg "TODO" dataset/lm_dataset.py model/model_minimind.py trainer/
```

例如 `TODO(Assignment 02 · Task B)` 表示这是第 2 阶段、任务 B。
完成一个任务后，删除对应的 `raise NotImplementedError`。

## 5. 参考答案的使用边界

`solutions` 分支是完整实现，也是这门作业的“评分标准”。请仅在以下场景使用：

1. 已经完成某个任务并让对应测试通过；
2. 测试失败但你已经尝试推导，确实无法继续；
3. 完成后做 code review，检查自己的实现与原始实现语义是否一致。

查看单个文件：

```bash
git show solutions:model/model_minimind.py | less
```

## 6. 验证命令

快速测试（不包含完整训练冒烟）：

```bash
python3 -m pytest tests/ -m "not slow" -v
```

完整验证：

```bash
python3 -m pytest tests/ -v
```

每个阶段的详细验证方式见对应文档。

## 7. 如果卡住了

先问自己三个问题，而不是立刻看答案：

1. 这个函数的输入/输出分别是什么形状和含义？
2. 背后的数学或算法想解决什么问题？
3. 如果让我给一个 5 岁的孩子讲，我会怎么说？

如果仍然没有头绪，去 [论文阅读清单](reading-list.md) 找对应资料，通常答案就在论文的某个章节里。
