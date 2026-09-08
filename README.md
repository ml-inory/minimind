# MiniMind Homework：从零写一个大语言模型

这是一个基于 [MiniMind](https://github.com/jingyaogong/minimind) 改造的 LLM 作业仓库。
目标不是“跑通一个框架”，而是像 CS229 的编程作业一样：**先理解每一行代码，再亲手把空白的 TODO 填回来**，最终独立实现一个小型 GPT/Decoder-Only 模型及其预训练、SFT 流程。

> ⚠️ 本分支是作业骨架：核心实现已经被替换成 `TODO`。
> 完整的原始实现保存在 **`solutions` 分支** 中，作为你的“参考答案/ground truth”，平时不要看。

## 学习路线

| 阶段 | 内容 | 文档 | 主要文件 |
|------|------|------|----------|
| 0 | 代码地图、环境、如何使用本仓库 | [docs/00-overview.md](docs/00-overview.md) | — |
| 1 | 数据准备：Pretrain / SFT 数据集 | [docs/01-dataset.md](docs/01-dataset.md) | `dataset/lm_dataset.py` |
| 2 | 网络结构：RMSNorm、RoPE、Attention、FFN、CausalLM | [docs/02-model.md](docs/02-model.md) | `model/model_minimind.py` |
| 3 | 预训练：训练循环、梯度累积、学习率、混合精度 | [docs/03-pretrain.md](docs/03-pretrain.md) | `trainer/train_pretrain.py`、`trainer/trainer_utils.py` |
| 4 | SFT 与生成：指令微调、loss mask、采样 | [docs/04-sft-generation.md](docs/04-sft-generation.md) | `trainer/train_full_sft.py`、`model/model_minimind.py` |
| 5（可选） | LoRA / MoE / KV Cache / YaRN / RL 等进阶方向 | [docs/05-advanced.md](docs/05-advanced.md) | 各进阶文件 |

## TODO 分布

代码中的 `TODO` 都以 `Assignment NN ·` 开头，例如：

```python
# TODO(Assignment 02 · Task B): 计算 RoPE 基础频率
raise NotImplementedError("Assignment 02 · Task B")
```

建议按文档顺序完成：先做数据集，再做模型，最后做训练循环。只有前一步通过，后一步才能运行。

## 如何拿参考答案

原始代码保留在两个地方：

```bash
git branch solutions            # 完整实现分支
git tag reference/original-source  # 改造前的原始提交
```

如果想查看某个文件的原始实现，先完成作业后对比，不要一开始就抄：

```bash
# 完整切到答案分支（会覆盖当前工作区，注意先提交自己的代码）
git switch solutions

# 或者只看某个文件的原始版本：
git show solutions:model/model_minimind.py | less
```

如果你计划把本仓库公开为作业仓库，建议 **只推送 master，不推送 solutions 分支**。

## 验证命令

快速验证（除训练冒烟测试外）：

```bash
python3 -m pytest tests/ -m "not slow" -v
```

完整验证（包含一个极小型预训练 / SFT 冒烟测试，CPU 上约几十秒）：

```bash
python3 -m pytest tests/ -v
```

也可以单独跑某阶段的测试：

```bash
python3 -m pytest tests/test_dataset.py -v
python3 -m pytest tests/test_model_components.py -v
python3 -m pytest tests/test_trainer_utils.py -v
python3 -m pytest tests/test_pretrain_smoke.py -v
```

## 环境

本项目沿用 MiniMind 的依赖：

```bash
pip install -r requirements-dev.txt
```

CPU 上也能完成全部作业和冒烟测试；真正的预训练 / SFT 建议使用至少 8GB 显存的 GPU。
训练数据下载方式见 [README_original.md](README_original.md) 的“下载数据”部分。

## 文件导航

原版 MiniMind README（数据下载、完整训练说明、RL 等）已保留在：

- [README_original.md](README_original.md)（中文）
- [README_original_en.md](README_original_en.md)（英文）
