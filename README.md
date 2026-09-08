# MiniMind Homework：从零写一个大语言模型

这是一个基于 [MiniMind](https://github.com/jingyaogong/minimind) 改造的 LLM 作业仓库。
目标不是“跑通一个框架”，而是像 CS229 的编程作业一样：**先理解每一行代码，再亲手把空白的 TODO 填回来**，最终独立实现一个小型 GPT/Decoder-Only 模型及其预训练、SFT 流程。

> ⚠️ 当前 `master` 是作业骨架：核心实现已经被替换成 `TODO`。
> 完整的原始实现保存在 **`solutions` 分支**，作为你的“参考答案/ground truth”，建议独立完成后再查看。

---

## 一、作业说明

### 1.1 你最终要完成什么

全部 TODO 完成后，你应该能独立跑通这条主链路：

```text
原始文本
  → PretrainDataset / SFTDataset（tokenize、padding、loss mask）
  → MiniMindForCausalLM（embedding → Transformer Block × N → lm_head）
  → train_pretrain.py（预训练：下一个 token 预测）
  → train_full_sft.py（SFT：只学习 assistant 回答）
  → generate()（temperature / top-k / top-p 采样）
```

### 1.2 学习路线与 TODO 分布

| 阶段 | 内容 | 文档 | 主要文件 | 对应测试 |
|------|------|------|----------|----------|
| 0 | 代码地图、仓库使用方法 | [docs/00-overview.md](docs/00-overview.md) | — | — |
| 1 | 数据准备：Pretrain / SFT 数据集 | [docs/01-dataset.md](docs/01-dataset.md) | `dataset/lm_dataset.py` | `tests/test_dataset.py` |
| 2 | 网络结构：RMSNorm、RoPE、Attention、FFN、CausalLM | [docs/02-model.md](docs/02-model.md) | `model/model_minimind.py` | `tests/test_model_components.py` |
| 3 | 预训练：训练循环、梯度累积、学习率、混合精度 | [docs/03-pretrain.md](docs/03-pretrain.md) | `trainer/train_pretrain.py`、`trainer/trainer_utils.py` | `tests/test_trainer_utils.py`、`tests/test_pretrain_smoke.py` |
| 4 | SFT 与生成：指令微调、loss mask、采样 | [docs/04-sft-generation.md](docs/04-sft-generation.md) | `trainer/train_full_sft.py`、`model/model_minimind.py` | `tests/test_sft_smoke.py` |
| 5（可选） | LoRA / MoE / KV Cache / YaRN / RL | [docs/05-advanced.md](docs/05-advanced.md) | 各进阶文件 | — |

代码中的 `TODO` 都以 `Assignment NN ·` 开头，例如：

```python
# TODO(Assignment 02 · Task B): 计算 RoPE 基础频率
raise NotImplementedError("Assignment 02 · Task B")
```

建议按阶段顺序完成：先做数据，再做模型，最后做训练循环。只有前一步通过，后一步才能运行。

### 1.3 建议作业流程

```bash
# 1. 先读对应阶段的文档，理解原理
cat docs/01-dataset.md

# 2. 查看该阶段所有 TODO
rg "TODO" dataset/lm_dataset.py

# 3. 独立实现，删掉对应的 raise NotImplementedError
# 4. 运行该阶段的测试，直到全绿
python3 -m pytest tests/test_dataset.py -v
```

### 1.4 参考答案怎么用

原始代码保留在两个地方：

```bash
git branch solutions                # 完整实现分支
git tag reference/original-source   # 改造前的原始提交
```

查看某个文件的原始实现（建议完成后再对比，不要一开始就抄）：

```bash
# 只看某个文件的原始版本
git show solutions:model/model_minimind.py | less

# 或者完整切到答案分支（会覆盖工作区，先提交自己的代码）
git switch solutions
```

如果你计划把本仓库公开为作业仓库，建议 **只推送 master，不推送 solutions 分支**。

### 1.5 完成标准

快速测试全部通过：

```bash
python3 -m pytest tests/ -m "not slow" -v
```

两个真实脚本冒烟测试也通过：

```bash
python3 -m pytest tests/test_pretrain_smoke.py tests/test_sft_smoke.py -v
```

> 未实现前，测试会以 `NotImplementedError: Assignment ...` 失败，这是正常状态。

---

## 二、环境安装

### 2.1 版本建议

- Python：`3.10+`（上游 MiniMind 使用 Python 3.10）
- 系统：Linux / macOS 均可；CPU 能完成全部作业与冒烟测试
- GPU：真正跑 MiniMind 预训练 / SFT 建议 8GB+ 显存（单张 3090 即可）
- PyTorch：建议先按下方命令单独安装，再安装项目依赖

### 2.2 创建虚拟环境

推荐 `conda`：

```bash
conda create -n minimind-homework python=3.10 -y
conda activate minimind-homework
```

也可以使用系统自带的 `venv`：

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 2.3 安装 PyTorch

CPU 环境：

```bash
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

CUDA 环境（按你的驱动版本选择 `cu121` / `cu124` / `cu126` 等）：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 2.4 安装项目依赖

`requirements-dev.txt` 包含项目依赖和测试工具 `pytest`：

```bash
python -m pip install -r requirements-dev.txt
```

国内网络可用阿里云镜像加速：

```bash
python -m pip install -r requirements-dev.txt -i https://mirrors.aliyun.com/pypi/simple
```

### 2.5 验证环境

```bash
python - <<'PY'
import torch
import transformers
import datasets
import pytest
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("pytest:", pytest.__version__)
PY
```

> 自动测试使用仓库内 `tests/data/` 的小型 JSONL fixture，不需要提前下载 MiniMind 训练数据；
> 要真正训练完整模型时，再按 [README_original.md](README_original.md) 的“下载数据”部分准备数据。

---

## 三、如何测试

### 3.1 快速测试（推荐每次提交前跑）

```bash
python3 -m pytest tests/ -m "not slow" -v
```

覆盖：数据集的 tokenize / loss mask、模型组件数值、CausalLM forward、学习率函数。

### 3.2 分阶段测试

| 阶段 | 测试命令 |
|------|----------|
| 1 数据准备 | `python3 -m pytest tests/test_dataset.py -v` |
| 2 网络结构 | `python3 -m pytest tests/test_model_components.py -v` |
| 3 学习率 | `python3 -m pytest tests/test_trainer_utils.py -v` |
| 3 预训练冒烟 | `python3 -m pytest tests/test_pretrain_smoke.py -v` |
| 4 SFT 冒烟 | `python3 -m pytest tests/test_sft_smoke.py -v` |

### 3.3 完整验证

包含两个“真的把训练脚本跑起来”的冒烟测试，CPU 上约几十秒：

```bash
python3 -m pytest tests/ -v
```

### 3.4 完成后的自查

把自己实现的版本和参考答案做 diff，确认没有偏离原始语义：

```bash
git diff solutions -- model/model_minimind.py
```

---

## 四、相关文档

- [docs/00-overview.md](docs/00-overview.md)：仓库地图与学习方法
- [README_original.md](README_original.md)：原版 MiniMind 中文 README（数据下载、完整训练、RL 等）
- [README_original_en.md](README_original_en.md)：原版 MiniMind 英文 README
