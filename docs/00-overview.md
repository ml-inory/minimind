# 第 0 讲：仓库地图与使用方法

## 1. 你要最终做出什么

当全部 TODO 完成后，你应该能独立跑通这条主链路：

```text
原始文本
  → PretrainDataset / SFTDataset（tokenize、padding、loss mask）
  → MiniMindForCausalLM（embedding → Transformer Block × N → lm_head）
  → train_pretrain.py（预训练：下一个 token 预测）
  → train_full_sft.py（SFT：只学习 assistant 回答）
  → generate()（温度采样 / top-k / top-p 解码）
```

每个环节都已经被“挖空”，文档会先讲原理，再给你任务和自测方式。

## 2. 仓库结构

```text
dataset/lm_dataset.py        # 数据：Pretrain / SFT / DPO 等 Dataset
model/model_minimind.py      # 模型：Config、RMSNorm、RoPE、Attention、FFN、CausalLM
model/model_lora.py          # 可选：LoRA 实现
trainer/trainer_utils.py     # 学习率、随机种子、分布式、checkpoint 等工具
trainer/train_pretrain.py    # 预训练入口
trainer/train_full_sft.py    # 全参数 SFT 入口
trainer/train_lora.py        # 可选：LoRA 训练入口
eval_llm.py                  # 用生成接口测试模型
docs/                        # 本作业的学习文档
tests/                       # 自动测试（你的“判卷器”）
```

## 3. 主干代码中的“给定部分”和“TODO 部分”

作业不会让你从空文件开始。保留的内容有两类：

1. **纯工程代码**：命令行参数、目录创建、DDP/compile 包装、checkpoint 保存等。
2. **数据流骨架**：类的定义、函数签名、以及前后已经写好的调用。

需要你补的是**算法核心**。代码中的 TODO 是唯一的“题面”，例如：

```python
def norm(self, x):
    # TODO(Assignment 02 · Task A): 实现 RMSNorm 的归一化公式
    raise NotImplementedError("Assignment 02 · Task A")
```

完成一个 TODO 后，删掉对应的 `raise NotImplementedError` 即可。

## 4. Ground truth 怎么用

仓库在改造前已经建立了两个引用：

```bash
git branch solutions               # 完整实现分支
git tag reference/original-source  # 改造前原始提交
```

建议的自我约束：

1. 每个任务至少自己尝试 20 分钟。
2. 先用 `pytest` 看清失败点，再回到公式文档里推理。
3. 实在卡住时，只查看“当前卡住的函数”，例如：

```bash
git show solutions:model/model_minimind.py | sed -n '70,140p'
```

4. 看懂后**关掉答案，凭理解重写**，而不是粘贴。

## 5. 判卷方式

仓库里的测试就是作业的验收条件：

```bash
python3 -m pytest tests/ -m "not slow" -v
```

每个测试文件对应一个阶段：

| 测试文件 | 覆盖阶段 |
|----------|----------|
| `test_dataset.py` | Assignment 01 |
| `test_model_components.py` | Assignment 02 |
| `test_trainer_utils.py` | Assignment 03 的 get_lr |
| `test_pretrain_smoke.py` | Assignment 03 的完整预训练循环（slow） |
| `test_sft_smoke.py` | Assignment 04 的完整 SFT 循环（slow） |

做完一个阶段，跑对应文件，让红色变成绿色。
