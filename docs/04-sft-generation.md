# Assignment 04：SFT 与生成

预训练让模型学会“像人一样续写”，但不会“听指令”。SFT 在有监督对话数据上继续训练，让它学会以 assistant 身份回答。

## 1. SFT 和预训练的差别

| | 预训练 | SFT |
|---|---|---|
| 数据 | 纯文本 | 多轮对话 |
| label | 所有真实 token 都学 | 只学 assistant 回答 |
| 主要目的 | 语言能力 / 知识 | 指令遵循 / 对话格式 |

## 2. 任务 A：`train_full_sft.py::train_epoch`

文件：`trainer/train_full_sft.py`

这段代码与 `train_pretrain.py::train_epoch` 几乎一样。请**不要复制粘贴**，而是先理解 Assignment 03 的实现，再在 SFT 脚本里独立写一遍。

SFT 默认：

- `learning_rate=1e-5`（比预训练低，避免破坏已有知识）
- `accumulation_steps=1`
- 数据来自 `SFTDataset`，labels 只有 assistant 回答不是 `-100`

冒烟验证：

```bash
python3 -m pytest tests/test_sft_smoke.py -v
```

## 3. 任务 B：`MiniMindForCausalLM.generate`

文件：`model/model_minimind.py`

训练结束后，模型一次只前向输出所有位置的 logits，但生成是**逐 token 自回归**的：

```text
输入已有 token
  → forward 得到最后一个位置的 logits
  → logits / temperature
  → 可选 repetition penalty
  → top-k 截断
  → top-p（nucleus）截断
  → softmax + multinomial 采样（do_sample=True）或 argmax
  → 拼接新 token，循环直到 EOS 或 max_new_tokens
```

`generate` 的输入输出接口、KV cache 更新、streamer 逻辑已经保留，需要补的是采样决策部分。建议按以下顺序阅读原函数结构：

1. `logits = outputs.logits[:, -1, :] / temperature`
2. repetition penalty 按“已出现 token”惩罚/奖励其分数
3. top-k：把 logits 中低于第 k 大值的分数设为 `-inf`
4. top-p：按概率累积排序，超过阈值 `top_p` 之后的 token 屏蔽
5. 采样或贪心得到 `next_token`

为什么要除 temperature？温度大于 1 让分布更均匀（更随机），小于 1 让分布更尖锐（更确定）。

## 4. 端到端验证

先跑完预训练和 SFT 冒烟测试，再在极小的 toy 权重上体验生成：

```bash
cd trainer && python3 train_pretrain.py ... # 得到 out/pretrain_16.pth
cd trainer && python3 train_full_sft.py ... # 得到 out/full_sft_16.pth
python3 eval_llm.py --weight full_sft --hidden_size 16 --num_hidden_layers 2
```

注意：16 维 toy 模型基本不会产生有意义文本；这个命令只是验证代码链路。想看到真正对话，用默认 768 维配置并下载完整数据。

## 5. 完成标准

```bash
python3 -m pytest tests/ -m "not slow" -v
python3 -m pytest tests/test_pretrain_smoke.py tests/test_sft_smoke.py -v
```

全部通过后，你已经从数据到训练到推理，亲手实现了一个 mini GPT。
