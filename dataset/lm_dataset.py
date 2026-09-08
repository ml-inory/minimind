from torch.utils.data import Dataset
import torch
import json
import os
import random
from typing import Any, Dict, List, Tuple
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    """读入 jsonl 预训练文本，按 index 返回一个 (input_ids, labels) 样本。"""

    def __init__(self, data_path: str, tokenizer: Any, max_length: int = 512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # self.samples: HF datasets.Dataset（Arrow 表，支持 len/索引/迭代）
        # self.samples[index] -> Dict[str, Any]
        # pretrain 每行形如 {"text": "..."}，取文本用 sample["text"]
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # sample: Dict[str, Any]，例如 {"text": str}
        sample = self.samples[index]
        # 可用工具：
        #   self.tokenizer(text, add_special_tokens=False, truncation=True, max_length=...) -> {"input_ids": List[int]}
        #   self.tokenizer.bos_token_id / eos_token_id / pad_token_id -> int
        # 输出约定：
        #   input_ids/labels 都是形状 (self.max_length,) 的 torch.long；
        #   labels 中不应学习的 padding 位置设为 -100，其余与 input_ids 相同。
        # TODO(Assignment 01 · Task A): 实现预训练样本构造
        # 返回 (input_ids, labels)；先想清楚模型在每个位置“该学什么”
        raise NotImplementedError("Assignment 01 · Task A")


class SFTDataset(Dataset):
    """读入多轮对话 jsonl，把每轮套上 chat template，只保留 assistant 回答作为 label。"""

    def __init__(self, jsonl_path: str, tokenizer: Any, max_length: int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        # self.samples[index] -> {"conversations": List[Dict[str, Any]]}
        # 其中每个 message 至少含 "role" 与 "content"（还有可选 reasoning_content/tools/tool_calls）
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        # TODO(Assignment 01 · Task B-0): 预计算回答边界的 token 序列
        # 后续 generate_labels 会按这两个序列寻找“assistant 回答起点/终点”。
        # 类型: List[int]（长度不限，通常只有几个 token）
        # 可先用 self.tokenizer.apply_chat_template(...) 打印真实字符串，再决定边界。
        self.bos_id: Optional[List[int]] = None
        self.eos_id: Optional[List[int]] = None

    def __len__(self) -> int:
        return len(self.samples)

    def create_chat_prompt(self, conversations: List[Dict[str, Any]]) -> str:
        """把角色消息列表渲染成一段可 tokenize 的 chat 文本。"""
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids: List[int]) -> List[int]:
        # input_ids: 长度 self.max_length 的 token id 列表（已含 padding）
        # 返回: 等长 label 列表；需要模型学习的位置保留原 token id，其余为 -100。
        # TODO(Assignment 01 · Task B-1): 生成 SFT loss mask
        # 返回等长 label；哪些 token 值得让模型学习、哪些应该忽略？
        raise NotImplementedError("Assignment 01 · Task B-1")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample: Dict[str, Any] = self.samples[index]  # sample["conversations"]: List[Dict]
        conversations: List[Dict[str, Any]] = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    def __init__(self, file_path: str, tokenizer: Any, max_length: int = 4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # TODO(Assignment 01 · Task C-0，可选): 预计算 DPO 回答边界
        # 类型与 SFT 的 bos_id/eos_id 相同: List[int]
        self.bos_id: Optional[List[int]] = None
        self.eos_id: Optional[List[int]] = None
        # self.samples[index] -> Dict[str, Any]，包含 "chosen"/"rejected"
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids: List[int]) -> List[int]:
        # input_ids: 长度 self.max_length 的 token id 列表
        # 返回: 等长 0/1 mask；1 = 参与 DPO loss，0 = 忽略
        # TODO(Assignment 01 · Task C，可选): 生成 DPO 的 0/1 loss mask
        # DPO 的 mask 使用 0/1；请自行决定哪些 token 记为 1
        raise NotImplementedError("Assignment 01 · Task C-1")


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 按概率开启 thinking
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )
    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': ""
        }

class AgentRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


if __name__ == "__main__":
    pass
