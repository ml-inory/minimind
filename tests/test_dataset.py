import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from dataset.lm_dataset import PretrainDataset, SFTDataset

ROOT = Path(__file__).resolve().parents[1]


def find_sequence(haystack, needle):
    """Return first index where needle occurs in haystack."""
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(str(ROOT / "model"))


@pytest.fixture(scope="module")
def pretrain_ds(tokenizer):
    return PretrainDataset(str(ROOT / "tests/data/pretrain_toy.jsonl"), tokenizer, max_length=24)


@pytest.fixture(scope="module")
def sft_ds(tokenizer):
    return SFTDataset(str(ROOT / "tests/data/sft_toy.jsonl"), tokenizer, max_length=80)


class TestPretrainDataset:
    def test_bos_eos_padding_and_labels(self, tokenizer, pretrain_ds):
        raw_text = "hello minimind"
        expected_tokens = tokenizer(raw_text, add_special_tokens=False).input_ids
        input_ids, labels = pretrain_ds[0]

        assert isinstance(input_ids, torch.Tensor) and input_ids.dtype == torch.long
        assert input_ids.shape == torch.Size([24])
        assert input_ids[0].item() == tokenizer.bos_token_id
        assert input_ids[1 : 1 + len(expected_tokens)].tolist() == expected_tokens
        assert input_ids[1 + len(expected_tokens)].item() == tokenizer.eos_token_id
        assert input_ids[-1].item() == tokenizer.pad_token_id

        pad_mask = input_ids == tokenizer.pad_token_id
        assert (labels[pad_mask] == -100).all()
        assert (labels[~pad_mask] == input_ids[~pad_mask]).all()

    def test_len_matches_file(self, pretrain_ds):
        with open(ROOT / "tests/data/pretrain_toy.jsonl") as f:
            n_lines = sum(1 for _ in f)
        assert len(pretrain_ds) == n_lines


class TestSFTDataset:
    def test_labels_only_assistant(self, tokenizer, sft_ds):
        with open(ROOT / "tests/data/sft_toy.jsonl") as f:
            sample = json.loads(f.readline())

        input_ids, labels = sft_ds[0]
        input_ids = input_ids.tolist()
        labels = labels.tolist()
        assert len(input_ids) == len(labels) == 80

        answer_start = find_sequence(input_ids, tokenizer("2", add_special_tokens=False).input_ids)
        assert answer_start > 0

        # assistant answer tokens participate in the loss
        answer = tokenizer("2", add_special_tokens=False).input_ids
        for offset in range(len(answer)):
            assert labels[answer_start + offset] == input_ids[answer_start + offset]

        # prompt-side user text must be masked
        user_ids = tokenizer("1+1=?", add_special_tokens=False).input_ids
        user_pos = find_sequence(input_ids, user_ids)
        assert user_pos > 0
        for offset in range(len(user_ids)):
            assert labels[user_pos + offset] == -100

        # padding is masked as well
        pad_positions = [i for i, v in enumerate(input_ids) if v == tokenizer.pad_token_id]
        assert pad_positions
        assert all(labels[i] == -100 for i in pad_positions)
