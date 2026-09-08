import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers.activations import ACT2FN

from model.model_minimind import (
    Attention,
    FeedForward,
    MiniMindConfig,
    MiniMindForCausalLM,
    RMSNorm,
    apply_rotary_pos_emb,
    precompute_freqs_cis,
    repeat_kv,
)

ROOT = Path(__file__).resolve().parents[1]


def make_config(**overrides):
    defaults = dict(
        hidden_size=16,
        num_hidden_layers=2,
        vocab_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=16,
        max_position_embeddings=64,
        dropout=0.0,
        flash_attn=False,
    )
    defaults.update(overrides)
    return MiniMindConfig(**defaults)


class TestRMSNorm:
    def test_norm_value(self):
        eps = 1e-5
        module = RMSNorm(dim=8, eps=eps)
        torch.manual_seed(0)
        x = torch.randn(3, 5, 8)

        expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        assert torch.allclose(module.norm(x), expected, atol=1e-6)

    def test_forward_scales_by_weight(self):
        module = RMSNorm(dim=8, eps=1e-5)
        torch.manual_seed(1)
        module.weight.data.normal_()
        x = torch.randn(2, 4, 8)
        assert torch.allclose(module(x), module.weight * module.norm(x), atol=1e-6)


class TestFeedForward:
    def test_swiglu_value(self):
        config = make_config()
        module = FeedForward(config)
        torch.manual_seed(0)
        x = torch.randn(2, 5, 16)

        expected = module.down_proj(
            ACT2FN[config.hidden_act](module.gate_proj(x)) * module.up_proj(x)
        )
        assert torch.allclose(module(x), expected, atol=1e-6)


class TestRotaryEmbeddings:
    def test_precompute_freqs_cis_shape_and_value(self):
        dim, end, rope_base = 8, 10, 10_000.0
        cos, sin = precompute_freqs_cis(dim=dim, end=end, rope_base=rope_base)
        assert cos.shape == (end, dim) and sin.shape == (end, dim)

        half = dim // 2
        index = torch.arange(0, dim, 2)[:half].float()
        freqs = 1.0 / (rope_base ** (index / dim))
        angles = torch.outer(torch.arange(end, dtype=torch.float32), freqs)
        expected_cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)
        expected_sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)
        assert torch.allclose(cos, expected_cos, atol=1e-6)
        assert torch.allclose(sin, expected_sin, atol=1e-6)

    def test_apply_rotary_pos_emb_matches_manual_rotate_half(self):
        dim, end = 4, 3
        cos, sin = precompute_freqs_cis(dim=dim, end=end, rope_base=10_000.0)
        torch.manual_seed(0)
        q = torch.randn(1, end, 1, dim)
        k = torch.randn(1, end, 1, dim)

        def rotate_half(x):
            half = x.shape[-1] // 2
            return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)
        assert torch.allclose(q_embed, q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1), atol=1e-6)
        assert torch.allclose(k_embed, k * cos.unsqueeze(1) + rotate_half(k) * sin.unsqueeze(1), atol=1e-6)


class TestAttention:
    @pytest.fixture()
    def attention(self):
        config = make_config(hidden_size=8, num_attention_heads=2, num_key_value_heads=1, head_dim=4)
        return Attention(config)

    def test_matches_scaled_dot_product_attention(self, attention):
        config = make_config(hidden_size=8, num_attention_heads=2, num_key_value_heads=1, head_dim=4)
        torch.manual_seed(0)
        x = torch.randn(1, 6, 8)
        head_dim = config.head_dim
        n_rep = config.num_attention_heads // config.num_key_value_heads

        # Reference path (independent of Attention.forward):
        xq, xk, xv = attention.q_proj(x), attention.k_proj(x), attention.v_proj(x)
        xq = xq.view(1, 6, config.num_attention_heads, head_dim)
        xk = xk.view(1, 6, config.num_key_value_heads, head_dim)
        xv = xv.view(1, 6, config.num_key_value_heads, head_dim)
        xq, xk = attention.q_norm(xq), attention.k_norm(xk)
        cos = torch.ones(6, head_dim)
        sin = torch.zeros(6, head_dim)
        from model.model_minimind import apply_rotary_pos_emb

        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, n_rep).transpose(1, 2)
        xv = repeat_kv(xv, n_rep).transpose(1, 2)
        expected = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        expected = expected.transpose(1, 2).reshape(1, 6, -1)
        expected = attention.o_proj(expected)

        output, _ = attention(x, position_embeddings=(cos, sin))
        assert torch.allclose(output, expected, atol=1e-5)


class TestCausalLM:
    @pytest.fixture()
    def model(self):
        config = make_config(hidden_size=16, num_hidden_layers=2)
        torch.manual_seed(42)
        return MiniMindForCausalLM(config)

    def test_logits_shape_and_loss_shift(self, model):
        torch.manual_seed(7)
        input_ids = torch.randint(0, 64, (2, 13))
        logits = model(input_ids).logits
        assert logits.shape == (2, 13, 64)

        labels = input_ids.clone()
        labels[0, :4] = -100
        output = model(input_ids, labels=labels)

        expected = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, 64),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        assert torch.allclose(output.loss, expected, atol=1e-6)

    def test_loss_is_finite_and_small_for_cheating_forward(self, model):
        torch.manual_seed(3)
        input_ids = torch.randint(0, 64, (1, 9))
        labels = input_ids.clone()
        output = model(input_ids, labels=labels)
        assert torch.isfinite(output.loss)
