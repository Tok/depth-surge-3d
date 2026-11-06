"""Tests for the attention backend guard (xformers -> torch SDPA fallback)."""

import sys
from types import ModuleType
from unittest.mock import patch

import pytest
import torch

from src.depth_surge_3d.utils.system import xformers_guard
from src.depth_surge_3d.utils.system.xformers_guard import (
    ensure_attention_backend,
    memory_efficient_attention,
)

pytestmark = pytest.mark.real_xformers_check


def _naive_attention(q, k, v):
    """Reference attention in xformers layout [B, M, H, K]."""
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    attn = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    return (attn.softmax(-1) @ v).transpose(1, 2)


class TestMemoryEfficientAttention:
    def test_matches_naive_4d(self):
        q, k, v = (torch.randn(2, 10, 4, 8) for _ in range(3))
        assert torch.allclose(
            memory_efficient_attention(q, k, v), _naive_attention(q, k, v), atol=1e-5
        )

    def test_3d_layout(self):
        q, k, v = (torch.randn(3, 7, 16) for _ in range(3))
        out = memory_efficient_attention(q, k, v)
        ref = _naive_attention(q.unsqueeze(2), k.unsqueeze(2), v.unsqueeze(2)).squeeze(2)
        assert out.shape == (3, 7, 16)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_tensor_bias(self):
        q, k, v = (torch.randn(1, 5, 2, 4) for _ in range(3))
        bias = torch.zeros(1, 2, 5, 5)
        assert torch.allclose(
            memory_efficient_attention(q, k, v, attn_bias=bias),
            memory_efficient_attention(q, k, v),
            atol=1e-6,
        )

    def test_structured_bias_unsupported(self):
        q = torch.randn(1, 4, 1, 4)
        with pytest.raises(NotImplementedError):
            memory_efficient_attention(q, q, q, attn_bias=object())


def _fake_xformers(has_kernels: bool) -> dict:
    mod = ModuleType("xformers")
    mod._has_cpp_library = has_kernels
    return {"xformers": mod, "xformers.ops": ModuleType("xformers.ops")}


class TestEnsureAttentionBackend:
    def test_broken_kernels_replaced_by_shim(self):
        with patch.dict(sys.modules, _fake_xformers(False)):
            assert ensure_attention_backend() == "sdpa"
            from xformers.ops import memory_efficient_attention as mea, unbind, fmha  # noqa: F401

            assert mea is memory_efficient_attention
            # optional extras must be missing so models use their own implementations
            with pytest.raises(ImportError):
                from xformers.ops import SwiGLU  # noqa: F401
            with pytest.raises(ImportError):
                from xformers.ops import scaled_index_add  # noqa: F401

    def test_working_kernels_kept(self):
        fake = _fake_xformers(True)
        with patch.dict(sys.modules, fake):
            assert ensure_attention_backend() == "xformers"
            assert sys.modules["xformers"] is fake["xformers"]

    def test_missing_xformers_gets_shim(self):
        with patch.dict(sys.modules):
            sys.modules.pop("xformers", None)
            sys.modules.pop("xformers.ops", None)
            with patch.object(xformers_guard, "_real_xformers_status", return_value="missing"):
                assert ensure_attention_backend() == "sdpa"
            import xformers
            import xformers.ops

            assert xformers.ops.memory_efficient_attention is memory_efficient_attention

    def test_idempotent(self):
        with patch.dict(sys.modules, _fake_xformers(False)):
            assert ensure_attention_backend() == "sdpa"
            shim = sys.modules["xformers"]
            assert ensure_attention_backend() == "sdpa"
            assert sys.modules["xformers"] is shim
