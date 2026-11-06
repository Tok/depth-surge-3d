"""
Attention backend guard for the depth models.

Video-Depth-Anything and Depth-Anything-3 use ``xformers.ops.memory_efficient_attention``
when ``import xformers`` succeeds and otherwise fall back to naive attention, which
materializes the full attention matrix (~20 GB for a 32-frame window at 720p).

That leaves two broken states:

- xformers is installed (DA3 depends on it) but its compiled kernels don't match the
  installed torch/CUDA build: the import succeeds, then the first attention call fails
  with "No operator found for memory_efficient_attention_forward".
- xformers isn't installed (macOS, CPU, ROCm): the naive fallback runs out of memory.

In both cases we register a tiny stand-in ``xformers`` package whose
``memory_efficient_attention`` is backed by ``torch.nn.functional.scaled_dot_product_attention``,
which is just as memory-efficient and ships with every torch build. Everything else the
models optionally import from xformers (SwiGLU, fused index ops) is deliberately missing,
so those imports raise ImportError and the models use their own PyTorch implementations.
"""

from __future__ import annotations

import contextlib
import io
import sys
import types
import warnings
from typing import Any

SHIM_VERSION = "0.0.0+sdpa-shim"


def memory_efficient_attention(
    query: Any,
    key: Any,
    value: Any,
    attn_bias: Any = None,
    p: float = 0.0,
    scale: float | None = None,
    **_: Any,
) -> Any:
    """
    xformers-compatible attention on top of torch's scaled_dot_product_attention.

    Accepts xformers' layouts: [B, M, H, K] or [B, M, K]. ``attn_bias`` may be None or an
    additive tensor broadcastable to [B, H, M, N]; xformers' structured bias classes
    (block-diagonal masks for nested tensors) are not supported.
    """
    import torch
    import torch.nn.functional as F

    if attn_bias is not None and not isinstance(attn_bias, torch.Tensor):
        raise NotImplementedError(
            f"attn_bias of type {type(attn_bias).__name__} needs the real xformers kernels"
        )

    squeeze_heads = query.dim() == 3
    if squeeze_heads:
        query, key, value = (t.unsqueeze(2) for t in (query, key, value))

    # [B, M, H, K] -> [B, H, M, K]
    out = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=attn_bias,
        dropout_p=p,
        scale=scale,
    ).transpose(1, 2)

    return out.squeeze(2) if squeeze_heads else out


def _build_shim() -> tuple[types.ModuleType, types.ModuleType]:
    """Create the stand-in ``xformers`` and ``xformers.ops`` modules."""
    import torch

    pkg = types.ModuleType("xformers")
    pkg.__path__ = []  # mark as package so "import xformers.ops" resolves via sys.modules
    pkg.__version__ = SHIM_VERSION  # type: ignore[attr-defined]
    pkg._has_cpp_library = False  # type: ignore[attr-defined]
    pkg._is_sdpa_shim = True  # type: ignore[attr-defined]

    ops = types.ModuleType("xformers.ops")
    ops.memory_efficient_attention = memory_efficient_attention  # type: ignore[attr-defined]
    ops.unbind = torch.unbind  # type: ignore[attr-defined]
    ops.fmha = types.ModuleType("xformers.ops.fmha")  # type: ignore[attr-defined]

    pkg.ops = ops  # type: ignore[attr-defined]
    return pkg, ops


def _real_xformers_status() -> str:
    """Return "ok", "broken" or "missing" for the installed xformers."""
    try:
        # xformers prints a long warning when its kernels fail to load; keep it quiet
        with (
            warnings.catch_warnings(),
            contextlib.redirect_stderr(io.StringIO()),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            warnings.simplefilter("ignore")
            import xformers
    except Exception:
        return "missing"
    return "ok" if getattr(xformers, "_has_cpp_library", True) else "broken"


def ensure_attention_backend() -> str:
    """
    Make sure the depth models get memory-efficient attention.

    Must run before the depth model code is imported. Idempotent.

    Returns:
        "xformers" if the real xformers kernels are usable, otherwise "sdpa"
        (the stand-in module was installed)

    Side effects:
        - May replace ``sys.modules["xformers"]`` / ``["xformers.ops"]`` with the shim
    """
    current = sys.modules.get("xformers")
    if current is not None and getattr(current, "_is_sdpa_shim", False):
        return "sdpa"

    if _real_xformers_status() == "ok":
        return "xformers"

    for name in list(sys.modules):
        if name == "xformers" or name.startswith("xformers."):
            del sys.modules[name]

    pkg, ops = _build_shim()
    sys.modules["xformers"] = pkg
    sys.modules["xformers.ops"] = ops
    return "sdpa"
