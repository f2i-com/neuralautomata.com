"""A local fast-weight primitive; not an end-to-end trained NCA.

W has shape [B,H,W,V,K]. Each cell has its own mutable matrix, but all
cells use the same update equation. No communication, global optimizer,
learned feature encoder, learned write gate or meta-training is supplied here.
The normalized delta update is differentiable for a later outer-loop trainer;
wrap runtime writes in torch.no_grad() when outer gradients are not needed.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor


def read(memory: Tensor, key: Tensor) -> Tensor:
    if memory.ndim != 5 or key.ndim != 4:
        raise ValueError('Expected memory [B,H,W,V,K] and key [B,H,W,K]')
    if memory.shape[:3] != key.shape[:3] or memory.shape[-1] != key.shape[-1]:
        raise ValueError('Incompatible memory/key dimensions')
    if memory.device != key.device or memory.dtype != key.dtype:
        raise ValueError('Memory and key must share device and dtype')
    if not memory.is_floating_point(): raise TypeError('Floating point tensors required')
    return torch.einsum('bhwvk,bhwk->bhwv',memory,key)


def write(memory: Tensor, key: Tensor, value: Tensor, *, rate: float = 1.0,
          write_mask: Tensor | None = None, eps: float = 1e-8) -> Tensor:
    """Observe value FIRST, then update for subsequent reads.

    W' = W + rate * mask * (value - W@key) outer key / (eps + ||key||^2)
    This is one gradient step on 0.5*||W@key-value||^2 with an input-normalized
    learning rate. For a fixed nonzero key and rate <= 1, its read residual
    contracts, absent a mask outside [0,1]. This is not a whole-memory bound:
    writes on correlated keys can interfere and varying keys can grow ||W||.
    """
    if not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError('rate must be finite and in [0,1]')
    if not math.isfinite(eps) or eps <= 0: raise ValueError('eps must be positive')
    prediction = read(memory,key)
    if value.shape != prediction.shape: raise ValueError('Value shape mismatch')
    if value.device != memory.device or value.dtype != memory.dtype:
        raise ValueError('Value must share device and dtype')
    if write_mask is None:
        write_mask = memory.new_ones(*memory.shape[:3],1)
    if write_mask.shape != (*memory.shape[:3],1): raise ValueError('Mask shape mismatch')
    if write_mask.device != memory.device: raise ValueError('Mask device mismatch')
    # Reference validation synchronizes on accelerators. Validate at a higher
    # level if profiling later justifies removing this check in a hot path.
    if not bool(torch.isfinite(write_mask).all() and
                ((write_mask>=0)&(write_mask<=1)).all()):
        raise ValueError('Mask must be finite and in [0,1]')
    error = value - prediction
    denominator = key.square().sum(-1,keepdim=True) + eps
    delta = error.unsqueeze(-1) * key.unsqueeze(-2)
    return memory + (rate*write_mask/denominator).unsqueeze(-1)*delta
