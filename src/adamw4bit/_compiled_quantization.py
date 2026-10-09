"""Shared rounding contracts and deterministic CUDA quantization kernels.

Callers supply normalized values and any random draws. Normalization and EDEN
scale corrections belong to the caller and can run eagerly or within a compiled
chunk update. The shared rounding functions return codes and can be compiled on
their own or inlined into that update. Exact-level guards are shared with eager
strategies, including when adjacent preconditioners coincide in FP32.
"""

from functools import lru_cache
from typing import Callable, Literal

import torch


def supported(scaled: torch.Tensor, codebook: torch.Tensor) -> bool:
    """Select the supported normalized FP32 CUDA path."""
    return (
        scaled.is_cuda
        and scaled.dtype == torch.float32
        and scaled.ndim == 2
        and codebook.is_cuda
        and codebook.dtype == torch.float32
        and codebook.ndim == 1
        and codebook.numel() == 16
        and codebook.device == scaled.device
    )


def _endpoints(
    scaled: torch.Tensor, codebook: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.searchsorted(codebook, scaled, out_int32=True)
    lower = (indices - 1).clamp(min=0)
    upper = indices.clamp(max=codebook.numel() - 1)
    a = torch.index_select(codebook, 0, lower.reshape(-1)).reshape(scaled.shape)
    b = torch.index_select(codebook, 0, upper.reshape(-1)).reshape(scaled.shape)
    return lower, upper, a, b


def _rtn(scaled: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    lower, upper, a, b = _endpoints(scaled, codebook)
    # Retain the original distance comparison and round ties upward.
    pick_lower = (scaled - a).abs() < (b - scaled).abs()
    return torch.where(pick_lower, lower, upper).to(torch.uint8)


def endpoint_probability(
    probability: torch.Tensor,
    at_lower: torch.Tensor,
    at_upper: torch.Tensor,
    lower_value: float,
    upper_value: float,
) -> torch.Tensor:
    """Make exact reconstruction levels and saturation independent of draws."""
    probability = torch.where(at_upper, upper_value, probability)
    return torch.where(at_lower, lower_value, probability)


def _state_probability(scaled: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    width = b - a
    probability = (scaled - a) / width.clamp_min(1e-12)
    probability = torch.where(width > 0, probability, 0.0).clamp(0.0, 1.0)
    # Exact levels and saturation must not depend on rounding a probability.
    return endpoint_probability(probability, scaled <= a, scaled >= b, 0.0, 1.0)


def _state_sr(scaled: torch.Tensor, codebook: torch.Tensor, draws: torch.Tensor) -> torch.Tensor:
    lower, upper, a, b = _endpoints(scaled, codebook)
    probability = _state_probability(scaled, a, b)
    return torch.where(draws < probability, upper, lower).to(torch.uint8)


def _update_probability(
    scaled: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    block_scales: torch.Tensor,
    bias: torch.Tensor,
    eps: torch.Tensor,
) -> torch.Tensor:
    scales = block_scales.unsqueeze(1)

    def preconditioner(levels: torch.Tensor) -> torch.Tensor:
        v_hat = (scales * levels / bias).clamp_min(0.0)
        return 1.0 / (v_hat.sqrt() + eps)

    f_a = preconditioner(a)
    f_b = preconditioner(b)
    f_x = preconditioner(scaled)
    denominator = f_a - f_b
    probability = (f_x - f_b) / denominator.clamp_min(1e-30)
    probability = torch.where((b > a) & (denominator > 0), probability, 1.0)
    probability = probability.clamp(0.0, 1.0)
    # A flat FP32 preconditioner interval does not override exact level writes.
    return endpoint_probability(probability, scaled <= a, scaled >= b, 1.0, 0.0)


def _update_sr(
    scaled: torch.Tensor,
    codebook: torch.Tensor,
    block_scales: torch.Tensor,
    draws: torch.Tensor,
    bias: torch.Tensor,
    eps: torch.Tensor,
) -> torch.Tensor:
    lower, upper, a, b = _endpoints(scaled, codebook)
    probability = _update_probability(scaled, a, b, block_scales, bias, eps)
    return torch.where(draws < probability, lower, upper).to(torch.uint8)


@lru_cache(maxsize=3)
def kernel(kind: Literal["rtn", "state_sr", "update_sr"]) -> Callable:
    """Compile lazily; tensor-valued scalars avoid per-step value guards."""
    function = {"rtn": _rtn, "state_sr": _state_sr, "update_sr": _update_sr}[kind]
    return torch.compile(function, fullgraph=True, dynamic=True)
