"""Bounded-chunk codecs used only by the opt-in optimizer backend.

Codebooks, packed layout and scale metadata come from the reference module.
Normalization and EDEN reductions remain eager; CUDA compiles only the pure
rounding formulas. Uniform draws stay outside compilation and use the optimizer's
checkpointed generators. Exact-level guards are shared by CPU and CUDA paths.
"""

import torch

from adamw4bit import _compiled_quantization as rounding
from adamw4bit.quantization import (
    QuantState,
    _STRATEGIES,
    _block,
    _unblock,
    _CodebookQuantizationStrategy,
    _StochasticCodebookQuantizationStrategy,
    _UpdateUnbiasedStochasticCodebookQuantizationStrategy,
    UnsignedDynamic4NoZeroStrategy,
    default_min_scale,
)


def _codebook(mode: str, device: torch.device) -> torch.Tensor:
    strategy = _STRATEGIES[mode]
    assert isinstance(strategy, _CodebookQuantizationStrategy)
    return strategy._get_codebook(device)


def _lookup(codebook: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    return torch.index_select(codebook, 0, codes.reshape(-1).to(torch.int32)).reshape(codes.shape)


def quantize(
    value: torch.Tensor, block_size: int, mode: str, *, eden_correction: bool = False,
    generator: torch.Generator | None = None, bias_correction: float | None = None,
    eps: float | None = None,
) -> tuple[torch.Tensor, QuantState]:
    """Encode an FP32 working chunk with the original normalized formulas."""
    assert value.dtype == torch.float32
    blocked, shape = _block(value, block_size)
    scales = blocked.abs().max(dim=1).values.clamp_min_(default_min_scale(mode))
    scaled = blocked / scales.unsqueeze(1)
    codebook = _codebook(mode, value.device)
    strategy = _STRATEGIES[mode]
    use_compile = rounding.supported(scaled, codebook)
    if isinstance(strategy, _UpdateUnbiasedStochasticCodebookQuantizationStrategy):
        draws = torch.rand(scaled.shape, device=scaled.device, dtype=scaled.dtype, generator=generator)
        bias = torch.tensor(max(bias_correction if bias_correction is not None else 1.0, 1e-12),
                            dtype=torch.float64)
        epsilon = torch.tensor(eps if eps is not None else 1e-8, dtype=torch.float64)
        function = rounding.kernel("update_sr") if use_compile else rounding._update_sr
        codes = function(scaled, codebook, scales, draws, bias, epsilon)
    elif isinstance(strategy, _StochasticCodebookQuantizationStrategy):
        draws = torch.rand(scaled.shape, device=scaled.device, dtype=scaled.dtype, generator=generator)
        function = rounding.kernel("state_sr") if use_compile else rounding._state_sr
        codes = function(scaled, codebook, draws)
    else:
        function = rounding.kernel("rtn") if use_compile else rounding._rtn
        codes = function(scaled, codebook)
    if isinstance(strategy, UnsignedDynamic4NoZeroStrategy):
        codes.masked_fill_(codes == strategy.ZERO_CODE, strategy.MIN_POSITIVE_CODE)
    if eden_correction:
        levels = _lookup(codebook, codes)
        numerator = (scaled * scaled).sum(dim=1)
        denominator = (scaled * levels).sum(dim=1)
        scales = scales * (numerator / denominator.clamp_min(1e-12))
    return _unblock(codes, shape), QuantState(
        absmax=scales, shape=shape, dtype=value.dtype, blocksize=block_size,
    )


def dequantize(codes: torch.Tensor, quant_state: QuantState, mode: str) -> torch.Tensor:
    """Decode one compact chunk without int64 codebook indices."""
    count = quant_state.shape.numel()
    if getattr(quant_state, "packed", False):
        packed = codes.reshape(-1)
        unpacked = torch.empty(count, device=codes.device, dtype=torch.uint8)
        torch.bitwise_and(packed, 0x0F, out=unpacked[0::2])
        torch.bitwise_right_shift(packed[:count // 2], 4, out=unpacked[1::2])
        codes = unpacked
    blocked, _ = _block(codes, quant_state.blocksize)
    values = _lookup(_codebook(mode, codes.device), blocked)
    scales = quant_state.absmax.unsqueeze(1)
    if scales.dtype == values.dtype:
        values.mul_(scales)
    else:
        values = values * scales
    return _unblock(values, quant_state.shape).to(quant_state.dtype)
