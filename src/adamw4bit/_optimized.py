"""Opt-in bounded-workspace AdamW, with an unchanged reference fallback.

Each parameter is preflighted before state initialization or random draws.
Only ordinary AdamW reads are supported here; research read-side variants,
callbacks, clipping and unusual tensor layouts run the reference implementation.
"""

from collections.abc import Callable
from functools import lru_cache
import math
from typing import TYPE_CHECKING, Any

import torch

from adamw4bit._moment_buffer import MomentBuffer, moment_tensors, overlaps, tensor_byte_range
from adamw4bit._optimized_quantization import _uniform_draws
from adamw4bit.quantization import (
    _STRATEGIES,
    _StochasticCodebookQuantizationStrategy,
    _UpdateUnbiasedStochasticCodebookQuantizationStrategy,
    default_block_size,
)

if TYPE_CHECKING:
    from adamw4bit.adamw import QuantizedAdamW

_Scalar = float | torch.Tensor
_STOCHASTIC_STRATEGIES = (
    _StochasticCodebookQuantizationStrategy,
    _UpdateUnbiasedStochasticCodebookQuantizationStrategy,
)

_CHUNK_NUMEL = 1 << 19
_CHUNK_KEY = "max_chunk_numel"
_FALLBACK_KEY = "_optimized_full_fallback"
_M1_SCHEMES = {"fp32", "nf4", "nf4_sr", "sdyn4", "sdyn4_sr"}
_M2_SCHEMES = {
    "fp32", "lin4", "lin4_sr", "lin4_nz", "dyn4", "dyn4_sr", "dyn4_nz",
    "lin4_upd_sr_fp32read", "dyn4_upd_sr_fp32read",
    "dyn4_lookahead_sr", "dyn4_upd_sr_fp32read_nextbc",
}
_NEXT_BIAS_SCHEMES = {"dyn4_lookahead_sr", "dyn4_upd_sr_fp32read_nextbc"}


class OptimizedBackend:
    """Own only dispatch and checkpoint policy; moment state stays on the optimizer."""

    def __init__(self, optimizer: "QuantizedAdamW") -> None:
        for group in optimizer.param_groups:
            group.setdefault(_CHUNK_KEY, _CHUNK_NUMEL)
        optimizer.register_load_state_dict_post_hook(self._after_load)

    @staticmethod
    def _after_load(optimizer: "QuantizedAdamW") -> None:
        # The reference loader restores original code/scale dtypes in its
        # prepended post-hook before this hook examines loaded storage.
        intervals: list[tuple[str, int, int, int]] = []
        for group in optimizer.param_groups:
            group.setdefault(_CHUNK_KEY, _CHUNK_NUMEL)
            for parameter in group["params"]:
                for tensor in moment_tensors(optimizer.state.get(parameter, {})):
                    intervals.append((*tensor_byte_range(tensor), id(parameter)))
        intervals.sort()
        aliased: set[int] = set()
        owners: set[int] = set()
        device, end, count = "", 0, 0
        for current_device, start, stop, owner in intervals:
            if current_device != device or start >= end:
                if count > 1:
                    aliased.update(owners)
                device, end, count, owners = current_device, stop, 1, {owner}
            else:
                end = max(end, stop)
                count += 1
                owners.add(owner)
        if count > 1:
            aliased.update(owners)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if id(parameter) in aliased:
                    # Reference writes can detach aliases. Persist this choice
                    # so later save/resume cannot change RNG chunk partitioning.
                    optimizer.state[parameter][_FALLBACK_KEY] = True

    @staticmethod
    def _plan(
        optimizer: "QuantizedAdamW", parameter: torch.Tensor, group: dict[str, Any],
    ) -> tuple[tuple[str, str], tuple[int, int], int] | None:
        gradient = parameter.grad
        assert gradient is not None
        state = optimizer.state.get(parameter, {})
        requested = (optimizer._m1_scheme_for(parameter, group["m1_quant_scheme"]),
                     group["m2_quant_scheme"])
        if (
            state.get(_FALLBACK_KEY, False)
            or requested[0] not in _M1_SCHEMES or requested[1] not in _M2_SCHEMES
            or parameter.layout != torch.strided or gradient.layout != torch.strided
            or not parameter.is_contiguous() or not gradient.is_contiguous()
            or parameter.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}
            or ("step" in state and not isinstance(state["step"], int))
        ):
            return None
        sizes = tuple(group["block_size"] if group["block_size"] is not None
                      else default_block_size(mode) for mode in requested)
        schemes = tuple(optimizer._effective_scheme(parameter, mode, size)
                        for mode, size in zip(requested, sizes))
        for key, scheme, size in zip(("m1", "m2"), schemes, sizes):
            if not MomentBuffer.compatible(state, key, scheme, size, parameter.shape, parameter.device):
                return None
        occupied = [tensor_byte_range(parameter), tensor_byte_range(gradient)]
        if overlaps(occupied[0], occupied[1]):
            # A chunk update must not change gradients used by later chunks.
            return None
        for tensor in moment_tensors(state):
            region = tensor_byte_range(tensor)
            if any(overlaps(region, previous) for previous in occupied):
                return None
            occupied.append(region)
        limit = group.get(_CHUNK_KEY, _CHUNK_NUMEL)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            return None
        quantized_sizes = [size for scheme, size in zip(schemes, sizes) if scheme != "fp32"]
        alignment = math.lcm(2, *quantized_sizes) if quantized_sizes else 1
        if limit < alignment:
            return None
        return (schemes[0], schemes[1]), (sizes[0], sizes[1]), limit // alignment * alignment

    def try_step_parameter(
        self, optimizer: "QuantizedAdamW", parameter: torch.Tensor, group: dict[str, Any],
    ) -> bool:
        """Fall back without consuming randomness or updating this parameter."""
        if optimizer.telemetry is not None or any(group.get(key) is not None for key in (
            "update_sign_clip", "update_norm_clip", "record_update_direction_every",
            "record_preconditioner_every", "record_preconditioner_steps",
        )):
            return False
        plan = self._plan(optimizer, parameter, group)
        if plan is None:
            return False
        schemes, sizes, chunk_size = plan
        group.setdefault(_CHUNK_KEY, _CHUNK_NUMEL)
        self._step_parameter(optimizer, parameter, group, schemes, sizes, chunk_size)
        return True

    @staticmethod
    def _step_parameter(
        optimizer: "QuantizedAdamW", parameter: torch.Tensor, group: dict[str, Any],
        schemes: tuple[str, str], sizes: tuple[int, int], chunk_size: int,
    ) -> None:
        gradient = parameter.grad
        assert gradient is not None
        state = optimizer.state[parameter]
        optimizer._last_update_directions.pop(parameter, None)
        optimizer._last_preconditioners.pop(parameter, None)
        optimizer._last_effective_preconditioners.pop(parameter, None)
        state.pop("m2_read_preconditioner", None)
        generators = (optimizer._quant_generator("m1", parameter.device),
                      optimizer._quant_generator("m2", parameter.device))
        initialize = "step" not in state
        first = MomentBuffer.open(
            state, "m1", schemes[0], sizes[0], parameter.shape, parameter.device, chunk_size,
            initialize=initialize, generator=generators[0],
        )
        second = MomentBuffer.open(
            state, "m2", schemes[1], sizes[1], parameter.shape, parameter.device, chunk_size,
            initialize=initialize, eden=group["use_eden_m2"], generator=generators[1],
        )
        state["step"] = state.get("step", 0) + 1
        beta1, beta2 = group["betas"]
        bias: tuple[_Scalar, _Scalar] = (1 - beta1 ** state["step"], 1 - beta2 ** state["step"])
        storage_bias: _Scalar = 1 - beta2 ** (state["step"] + 1) if schemes[1] in _NEXT_BIAS_SCHEMES else bias[1]
        flat_parameter, flat_gradient = parameter.data.view(-1), gradient.view(-1)
        lr: _Scalar = group["lr"]
        step_chunk = _step_chunk
        use_compile = parameter.is_cuda and parameter.dtype == torch.float32
        if use_compile:
            # Step-dependent values remain inputs rather than graph constants.
            bias = (torch.as_tensor(bias[0], dtype=torch.float64, device="cpu"),
                    torch.as_tensor(bias[1], dtype=torch.float64, device="cpu"))
            storage_bias = torch.as_tensor(storage_bias, dtype=torch.float64, device="cpu")
            lr = torch.as_tensor(lr, dtype=torch.float64, device="cpu")
            step_chunk = _compiled_step_chunk(schemes[0], schemes[1], group["use_eden_m2"], parameter.device)
        for start in range(0, parameter.numel(), chunk_size):
            stop = min(start + chunk_size, parameter.numel())
            step_chunk(
                flat_parameter[start:stop].detach(), flat_gradient[start:stop].detach(),
                first.chunk_view(start, stop), second.chunk_view(start, stop),
                betas=(beta1, beta2), bias=bias, storage_bias=storage_bias,
                lr=lr, weight_decay=group["weight_decay"], eps=group["eps"],
                eden=group["use_eden_m2"], generators=generators,
            )


def _step_chunk(
    parameter: torch.Tensor, gradient: torch.Tensor,
    first: MomentBuffer, second: MomentBuffer,
    *, betas: tuple[float, float], bias: tuple[_Scalar, _Scalar], storage_bias: _Scalar,
    lr: _Scalar, weight_decay: float, eps: float, eden: bool,
    generators: tuple[torch.Generator | None, torch.Generator | None],
) -> None:
    # Separate frame: scratch from this chunk dies before the next decode.
    g = gradient.to(dtype=torch.float32)
    m1, m2 = first.read(), second.read()
    m1.lerp_(g, 1 - betas[0])
    m2.lerp_(g.square(), 1 - betas[1])
    bias_sqrt = bias[1].sqrt() if isinstance(bias[1], torch.Tensor) else math.sqrt(bias[1])
    denominator = m2.sqrt().div_(bias_sqrt).add_(eps)
    parameter.mul_(1 - lr * weight_decay)
    parameter.addcdiv_(m1, denominator, value=-lr / bias[0])
    # Release update scratch before allocating quantization intermediates.
    del g, denominator
    draws = None
    if isinstance(_STRATEGIES.get(first.scheme), _STOCHASTIC_STRATEGIES):
        draws = _uniform_draws(m1, generators[0], block_size=first.block_size)
    first.write(m1, generator=generators[0], draws=draws)
    del m1, draws
    draws = None
    if isinstance(_STRATEGIES.get(second.scheme), _STOCHASTIC_STRATEGIES):
        draws = _uniform_draws(m2, generators[1], block_size=second.block_size)
    second.write(m2, eden=eden, generator=generators[1],
                 bias_correction=storage_bias, eps=eps, draws=draws)


@lru_cache(maxsize=None)
def _compiled_step_chunk(
    m1_scheme: str, m2_scheme: str, eden: bool, device: torch.device,
) -> Callable[..., None]:
    """Use the arguments as cache keys to isolate each scheme/device configuration."""
    return torch.compile(_step_chunk, dynamic=True, isolate_recompiles=True)
