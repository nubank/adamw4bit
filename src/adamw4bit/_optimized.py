"""Opt-in bounded-workspace AdamW, with an unchanged reference fallback.

Each parameter is preflighted before state initialization or random draws.
Only ordinary AdamW reads are supported here; research read-side variants,
callbacks, clipping and unusual tensor layouts run the reference implementation.
"""

import math
from typing import TYPE_CHECKING, Any

import torch

from adamw4bit._moment_buffer import MomentBuffer, moment_tensors, overlaps, tensor_byte_range
from adamw4bit.quantization import default_block_size

if TYPE_CHECKING:
    from adamw4bit.adamw import QuantizedAdamW

_CHUNK_NUMEL = 1 << 20
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
        bias = (1 - beta1 ** state["step"], 1 - beta2 ** state["step"])
        storage_bias = 1 - beta2 ** (state["step"] + 1) if schemes[1] in _NEXT_BIAS_SCHEMES else bias[1]
        flat_parameter, flat_gradient = parameter.data.view(-1), gradient.view(-1)
        for start in range(0, parameter.numel(), chunk_size):
            stop = min(start + chunk_size, parameter.numel())
            _step_chunk(
                flat_parameter[start:stop], flat_gradient[start:stop], first, second, start, stop,
                betas=(beta1, beta2), bias=bias, storage_bias=storage_bias,
                lr=group["lr"], weight_decay=group["weight_decay"], eps=group["eps"],
                eden=group["use_eden_m2"], generators=generators,
            )


def _step_chunk(
    parameter: torch.Tensor, gradient: torch.Tensor,
    first: MomentBuffer, second: MomentBuffer, start: int, stop: int,
    *, betas: tuple[float, float], bias: tuple[float, float], storage_bias: float,
    lr: float, weight_decay: float, eps: float, eden: bool,
    generators: tuple[torch.Generator | None, torch.Generator | None],
) -> None:
    # Separate frame: scratch from this chunk dies before the next decode.
    g = gradient.to(dtype=torch.float32)
    m1, m2 = first.read(start, stop), second.read(start, stop)
    m1.lerp_(g, 1 - betas[0])
    m2.lerp_(g.square(), 1 - betas[1])
    denominator = m2.sqrt().div_(math.sqrt(bias[1])).add_(eps)
    parameter.mul_(1 - lr * weight_decay)
    parameter.addcdiv_(m1, denominator, value=-lr / bias[0])
    first.write(start, stop, m1, generator=generators[0])
    second.write(start, stop, m2, eden=eden, generator=generators[1],
                 bias_correction=storage_bias, eps=eps)
