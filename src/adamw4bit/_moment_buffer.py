"""Block-aligned views of persistent moment state with bounded scratch space."""

from dataclasses import dataclass
import math
from typing import Any

import torch

from adamw4bit.quantization import QuantState, pack_4bit_codes
from adamw4bit._optimized_quantization import _codebook, dequantize, quantize


def tensor_byte_range(tensor: torch.Tensor) -> tuple[str, int, int]:
    """Describe storage without reading device values or allocating a copy."""
    if tensor.is_contiguous():
        start = tensor.data_ptr()
        return str(tensor.device), start, start + tensor.numel() * tensor.element_size()
    storage = tensor.untyped_storage()
    start = storage.data_ptr()
    return str(tensor.device), start, start + storage.nbytes()


def overlaps(left: tuple[str, int, int], right: tuple[str, int, int]) -> bool:
    return left[0] == right[0] and left[1] < right[2] and right[1] < left[2]


def moment_tensors(state: dict[str, Any]) -> list[torch.Tensor]:
    """The bounded list of writable buffers belonging to one parameter."""
    tensors = []
    for key in ("m1", "m2"):
        for name in (key, f"{key}_code"):
            value = state.get(name)
            if isinstance(value, torch.Tensor) and value.layout == torch.strided and value.numel():
                tensors.append(value)
        meta = state.get(f"{key}_quant_state")
        if (
            isinstance(meta, QuantState) and isinstance(meta.absmax, torch.Tensor)
            and meta.absmax.layout == torch.strided and meta.absmax.numel()
        ):
            tensors.append(meta.absmax)
    return tensors


@dataclass
class MomentBuffer:
    """A view of persistent FP32 values or packed codes and block scales."""

    scheme: str
    block_size: int
    values: torch.Tensor | None = None
    codes: torch.Tensor | None = None
    quant_state: QuantState | None = None

    @staticmethod
    def compatible(
        state: dict[str, Any], key: str, scheme: str, block_size: int,
        shape: torch.Size, device: torch.device,
    ) -> bool:
        """Check metadata without copying or decoding a complete tensor."""
        if "step" not in state:
            return True
        if scheme == "fp32":
            value = state.get(key)
            return (
                isinstance(value, torch.Tensor) and value.layout == torch.strided
                and value.is_contiguous() and value.shape == shape
                and value.device == device and value.dtype == torch.float32
            )
        codes = state.get(f"{key}_code")
        meta = state.get(f"{key}_quant_state")
        if not isinstance(codes, torch.Tensor) or not isinstance(meta, QuantState):
            return False
        n = shape.numel()
        if not (
            codes.layout == torch.strided and codes.is_contiguous()
            and codes.dtype == torch.uint8 and codes.device == device
            and meta.shape == shape and meta.blocksize == block_size
            and meta.dtype == torch.float32
            and isinstance(meta.absmax, torch.Tensor)
            and meta.absmax.ndim == 1 and meta.absmax.numel() == n // block_size
            and meta.absmax.is_contiguous()
            and meta.absmax.dtype == torch.float32 and meta.absmax.device == device
        ):
            return False
        if getattr(meta, "packed", False):
            return codes.numel() == (n + 1) // 2
        return codes.numel() in {n, (n + 1) // 2}

    @classmethod
    def open(
        cls, state: dict[str, Any], key: str, scheme: str, block_size: int,
        shape: torch.Size, device: torch.device, chunk_size: int,
        *, initialize: bool, eden: bool = False,
        generator: torch.Generator | None = None,
    ) -> "MomentBuffer":
        n = shape.numel()
        if scheme == "fp32":
            if initialize:
                # This allocation is the required persistent FP32 fallback
                # state, not a temporary decoded copy of quantized state.
                state[key] = torch.zeros(shape, device=device, dtype=torch.float32)
                state.pop(f"{key}_code", None)
                state.pop(f"{key}_quant_state", None)
            return cls(scheme, block_size, values=state[key].view(-1))

        # Populate per-device caches before tracing, including an NF4/SR switch.
        _codebook(scheme, device)
        if initialize:
            # Even tile size is important when a caller selects an odd block
            # size: repeating separately packed odd blocks inserts pad nibbles.
            tile_size = n if n <= chunk_size else math.lcm(block_size, 2)
            zero_tile = torch.zeros(tile_size, device=device, dtype=torch.float32)
            tile_codes, tile_meta = quantize(
                zero_tile, block_size=block_size, mode=scheme,
                eden_correction=eden, generator=generator,
            )
            first_byte = pack_4bit_codes(tile_codes)[:1]
            codes = first_byte.expand((n + 1) // 2).clone()
            if n % 2:
                codes[-1:].bitwise_and_(0x0F)
            scales = tile_meta.absmax[:1].expand(n // block_size).clone()
            meta = QuantState(absmax=scales, shape=shape, dtype=torch.float32,
                              blocksize=block_size, packed=True)
            state[f"{key}_code"] = codes
            state[f"{key}_quant_state"] = meta
            state.pop(key, None)
        else:
            codes = state[f"{key}_code"]
            meta = state[f"{key}_quant_state"]
            if codes.numel() == n and not getattr(meta, "packed", False):
                # Legacy unpacked codes need only a compact destination and a
                # bounded byte workspace; never dequantize them to repack.
                unpacked = codes.view(-1)
                packed = torch.empty((n + 1) // 2, dtype=torch.uint8, device=device)
                for start in range(0, n, chunk_size):
                    stop = min(start + chunk_size, n)
                    packed[start // 2:(stop + 1) // 2].copy_(pack_4bit_codes(unpacked[start:stop]))
                codes = packed
                state[f"{key}_code"] = codes
            # Older checkpoints may have compact storage but lack this flag.
            meta.packed = True
        return cls(scheme, block_size, codes=codes.view(-1), quant_state=meta)

    def chunk_view(self, start: int, stop: int) -> "MomentBuffer":
        """View an aligned chunk with local offsets and shared persistent storage."""
        shape = torch.Size((stop - start,))
        if self.values is not None:
            return MomentBuffer(self.scheme, self.block_size,
                                values=self.values[start:stop].detach())
        assert self.codes is not None and self.quant_state is not None
        scales = self.quant_state.absmax[start // self.block_size:stop // self.block_size].detach()
        meta = QuantState(absmax=scales, shape=shape, dtype=self.quant_state.dtype,
                          blocksize=self.block_size, packed=True)
        return MomentBuffer(self.scheme, self.block_size,
                            codes=self.codes[start // 2:(stop + 1) // 2].detach(), quant_state=meta)

    def read(self) -> torch.Tensor:
        """Decode this view into FP32 working values."""
        if self.values is not None:
            return self.values
        assert self.codes is not None and self.quant_state is not None
        return dequantize(self.codes, self.quant_state, mode=self.scheme)

    def write(
        self, value: torch.Tensor, *, eden: bool = False,
        generator: torch.Generator | None = None, bias_correction: float | torch.Tensor | None = None,
        eps: float | None = None, draws: torch.Tensor | None = None,
    ) -> None:
        """Encode working values back into this view's persistent storage."""
        if self.values is not None:
            self.values.copy_(value)
            return
        assert self.codes is not None and self.quant_state is not None
        codes, meta = quantize(
            value, block_size=self.block_size, mode=self.scheme, eden_correction=eden,
            generator=generator, bias_correction=bias_correction, eps=eps, draws=draws,
        )
        self.codes.copy_(pack_4bit_codes(codes))
        self.quant_state.absmax.copy_(meta.absmax)
