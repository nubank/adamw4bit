import logging
from abc import ABC, abstractmethod
from typing import Dict, Literal, Tuple

import torch

try:
    import bitsandbytes.functional as F

    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False


class QuantState:
    """A container for quantization state components."""

    def __init__(
        self,
        absmax,
        shape=None,
        dtype=None,
        blocksize=None,
        packed: bool = False,
    ):
        self.absmax = absmax
        self.shape = shape
        self.dtype = dtype
        self.blocksize = blocksize
        self.packed = packed


# ====================================================================
#
#      Helper Functions
#
# ====================================================================


def _block(x: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, torch.Size]:
    """Pads and reshapes a tensor into blocks."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    original_shape = x.shape
    flat = x.flatten()
    n = flat.numel()
    n_blocks = (n + block_size - 1) // block_size
    padded_len = n_blocks * block_size
    if padded_len == n:
        return flat.view(n_blocks, block_size), original_shape
    padded = torch.zeros(padded_len, dtype=flat.dtype, device=flat.device)
    padded[:n] = flat
    return padded.view(n_blocks, block_size), original_shape


def _unblock(blocked_tensor: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """Unpads and reshapes blocks back to the original tensor shape."""
    n = original_shape.numel()
    flat = blocked_tensor.flatten()[:n]
    return flat.view(original_shape)


# ====================================================================
#
#      Quantization Strategy Pattern
#
# ====================================================================


class QuantizationStrategy(ABC):
    """Abstract base class for quantization strategies."""

    @abstractmethod
    def quantize(
        self,
        scaled_blocked: torch.Tensor,
        generator: torch.Generator | None = None,
        *,
        block_scales: torch.Tensor | None = None,
        bias_correction: float | None = None,
        eps: float | None = None,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def dequantize(self, blocked_codes: torch.Tensor) -> torch.Tensor:
        pass


class LinearQuantizationStrategy(QuantizationStrategy):
    """Strategy for linear quantization."""

    def quantize(
        self,
        scaled_blocked: torch.Tensor,
        generator: torch.Generator | None = None,
        *,
        block_scales: torch.Tensor | None = None,
        bias_correction: float | None = None,
        eps: float | None = None,
    ) -> torch.Tensor:
        return torch.clamp(scaled_blocked * 127.0, -127.0, 127.0).round().to(torch.int8)

    def dequantize(self, blocked_codes: torch.Tensor) -> torch.Tensor:
        return blocked_codes.to(torch.float32) / 127.0


class DynamicQuantizationStrategy(QuantizationStrategy):
    """Strategy for dynamic quantization."""

    def __init__(self):
        self._dynamic_map_cache: Dict[torch.device, torch.Tensor] = {}

    def _get_dynamic_map(self, device: torch.device) -> torch.Tensor:
        if device not in self._dynamic_map_cache:
            self._dynamic_map_cache[device] = self._create_dynamic_map().to(device)
        return self._dynamic_map_cache[device]

    def _create_dynamic_map(
        self, signed=True, max_exponent_bits=7, total_bits=8
    ) -> torch.Tensor:
        """Replicates the dynamic quantization map from bitsandbytes, optimized with PyTorch."""
        tensors = []
        non_sign_bits = total_bits - 1
        additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1

        for i in range(max_exponent_bits):
            fraction_items = int(2 ** (i + non_sign_bits - max_exponent_bits) + 1)
            boundaries = torch.linspace(0.1, 1, fraction_items, dtype=torch.float32)
            means = (boundaries[:-1] + boundaries[1:]) / 2.0

            base_multiplier = 10 ** (-(max_exponent_bits - 1) + i)
            tensors.append(base_multiplier * means)
            if signed:
                tensors.append(-base_multiplier * means)

        if additional_items > 0:
            boundaries = torch.linspace(
                0.1, 1, additional_items + 1, dtype=torch.float32
            )
            means = (boundaries[:-1] + boundaries[1:]) / 2.0

            # The original code has a bug, using `i` from the previous loop.
            # The last value of i is max_exponent_bits - 1.
            # So the multiplier is 10**(-(max_exponent_bits - 1) + max_exponent_bits - 1) = 10**0 = 1.
            base_multiplier = 1.0
            tensors.append(base_multiplier * means)
            if signed:
                tensors.append(-base_multiplier * means)

        # Combine all tensors, add special values, sort, and pad
        combined = torch.cat(tensors)
        combined = torch.cat([combined, torch.tensor([0.0, 1.0], dtype=torch.float32)])

        gap = 256 - combined.numel()
        if gap > 0:
            combined = torch.cat([combined, torch.zeros(gap, dtype=torch.float32)])

        sorted_map = torch.sort(combined).values
        return sorted_map

    def quantize(
        self,
        scaled_blocked: torch.Tensor,
        generator: torch.Generator | None = None,
        *,
        block_scales: torch.Tensor | None = None,
        bias_correction: float | None = None,
        eps: float | None = None,
    ) -> torch.Tensor:
        """Quantizes a tensor using an efficient sorted search."""
        dynamic_map = self._get_dynamic_map(scaled_blocked.device)

        # `torch.searchsorted` finds the indices of the buckets where the values should go.
        indices = torch.searchsorted(dynamic_map, scaled_blocked)

        # Get the indices of the two nearest candidates in the map
        indices_left = (indices - 1).clamp_(min=0)
        indices_right = indices.clamp_(max=255)

        # Get the actual candidate values
        vals_left = dynamic_map[indices_left]
        vals_right = dynamic_map[indices_right]

        # Determine which of the two candidates is closer
        diff_left = scaled_blocked - vals_left
        diff_right = vals_right - scaled_blocked

        # Choose the index of the closer candidate, replicating argmin's tie-breaking.
        codes = torch.where(
            diff_left.abs() <= diff_right.abs(), indices_left, indices_right
        )

        return codes.to(torch.uint8)

    def dequantize(self, blocked_codes: torch.Tensor) -> torch.Tensor:
        dynamic_map = self._get_dynamic_map(blocked_codes.device)
        return dynamic_map[blocked_codes.long()].to(torch.float32)


def _create_dynamic_map(
    signed: bool = True, max_exponent_bits: int = 7, total_bits: int = 8
) -> torch.Tensor:
    """Dettmers/bitsandbytes dynamic-exponent quantization map (a nonlinear codebook
    that packs more levels near zero). Generic over sign and bit-width, matching
    bitsandbytes / TorchAO ``create_dynamic_map``. Used here for 4-bit optimizer
    state codebooks: ``signed=True`` for signed m1, and ``signed=False`` for
    non-negative m2. Both variants include a zero level and are dense near zero.
    """
    data = []
    non_sign_bits = total_bits - 1
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    i = 0
    for i in range(max_exponent_bits):
        fraction_items = int(
            2 ** (i + non_sign_bits - max_exponent_bits) + 1
            if signed
            else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1
        )
        boundaries = torch.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
    if additional_items > 0:
        boundaries = torch.linspace(0.1, 1, additional_items + 1)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
    data.append(0.0)
    data.append(1.0)
    assert len(data) == 2 ** total_bits, (
        f"dynamic map has {len(data)} levels, expected {2 ** total_bits}"
    )
    data.sort()
    return torch.tensor(data, dtype=torch.float32)


class _CodebookQuantizationStrategy(QuantizationStrategy):
    """Nearest-neighbour quantization against a fixed 1-D sorted codebook."""

    def __init__(self):
        self._codebook_cache: Dict[torch.device, torch.Tensor] = {}

    def _build_codebook(self) -> torch.Tensor:
        raise NotImplementedError

    def _get_codebook(self, device: torch.device) -> torch.Tensor:
        if device not in self._codebook_cache:
            self._codebook_cache[device] = self._build_codebook().to(device)
        return self._codebook_cache[device]

    def quantize(
        self,
        scaled_blocked: torch.Tensor,
        generator: torch.Generator | None = None,
        *,
        block_scales: torch.Tensor | None = None,
        bias_correction: float | None = None,
        eps: float | None = None,
    ) -> torch.Tensor:
        cb = self._get_codebook(scaled_blocked.device)
        k = cb.numel()
        indices = torch.searchsorted(cb, scaled_blocked)
        left = (indices - 1).clamp_(min=0)
        right = indices.clamp_(max=k - 1)
        # Match TorchAO's 4-bit qmap rounding: values exactly halfway between
        # adjacent entries round up.
        pick_left = (scaled_blocked - cb[left]).abs() < (cb[right] - scaled_blocked).abs()
        codes = torch.where(pick_left, left, right)
        return codes.to(torch.uint8)

    def dequantize(self, blocked_codes: torch.Tensor) -> torch.Tensor:
        cb = self._get_codebook(blocked_codes.device)
        return cb[blocked_codes.long()].to(torch.float32)


class _StochasticCodebookQuantizationStrategy(_CodebookQuantizationStrategy):
    """Stochastic interpolation between adjacent codebook levels.

    For x in [a, b], samples b with probability (x - a) / (b - a). This makes
    the normalized dequantized value unbiased in expectation.
    """

    def quantize(
        self,
        scaled_blocked: torch.Tensor,
        generator: torch.Generator | None = None,
        *,
        block_scales: torch.Tensor | None = None,
        bias_correction: float | None = None,
        eps: float | None = None,
    ) -> torch.Tensor:
        cb = self._get_codebook(scaled_blocked.device)
        k = cb.numel()
        indices = torch.searchsorted(cb, scaled_blocked)
        left = (indices - 1).clamp_(min=0)
        right = indices.clamp_(max=k - 1)

        left_vals = cb[left]
        right_vals = cb[right]
        width = right_vals - left_vals
        prob_right = (scaled_blocked.float() - left_vals) / width.clamp_min(1e-12)
        prob_right = torch.where(width > 0, prob_right, torch.zeros_like(prob_right))
        prob_right = prob_right.clamp_(0.0, 1.0)

        draws = torch.rand(
            prob_right.shape,
            device=prob_right.device,
            dtype=prob_right.dtype,
            generator=generator,
        )
        codes = torch.where(draws < prob_right, right, left)
        return codes.to(torch.uint8)


class _UpdateUnbiasedStochasticCodebookQuantizationStrategy(_CodebookQuantizationStrategy):
    """Stochastic rounding in Adam preconditioner space.

    For second-moment value v = scale * x and adjacent codebook levels a <= x <= b,
    choose p so E[1 / (sqrt(scale * q / bias_correction) + eps)] equals the fp32
    preconditioner at x. This intentionally biases q high to remove Jensen
    inflation in the current coordinate update.
    """

    def quantize(
        self,
        scaled_blocked: torch.Tensor,
        generator: torch.Generator | None = None,
        *,
        block_scales: torch.Tensor | None = None,
        bias_correction: float | None = None,
        eps: float | None = None,
    ) -> torch.Tensor:
        cb = self._get_codebook(scaled_blocked.device)
        k = cb.numel()
        indices = torch.searchsorted(cb, scaled_blocked)
        left = (indices - 1).clamp_(min=0)
        right = indices.clamp_(max=k - 1)

        left_vals = cb[left]
        right_vals = cb[right]
        width = right_vals - left_vals

        if block_scales is None:
            block_scales = torch.ones(
                scaled_blocked.shape[0],
                device=scaled_blocked.device,
                dtype=torch.float32,
            )
        scales = block_scales.to(device=scaled_blocked.device, dtype=torch.float32).unsqueeze(1)
        bias = max(float(bias_correction) if bias_correction is not None else 1.0, 1e-12)
        adam_eps = float(eps) if eps is not None else 1e-8

        def preconditioner(levels: torch.Tensor) -> torch.Tensor:
            v_hat = (scales * levels.float() / bias).clamp_min(0.0)
            return 1.0 / (v_hat.sqrt() + adam_eps)

        f_left = preconditioner(left_vals)
        f_right = preconditioner(right_vals)
        f_x = preconditioner(scaled_blocked.float())
        denom = f_left - f_right
        # Compute the usually tiny left probability directly. Near the zero
        # code, f_left = 1/eps can be huge, so f_left - f_x loses precision.
        prob_left = (f_x - f_right) / denom.clamp_min(1e-30)
        prob_left = torch.where((width > 0) & (denom > 0), prob_left, torch.ones_like(prob_left))
        prob_left = prob_left.clamp_(0.0, 1.0)

        draws = torch.rand(
            prob_left.shape,
            device=prob_left.device,
            dtype=prob_left.dtype,
            generator=generator,
        )
        codes = torch.where(draws < prob_left, left, right)
        return codes.to(torch.uint8)


# --- Signed 4-bit codebooks for the AdamW first moment m1 ----------------------
#
# m1 is signed, so it needs a signed codebook. Keep this separate from the unsigned
# `dyn4`/`dyn4_nz` m2 schemes, whose 16 levels are all reserved for [0, 1].


class SignedDynamic4Strategy(_CodebookQuantizationStrategy):
    """Signed dynamic 4-bit for m1: 16 nonlinearly spaced levels including zero,
    with positive and negative levels dense near zero."""

    def _build_codebook(self) -> torch.Tensor:
        return _create_dynamic_map(signed=True, max_exponent_bits=3, total_bits=4)


class StochasticSignedDynamic4Strategy(_StochasticCodebookQuantizationStrategy):
    """Signed dynamic 4-bit for m1 with stochastic rounding."""

    def _build_codebook(self) -> torch.Tensor:
        return _create_dynamic_map(signed=True, max_exponent_bits=3, total_bits=4)


def _normal_float4_codebook() -> torch.Tensor:
    return torch.tensor(
        [
            -1.0000000,
            -0.6961928,
            -0.5250731,
            -0.3949175,
            -0.2844414,
            -0.1847734,
            -0.0910500,
            0.0000000,
            0.0795803,
            0.1609302,
            0.2461123,
            0.3379152,
            0.4407098,
            0.5626170,
            0.7229568,
            1.0000000,
        ],
        dtype=torch.float32,
    )


class NormalFloat4Strategy(_CodebookQuantizationStrategy):
    """NF4 signed 4-bit codebook for approximately normal first-moment values."""

    def _build_codebook(self) -> torch.Tensor:
        return _normal_float4_codebook()


class StochasticNormalFloat4Strategy(
    _StochasticCodebookQuantizationStrategy
):
    """NF4 codebook with value-unbiased adjacent-level stochastic rounding."""

    def _build_codebook(self) -> torch.Tensor:
        return _normal_float4_codebook()


# --- Unsigned 4-bit codebooks for the AdamW second moment m2 -------------------
#
# m2 >= 0, so after per-block absmax scaling it lives in [0, 1]. A *signed* 4-bit
# codebook (the old symmetric int4 / E2M1 fp4) would waste ~half its codes on the
# unused negative side (only 8 of 16 levels reachable). Open-source 4-bit optimizer
# state instead uses a dedicated UNSIGNED codebook for the second moment, giving all
# 16 levels to [0, 1]: a *linear* grid (Li 2023 / TorchAO 4-bit) or a *dynamic*
# (Dettmers/bitsandbytes-style) nonlinear grid. We mirror both. (Only correct for
# non-negative inputs -> use for m2 only, never for the signed m1.)
#
# The `_nz` variants exclude the zero level (the second-moment "zero-point problem":
# small m2 -> 0 collapses the preconditioner 1/(sqrt(m2)+eps) and inflates updates).
# For the linear grid this is TorchAO's distinct all-positive grid {1/16..1}; for the
# dynamic grid it is the zero level floored to the smallest positive code.


class UnsignedLinear4Strategy(_CodebookQuantizationStrategy):
    """Unsigned linear 4-bit for m2: 16 evenly spaced levels in [0, 1] INCLUDING
    zero (linspace(0,1,16) = {0, 1/15, ..., 1}). The *naive* unsigned grid -- small
    m2 can still round to the zero level."""

    def _build_codebook(self) -> torch.Tensor:
        return torch.linspace(0.0, 1.0, 16, dtype=torch.float32)


class UnsignedLinear4NoZeroStrategy(_CodebookQuantizationStrategy):
    """Unsigned linear 4-bit, zero-excluded (Li 2023 / TorchAO 4-bit second moment):
    16 levels linspace(0,1,17)[1:] = {1/16, ..., 1}. A distinct all-positive grid --
    NOT the include-zero grid with the zero code masked out."""

    def _build_codebook(self) -> torch.Tensor:
        return torch.linspace(0.0, 1.0, 17, dtype=torch.float32)[1:]


class StochasticUnsignedLinear4Strategy(_StochasticCodebookQuantizationStrategy):
    """Unsigned linear 4-bit with stochastic rounding for m2.

    Uses the same include-zero grid as ``lin4`` and samples adjacent levels so
    dequantized m2 is unbiased in expectation before any EDEN correction.
    """

    def _build_codebook(self) -> torch.Tensor:
        return torch.linspace(0.0, 1.0, 16, dtype=torch.float32)


class UpdateUnbiasedStochasticUnsignedLinear4Strategy(_UpdateUnbiasedStochasticCodebookQuantizationStrategy):
    """Unsigned linear 4-bit whose SR probability targets the Adam update."""

    def _build_codebook(self) -> torch.Tensor:
        return torch.linspace(0.0, 1.0, 16, dtype=torch.float32)


class UpdateUnbiasedStoreEdenUnsignedLinear4Strategy(_UpdateUnbiasedStochasticCodebookQuantizationStrategy):
    """Unsigned linear 4-bit update-SR with EDEN applied only to stored m2."""

    def _build_codebook(self) -> torch.Tensor:
        return torch.linspace(0.0, 1.0, 16, dtype=torch.float32)


class UnsignedDynamic4Strategy(_CodebookQuantizationStrategy):
    """Unsigned dynamic 4-bit (Dettmers / bitsandbytes-style) for m2: 16 non-negative,
    nonlinearly spaced levels (dense near 0, sparse near 1) INCLUDING zero."""

    def _build_codebook(self) -> torch.Tensor:
        return _create_dynamic_map(signed=False, max_exponent_bits=3, total_bits=4)


class StochasticUnsignedDynamic4Strategy(_StochasticCodebookQuantizationStrategy):
    """Unsigned dynamic 4-bit with stochastic rounding for m2."""

    def _build_codebook(self) -> torch.Tensor:
        return _create_dynamic_map(signed=False, max_exponent_bits=3, total_bits=4)


class UpdateUnbiasedStochasticUnsignedDynamic4Strategy(_UpdateUnbiasedStochasticCodebookQuantizationStrategy):
    """Unsigned dynamic 4-bit whose SR probability targets the Adam update."""

    def _build_codebook(self) -> torch.Tensor:
        return _create_dynamic_map(signed=False, max_exponent_bits=3, total_bits=4)


class UpdateUnbiasedStoreEdenUnsignedDynamic4Strategy(_UpdateUnbiasedStochasticCodebookQuantizationStrategy):
    """Unsigned dynamic 4-bit update-SR with EDEN applied only to stored m2."""

    def _build_codebook(self) -> torch.Tensor:
        return _create_dynamic_map(signed=False, max_exponent_bits=3, total_bits=4)


class UnsignedDynamic4NoZeroStrategy(UnsignedDynamic4Strategy):
    """Unsigned dynamic 4-bit with the zero code floored to the smallest positive
    level (zero-excluded by masking, mirroring the linear floor)."""

    ZERO_CODE = 0
    MIN_POSITIVE_CODE = 1

    def quantize(
        self,
        scaled_blocked: torch.Tensor,
        generator: torch.Generator | None = None,
        *,
        block_scales: torch.Tensor | None = None,
        bias_correction: float | None = None,
        eps: float | None = None,
    ) -> torch.Tensor:
        codes = super().quantize(
            scaled_blocked,
            generator=generator,
            block_scales=block_scales,
            bias_correction=bias_correction,
            eps=eps,
        )
        return codes.masked_fill(codes == self.ZERO_CODE, self.MIN_POSITIVE_CODE)


# ====================================================================
#
#      Stateless Functional API
#
# ====================================================================

_STRATEGIES: Dict[str, QuantizationStrategy] = {
    "linear": LinearQuantizationStrategy(),    # 8-bit symmetric int8
    "dynamic": DynamicQuantizationStrategy(),  # 8-bit signed dynamic codebook
    # 4-bit SIGNED first-moment (m1) codebook:
    "sdyn4": SignedDynamic4Strategy(),         # signed dynamic, incl. zero
    "sdyn4_sr": StochasticSignedDynamic4Strategy(), # signed dynamic, stochastic rounding
    "nf4": NormalFloat4Strategy(),             # signed NormalFloat4 (QLoRA) for m1
    "nf4_sr": StochasticNormalFloat4Strategy(),# NF4 codebook, stochastic rounding
    # 4-bit UNSIGNED second-moment (m2) codebooks (16 levels over [0, 1]):
    "lin4": UnsignedLinear4Strategy(),         # linear, incl. zero (naive)
    "lin4_sr": StochasticUnsignedLinear4Strategy(), # linear, stochastic rounding
    "lin4_upd_sr": UpdateUnbiasedStochasticUnsignedLinear4Strategy(), # linear, update-unbiased SR
    "lin4_upd_sr_store_eden": UpdateUnbiasedStoreEdenUnsignedLinear4Strategy(), # linear, update-SR read + EDEN store
    "lin4_upd_sr_fp32read": UpdateUnbiasedStochasticUnsignedLinear4Strategy(), # update-SR storage, fp32 current read
    "lin4_nz": UnsignedLinear4NoZeroStrategy(),# linear, zero-excluded (Li'23/TorchAO)
    "dyn4": UnsignedDynamic4Strategy(),        # dynamic (nonlinear), incl. zero (naive)
    "dyn4_sr": StochasticUnsignedDynamic4Strategy(), # dynamic, stochastic rounding
    "dyn4_upd_sr": UpdateUnbiasedStochasticUnsignedDynamic4Strategy(), # dynamic, update-unbiased SR
    "dyn4_upd_sr_store_eden": UpdateUnbiasedStoreEdenUnsignedDynamic4Strategy(), # dynamic, update-SR read + EDEN store
    "dyn4_upd_sr_fp32read": UpdateUnbiasedStochasticUnsignedDynamic4Strategy(), # update-SR storage, fp32 current read
    "dyn4_lookahead_sr": UpdateUnbiasedStochasticUnsignedDynamic4Strategy(), # fp32 read, next-step bias correction for storage SR
    "dyn4_upd_sr_fp32read_nextbc": UpdateUnbiasedStochasticUnsignedDynamic4Strategy(), # fp32 read, next-step bias correction for storage SR
    "dyn4_proxy_la_upd_sr": UnsignedDynamic4Strategy(), # dynamic, proxy lookahead update-SR storage
    "dyn4_vproxy_la_upd_sr": UnsignedDynamic4Strategy(), # dynamic, v_t-proxy lookahead update-SR storage
    "dyn4_nz": UnsignedDynamic4NoZeroStrategy(),# dynamic, zero floored to min-positive
}

_TORCHAO_SCALE_FLOOR_SCHEMES = {
    "sdyn4", "sdyn4_sr", "nf4", "nf4_sr",
    "lin4", "lin4_sr", "lin4_upd_sr", "lin4_upd_sr_store_eden", "lin4_upd_sr_fp32read", "lin4_nz",
    "dyn4", "dyn4_sr", "dyn4_upd_sr", "dyn4_upd_sr_store_eden", "dyn4_upd_sr_fp32read", "dyn4_lookahead_sr", "dyn4_upd_sr_fp32read_nextbc", "dyn4_la_upd_sr", "dyn4_proxy_la_upd_sr", "dyn4_vproxy_la_upd_sr", "dyn4_nz",
}


# --- Open-source-aligned block sizes -------------------------------------------
#
# Open-source optimizer-state quantizers use a *finer* block for 4-bit than for
# 8-bit: bitsandbytes (>= 0.44) and TorchAO use a 256-element block for 8-bit
# state, while TorchAO's 4-bit state (and lpmm's `2nd_moment_group_128`) use a
# 128-element block. We mirror that, keyed by the scheme's bit-width, so 4-bit
# state is not quantized with the coarser 8-bit block. (A coarser block lets one
# outlier set the absmax over more elements, rounding more small m2 values to
# zero -> worse preconditioner collapse, the exact 4-bit failure mode.)
_SCHEME_BITS: Dict[str, int] = {
    "linear": 8, "dynamic": 8,
    "sdyn4": 4, "sdyn4_sr": 4, "nf4": 4, "nf4_sr": 4,
    "lin4": 4, "lin4_sr": 4, "lin4_upd_sr": 4, "lin4_upd_sr_store_eden": 4, "lin4_upd_sr_fp32read": 4, "lin4_nz": 4,
    "dyn4": 4, "dyn4_sr": 4, "dyn4_upd_sr": 4, "dyn4_upd_sr_store_eden": 4, "dyn4_upd_sr_fp32read": 4, "dyn4_lookahead_sr": 4, "dyn4_upd_sr_fp32read_nextbc": 4, "dyn4_la_upd_sr": 4, "dyn4_proxy_la_upd_sr": 4, "dyn4_vproxy_la_upd_sr": 4, "dyn4_nz": 4,
}
DEFAULT_BLOCK_SIZE_BY_BITS: Dict[int, int] = {8: 256, 4: 128}


def is_4bit_scheme(mode: str) -> bool:
    """Return whether ``mode`` uses a 16-entry codebook."""
    return _SCHEME_BITS.get(mode) == 4


def _shape_numel(shape: torch.Size | tuple[int, ...] | None) -> int:
    if shape is None:
        raise ValueError("Quantized state is missing its logical shape")
    numel = 1
    for dimension in shape:
        numel *= int(dimension)
    return numel


def pack_4bit_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack two uint4 codebook indices into each uint8 storage element."""
    if codes.dtype != torch.uint8:
        raise TypeError(f"4-bit codes must use torch.uint8, found {codes.dtype}")
    flat = codes.reshape(-1)
    packed = flat[0::2].clone()
    high = flat[1::2]
    if high.numel():
        packed[: high.numel()].bitwise_or_(
            torch.bitwise_left_shift(high, 4)
        )
    return packed


def unpack_4bit_codes(codes: torch.Tensor, *, numel: int) -> torch.Tensor:
    """Unpack uint8 nibble storage into uint8 codebook indices."""
    if codes.dtype != torch.uint8:
        raise TypeError(f"Packed 4-bit codes must use torch.uint8, found {codes.dtype}")
    expected = (numel + 1) // 2
    if codes.numel() != expected:
        raise ValueError(
            f"Packed code length mismatch: expected {expected}, found {codes.numel()}"
        )
    flat = codes.reshape(-1)
    unpacked = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=flat.device)
    unpacked[0::2] = torch.bitwise_and(flat, 0x0F)
    unpacked[1::2] = torch.bitwise_right_shift(flat, 4)
    return unpacked[:numel]


def pack_codes_for_storage(
    codes: torch.Tensor,
    quant_state: "QuantState",
    *,
    mode: str,
) -> tuple[torch.Tensor, "QuantState"]:
    """Pack persistent 4-bit codes while leaving numerical APIs unpacked."""
    if not is_4bit_scheme(mode):
        quant_state.packed = False
        return codes, quant_state
    quant_state.packed = True
    return pack_4bit_codes(codes), quant_state


def codes_for_dequantization(
    codes: torch.Tensor,
    quant_state: "QuantState",
    *,
    mode: str,
) -> torch.Tensor:
    """Return logical-shape codes from packed or legacy unpacked state."""
    if not is_4bit_scheme(mode):
        return codes
    logical_numel = _shape_numel(quant_state.shape)
    packed_numel = (logical_numel + 1) // 2
    is_packed = bool(getattr(quant_state, "packed", False))
    if not is_packed and codes.numel() == logical_numel:
        return codes.reshape(quant_state.shape)
    if not is_packed and codes.numel() != packed_numel:
        raise ValueError(
            "4-bit code storage does not match either the logical or packed shape"
        )
    return unpack_4bit_codes(codes, numel=logical_numel).reshape(
        quant_state.shape
    )


def default_block_size(mode: str) -> int:
    """Open-source-aligned block size for a quant scheme (8-bit: 256, 4-bit: 128).

    'fp32' (unquantized) and unknown modes fall back to the 8-bit block size; the
    value is unused for fp32 since no quantization happens.
    """
    bits = _SCHEME_BITS.get(mode)
    if bits is None:
        return DEFAULT_BLOCK_SIZE_BY_BITS[8]
    return DEFAULT_BLOCK_SIZE_BY_BITS[bits]


def default_min_scale(mode: str) -> float:
    """Default per-block scale floor.

    Custom 4-bit optimizer-state schemes use TorchAO's 1e-12 floor so tiny
    blocks are not artificially inflated by the older 1e-8 clamp.
    """
    return 1e-12 if mode in _TORCHAO_SCALE_FLOOR_SCHEMES else 1e-8


def quantize(
    x: torch.Tensor,
    block_size: int = 2048,
    mode: Literal["linear", "dynamic", "sdyn4", "sdyn4_sr", "nf4", "nf4_sr", "lin4", "lin4_sr", "lin4_upd_sr", "lin4_upd_sr_store_eden", "lin4_upd_sr_fp32read", "lin4_nz", "dyn4", "dyn4_sr", "dyn4_upd_sr", "dyn4_upd_sr_store_eden", "dyn4_upd_sr_fp32read", "dyn4_lookahead_sr", "dyn4_upd_sr_fp32read_nextbc", "dyn4_proxy_la_upd_sr", "dyn4_vproxy_la_upd_sr", "dyn4_nz"] = "linear",
    use_bitsandbytes: bool = False,
    min_scale: float | None = None,
    eden_correction: bool = False,
    generator: torch.Generator | None = None,
    bias_correction: float | None = None,
    eps: float | None = None,
) -> Tuple[torch.Tensor, "QuantState"]:
    """
    Stateless functional API for block-wise quantization.

    When eden_correction=True, applies guarded EDEN block-scale calibration
    (Quartet II, arXiv:2601.22813). The per-block factor uses ||x||^2 divided
    by the guarded inner product <x, x_hat> and is baked into the block scale
    so dequantization needs no changes.
    """
    return _quantize_impl(
        x,
        block_size=block_size,
        mode=mode,
        use_bitsandbytes=use_bitsandbytes,
        min_scale=min_scale,
        eden_correction=eden_correction,
        generator=generator,
        bias_correction=bias_correction,
        eps=eps,
        return_dequantized=False,
    )


def quantize_with_dequantized(
    x: torch.Tensor,
    block_size: int = 2048,
    mode: Literal["linear", "dynamic", "sdyn4", "sdyn4_sr", "nf4", "nf4_sr", "lin4", "lin4_sr", "lin4_upd_sr", "lin4_upd_sr_store_eden", "lin4_upd_sr_fp32read", "lin4_nz", "dyn4", "dyn4_sr", "dyn4_upd_sr", "dyn4_upd_sr_store_eden", "dyn4_upd_sr_fp32read", "dyn4_lookahead_sr", "dyn4_upd_sr_fp32read_nextbc", "dyn4_proxy_la_upd_sr", "dyn4_vproxy_la_upd_sr", "dyn4_nz"] = "linear",
    use_bitsandbytes: bool = False,
    min_scale: float | None = None,
    eden_correction: bool = False,
    generator: torch.Generator | None = None,
    bias_correction: float | None = None,
    eps: float | None = None,
) -> Tuple[torch.Tensor, "QuantState", torch.Tensor]:
    """Quantize and return the matching dequantized tensor from the same sample.

    This is useful when the caller needs both the stored quantized state and the
    exact sampled value for the current update. It avoids re-blocking codes and
    doing a second codebook lookup through ``dequantize()``.
    """
    return _quantize_impl(
        x,
        block_size=block_size,
        mode=mode,
        use_bitsandbytes=use_bitsandbytes,
        min_scale=min_scale,
        eden_correction=eden_correction,
        generator=generator,
        bias_correction=bias_correction,
        eps=eps,
        return_dequantized=True,
    )


def quantize_dyn4_la_upd_sr(
    z_prev: torch.Tensor,
    grad: torch.Tensor,
    *,
    beta2: float,
    bias_correction: float,
    eps: float,
    block_size: int,
    generator: torch.Generator | None = None,
    min_scale: float | None = None,
    fresh_proxy: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, "QuantState", torch.Tensor]:
    """Oracle lookahead update-SR sample for a pending AdamW second moment.

    The pending previous state ``z_prev`` is quantized with the existing unsigned
    ``dyn4`` codebook, but the rounding probability is chosen after seeing
    ``grad`` so the next-step Adam preconditioner is unbiased:

        E[1 / (sqrt((beta2 * A + (1-beta2) * grad^2) / c2) + eps)]
        =
        1 / (sqrt((beta2 * z_prev + (1-beta2) * grad^2) / c2) + eps).

    This helper returns transient codes and the sampled dequantized ``A``. The
    oracle optimizer keeps the resulting current ``v_t`` in fp32, so these codes
    are for diagnostics only and are not needed for memory savings.
    """
    if z_prev.shape != grad.shape:
        raise ValueError(f"z_prev and grad must have the same shape, got {z_prev.shape} and {grad.shape}")
    if fresh_proxy is not None and fresh_proxy.shape != z_prev.shape:
        raise ValueError(
            f"fresh_proxy and z_prev must have the same shape, got {fresh_proxy.shape} and {z_prev.shape}"
        )
    if beta2 < 0.0 or beta2 >= 1.0:
        raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
    if bias_correction <= 0.0:
        raise ValueError(f"bias_correction must be positive, got {bias_correction}")
    if eps < 0.0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    if min_scale is None:
        min_scale = default_min_scale("dyn4_la_upd_sr")

    if z_prev.numel() == 0:
        codes = torch.empty_like(z_prev, dtype=torch.uint8)
        scales = torch.empty(0, dtype=torch.float32, device=z_prev.device)
        quant_state = QuantState(
            absmax=scales, shape=z_prev.shape, dtype=z_prev.dtype, blocksize=block_size
        )
        return codes, quant_state, torch.empty_like(z_prev)

    z_prev = z_prev.clamp_min(0)
    if fresh_proxy is not None:
        fresh_proxy = fresh_proxy.clamp_min(0)
    blocked_z, original_shape = _block(z_prev, block_size)
    blocked_g, _ = _block(grad, block_size)
    blocked_fresh_proxy = None
    if fresh_proxy is not None:
        blocked_fresh_proxy, _ = _block(fresh_proxy, block_size)
    scales = blocked_z.abs().max(dim=1).values.to(torch.float32)
    scales.clamp_(min=min_scale)
    scaled = (blocked_z / scales.unsqueeze(1)).clamp_(0.0, 1.0)

    strategy = _STRATEGIES["dyn4"]
    assert isinstance(strategy, _CodebookQuantizationStrategy)
    cb = strategy._get_codebook(z_prev.device)
    k = cb.numel()
    indices = torch.searchsorted(cb, scaled)
    left = (indices - 1).clamp_(min=0)
    right = indices.clamp_(max=k - 1)

    cb64 = cb.to(torch.float64)
    scales64 = scales.to(torch.float64).unsqueeze(1)
    z64 = blocked_z.to(torch.float64)
    if blocked_fresh_proxy is None:
        fresh = (1.0 - float(beta2)) * blocked_g.to(torch.float64).square()
    else:
        fresh = blocked_fresh_proxy.to(torch.float64)
    bias = max(float(bias_correction), 1e-12)
    adam_eps = float(eps)

    q_lo = scales64 * cb64[left.long()]
    q_hi = scales64 * cb64[right.long()]
    next_lo = (float(beta2) * q_lo + fresh).clamp_min(0.0)
    next_ref = (float(beta2) * z64 + fresh).clamp_min(0.0)
    next_hi = (float(beta2) * q_hi + fresh).clamp_min(0.0)

    p_lo = 1.0 / (next_lo.div(bias).sqrt() + adam_eps)
    p_ref = 1.0 / (next_ref.div(bias).sqrt() + adam_eps)
    p_hi = 1.0 / (next_hi.div(bias).sqrt() + adam_eps)
    denom = p_lo - p_hi
    width = q_hi - q_lo

    raw_prob_lo = (p_ref - p_hi) / denom.clamp_min(1e-300)
    store_prob_hi = (z64 - q_lo) / width.clamp_min(1e-300)
    fallback_prob_lo = 1.0 - store_prob_hi.clamp(0.0, 1.0)

    degenerate = width <= 0.0
    scale_for_flat = torch.maximum(p_lo.abs(), p_hi.abs()).clamp_min(1.0)
    flat = (~degenerate) & (denom.abs() <= 1e-12 * scale_for_flat)
    prob_lo = torch.where(flat, fallback_prob_lo, raw_prob_lo)
    prob_lo = torch.where(degenerate, torch.ones_like(prob_lo), prob_lo)
    prob_lo = prob_lo.clamp(0.0, 1.0)

    draws = torch.rand(
        prob_lo.shape,
        device=prob_lo.device,
        dtype=prob_lo.dtype,
        generator=generator,
    )
    codes_blocked = torch.where(draws < prob_lo, left, right).to(torch.uint8)
    selected = torch.where(draws < prob_lo, q_lo, q_hi)

    codes = _unblock(codes_blocked, original_shape)
    quant_state = QuantState(
        absmax=scales, shape=original_shape, dtype=z_prev.dtype, blocksize=block_size
    )
    sampled = _unblock(selected, original_shape).to(z_prev.dtype)
    return codes, quant_state, sampled


def _quantize_impl(
    x: torch.Tensor,
    block_size: int,
    mode: str,
    use_bitsandbytes: bool,
    min_scale: float | None,
    eden_correction: bool,
    generator: torch.Generator | None,
    bias_correction: float | None,
    eps: float | None,
    return_dequantized: bool,
) -> Tuple[torch.Tensor, "QuantState"] | Tuple[torch.Tensor, "QuantState", torch.Tensor]:
    if use_bitsandbytes:
        if return_dequantized:
            raise ValueError("quantize_with_dequantized does not support use_bitsandbytes=True")
        if not HAS_BITSANDBYTES:
            raise ImportError("bitsandbytes is not installed.")
        if eden_correction:
            raise ValueError(
                "eden_correction is not supported with use_bitsandbytes=True. "
                "Use the custom 'linear' or 'dynamic' quantization instead."
            )
        return F.quantize_blockwise(x, blocksize=block_size)

    if mode not in _STRATEGIES:
        raise ValueError(f"Unsupported quantization mode: {mode}")
    if min_scale is None:
        min_scale = default_min_scale(mode)

    if x.numel() == 0:
        codes = torch.empty_like(x, dtype=torch.int8)
        scales = torch.empty(0, dtype=torch.float32, device=x.device)
        quant_state = QuantState(
            absmax=scales, shape=x.shape, dtype=x.dtype, blocksize=block_size
        )
        if return_dequantized:
            return codes, quant_state, torch.empty_like(x)
        return codes, quant_state

    blocked, original_shape = _block(x, block_size)
    scales = blocked.abs().max(dim=1).values.to(torch.float32)
    scales.clamp_(min=min_scale)
    scaled_blocked = blocked / scales.unsqueeze(1)

    strategy = _STRATEGIES[mode]
    codes_blocked = strategy.quantize(
        scaled_blocked,
        generator=generator,
        block_scales=scales,
        bias_correction=bias_correction,
        eps=eps,
    )

    dequant_blocked = None
    if eden_correction or return_dequantized:
        dequant_blocked = strategy.dequantize(codes_blocked)

    if eden_correction:
        assert dequant_blocked is not None
        numerator = (scaled_blocked.float() * scaled_blocked.float()).sum(dim=1)
        denominator = (scaled_blocked.float() * dequant_blocked.float()).sum(dim=1)
        correction = numerator / denominator.clamp(min=1e-12)
        scales = scales * correction

    codes = _unblock(codes_blocked, original_shape)
    quant_state = QuantState(
        absmax=scales, shape=original_shape, dtype=x.dtype, blocksize=block_size
    )
    logging.debug("Quantized %s to %s with %s quantization", x.shape, codes.shape, mode)
    if return_dequantized:
        assert dequant_blocked is not None
        dequantized = _unblock(dequant_blocked * scales.unsqueeze(1), original_shape).to(x.dtype)
        return codes, quant_state, dequantized
    return codes, quant_state



def dequantize(
    codes: torch.Tensor,
    quant_state: "QuantState",
    mode: Literal["linear", "dynamic", "sdyn4", "sdyn4_sr", "nf4", "nf4_sr", "lin4", "lin4_sr", "lin4_upd_sr", "lin4_upd_sr_store_eden", "lin4_upd_sr_fp32read", "lin4_nz", "dyn4", "dyn4_sr", "dyn4_upd_sr", "dyn4_upd_sr_store_eden", "dyn4_upd_sr_fp32read", "dyn4_lookahead_sr", "dyn4_upd_sr_fp32read_nextbc", "dyn4_proxy_la_upd_sr", "dyn4_vproxy_la_upd_sr", "dyn4_nz"] = "linear",
    use_bitsandbytes: bool = False,
) -> torch.Tensor:
    """
    Stateless functional API for block-wise dequantization.
    """
    if use_bitsandbytes:
        if not HAS_BITSANDBYTES:
            raise ImportError("bitsandbytes is not installed.")
        return F.dequantize_blockwise(codes, quant_state)

    if mode not in _STRATEGIES:
        raise ValueError(f"Unsupported dequantization mode: {mode}")

    codes = codes_for_dequantization(codes, quant_state, mode=mode)
    if codes.numel() == 0:
        return torch.empty_like(codes, dtype=quant_state.dtype)

    blocked_codes, _ = _block(codes, quant_state.blocksize)

    strategy = _STRATEGIES[mode]
    decoded_blocked = strategy.dequantize(blocked_codes)

    decoded_scaled = decoded_blocked * quant_state.absmax.unsqueeze(1)
    unblocked = _unblock(decoded_scaled, quant_state.shape)
    logging.debug("Dequantized %s to %s with %s dequantization", codes.shape, unblocked.shape, mode)
    return unblocked.to(quant_state.dtype)


def apply_eden_correction_to_quant_state(
    x: torch.Tensor,
    codes: torch.Tensor,
    quant_state: "QuantState",
    mode: str,
) -> "QuantState":
    """Apply EDEN's block-scale correction to already sampled quantization codes."""
    if mode not in _STRATEGIES:
        raise ValueError(f"Unsupported quantization mode: {mode}")
    if x.numel() == 0:
        return quant_state

    codes = codes_for_dequantization(codes, quant_state, mode=mode)
    strategy = _STRATEGIES[mode]
    blocked, _ = _block(x, quant_state.blocksize)
    blocked_codes, _ = _block(codes, quant_state.blocksize)
    scales = quant_state.absmax.to(device=x.device, dtype=torch.float32)
    scaled = blocked / scales.unsqueeze(1)
    dequant_blocked = strategy.dequantize(blocked_codes)
    numerator = (scaled.float() * scaled.float()).sum(dim=1)
    denominator = (scaled.float() * dequant_blocked.float()).sum(dim=1)
    correction = numerator / denominator.clamp(min=1e-12)
    return QuantState(
        absmax=scales * correction,
        shape=quant_state.shape,
        dtype=quant_state.dtype,
        blocksize=quant_state.blocksize,
        packed=bool(getattr(quant_state, "packed", False)),
    )


# ====================================================================
#
#      Diagnostics (for 4-bit m2 failure-mode analysis)
#
# ====================================================================


def _strided_sample(t: torch.Tensor, n: int = 1_000_000) -> torch.Tensor:
    """Deterministic sub-sample of a flattened tensor for cheap quantiles
    (torch.quantile rejects tensors larger than ~16M elements)."""
    t = t.flatten()
    if t.numel() > n:
        t = t[:: t.numel() // n]
    return t


def m2_quant_diagnostics(
    v: torch.Tensor,
    block_size: int,
    mode: str,
    eps: float,
    eden: bool = False,
    bias_correction: float = 1.0,
) -> Dict[str, float]:
    """Diagnose what quantizing a second-moment tensor `v` to `mode` does to the
    AdamW preconditioner `1 / (sqrt(v) + eps)`.

    Mirrors `quantize()` exactly (block absmax + optional EDEN correction), then
    reports the three failure-mode signals from the replication plan:
      - fraction of dequantized m2 that collapses to ~0 (zeroing),
      - preconditioner inflation ratio (sqrt(v)+eps)/(sqrt(qhat)+eps),
      - EDEN per-block correction factor c_b distribution.
    """
    if v.numel() == 0 or mode not in _STRATEGIES:
        return {}
    blocked, shape = _block(v, block_size)
    scales = blocked.abs().max(dim=1).values.to(torch.float32).clamp_(min=default_min_scale(mode))
    scaled = blocked / scales.unsqueeze(1)
    strategy = _STRATEGIES[mode]
    generator = None
    if mode.endswith("_sr") or "_sr_" in mode:
        generator = torch.Generator(device=scaled.device)
        generator.manual_seed(0)
    codes = strategy.quantize(
        scaled,
        generator=generator,
        block_scales=scales,
        bias_correction=bias_correction,
        eps=eps,
    )
    deq = strategy.dequantize(codes).float()  # normalized, pre-correction
    if eden:
        num = (scaled.float() * scaled.float()).sum(dim=1)
        den = (scaled.float() * deq).sum(dim=1)
        c_b = num / den.clamp(min=1e-12)
    else:
        c_b = torch.ones_like(scales)
    qhat = _unblock(deq * (scales * c_b).unsqueeze(1), shape).float()
    vf = v.float()

    bias = max(float(bias_correction), 1e-12)
    pre = (vf.div(bias).sqrt() + eps) / (qhat.div(bias).sqrt() + eps)
    pre_s = _strided_sample(pre)
    pre_q = torch.quantile(pre_s, torch.tensor([0.5, 0.99], device=pre.device, dtype=pre_s.dtype))
    return {
        "numel": int(v.numel()),
        "frac_qv_eq_zero": (qhat == 0).float().mean().item(),
        "frac_qv_le_eps": (qhat <= eps).float().mean().item(),
        "pre_ratio_p50": pre_q[0].item(),
        "pre_ratio_p99": pre_q[1].item(),
        "pre_ratio_max": pre.max().item(),
        "frac_pre_gt_10": (pre > 10).float().mean().item(),
        "frac_pre_gt_100": (pre > 100).float().mean().item(),
        "c_b_mean": c_b.mean().item(),
        "c_b_p99": torch.quantile(c_b.float(), 0.99).item(),
        "c_b_max": c_b.max().item(),
        "frac_cb_off_10pct": (c_b.sub(1).abs() > 0.1).float().mean().item(),
    }