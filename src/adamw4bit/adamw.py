import math
from collections.abc import Mapping
from typing import Iterable, Iterator

import torch
from adamw4bit.quantization import (
    QuantState,
    apply_eden_correction_to_quant_state,
    default_block_size,
    dequantize,
    pack_codes_for_storage,
    quantize,
    quantize_dyn4_la_upd_sr,
    quantize_with_dequantized,
)
from torch.optim.optimizer import Optimizer


_TORCHAO_MIN_QUANT_NUMEL = 4096


def _block_rms_clip(u: torch.Tensor, block_size: int, c: float) -> torch.Tensor:
    """Direction-preserving update clip (the gradient-clipping analog).

    Rescale each ``block_size``-element block of the flattened update ``u`` so its
    RMS is at most ``c``, leaving the block's *direction* unchanged. This contrasts
    with the element-wise clamp (``update_sign_clip``), which caps each coordinate
    independently and so changes direction (collapsing toward signSGD when many
    coordinates saturate). The block matches the m2 quantizer block — the unit over
    which a shared absmax can collapse small values to zero.
    """
    shape = u.shape
    flat = u.reshape(-1)
    n = flat.numel()
    n_blocks = (n + block_size - 1) // block_size
    pad = n_blocks * block_size - n
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(n_blocks, block_size)
    rms = blocks.norm(dim=1).div_(math.sqrt(block_size))
    scale = (c / rms.clamp_min(1e-12)).clamp_(max=1.0)  # min(1, c / rms), direction kept
    blocks = blocks * scale.unsqueeze(1)
    return blocks.reshape(-1)[:n].reshape(shape)


class QuantizedAdamW(Optimizer):
    r"""
    AdamW with configurable 4-bit, 8-bit, or FP32 moment storage.

    Args:
        params: Iterator of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.95))
        eps: Term added to denominator to improve numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 1e-1)
        block_size: Elements per block for scaling. None (default) auto-selects per
            moment from its scheme's bit-width to match open-source optimizers
            (8-bit -> 256, 4-bit -> 128; see quantization_utils.default_block_size).
            An explicit int forces a single block size for both moments.
        quant_scheme: Default scheme for both moments - 8-bit "linear"/"dynamic", the signed
            4-bit m1 codebooks "sdyn4"/"sdyn4_sr"/"nf4"/"nf4_sr", or the unsigned 4-bit m2 codebooks
            "lin4"/"lin4_sr"/"lin4_upd_sr"/"lin4_upd_sr_store_eden"/
            "lin4_upd_sr_fp32read"/"lin4_nz" (linear) and
            "dyn4"/"dyn4_sr"/"dyn4_upd_sr"/"dyn4_upd_sr_store_eden"/
            "dyn4_upd_sr_fp32read"/"dyn4_lookahead_sr"/
            "dyn4_upd_sr_fp32read_nextbc"/
            "dyn4_la_upd_sr"/"dyn4_proxy_la_upd_sr"/"dyn4_vproxy_la_upd_sr"/"dyn4_nz"
            (dynamic)
            (default: "linear")
        use_eden_m2: Apply guarded EDEN block-scale calibration to second moment quantization (default: False)
        m1_quant_scheme: Override scheme for the first moment m1. "fp32" keeps m1 unquantized.
            Defaults to quant_scheme.
        m2_quant_scheme: Override scheme for the second moment m2. "fp32" keeps m2 unquantized.
            Defaults to quant_scheme. (Used for isolation studies, e.g. m1=fp32, m2=lin4.)
        quant_rng_seed: Optional base seed for independent, checkpointed m1
            and m2 quantization generators. None preserves the legacy shared
            global RNG stream.
        telemetry: Optional callable telemetry(param, m2_fp32, mode, block_size, eps, eden, step, beta2)
            invoked each step on the post-update m2 (only when m2 is quantized). For diagnostics.
        Quantized moments follow TorchAO optimizer-state eligibility: tensors with fewer than
            4096 elements or a size not divisible by the moment block size keep fp32 state.
        update_sign_clip: If set, clamp the per-coordinate preconditioned update
            u = m̂1 / (√m̂2 + ε) element-wise to [-c, c] before applying it. Bounds the damage when a
            quantized m2 collapses to ~0 in the denominator, without changing the quantizer, but it
            caps each coordinate independently and so changes direction (≈ c·signSGD when many
            coordinates saturate). None disables it. (Formerly `update_clip`.)
        update_norm_clip: If set, rescale each m2-quantization block of the preconditioned update so
            its RMS <= c, preserving the block's direction (the gradient-clipping analog of
            update_sign_clip). None disables it. Mutually exclusive with update_sign_clip.
        record_update_direction_every: If set, keep the post-clip update direction for diagnostic
            steps outside the optimizer state_dict so callbacks can compare the update that actually
            moved the iterates.
        record_preconditioner_every: If set, keep the actual read-side preconditioner
            ``1 / (sqrt(m2hat) + eps)`` used by the update for moment-history snapshots.
        record_preconditioner_steps: Optional explicit optimizer steps at which to keep that
            preconditioner, used to include non-interval final snapshots without recording every step.
            For sign-clipped rows, the same steps also keep ``effective_preconditioner``, the
            multiplier after the element-wise sign clip has been applied.
    """

    _VALID_SCHEMES = (
        "linear", "dynamic",
        "sdyn4", "sdyn4_sr", "nf4", "nf4_sr",
        "lin4", "lin4_sr", "lin4_upd_sr", "lin4_upd_sr_store_eden", "lin4_upd_sr_fp32read", "lin4_nz",
        "dyn4", "dyn4_sr", "dyn4_upd_sr", "dyn4_upd_sr_store_eden", "dyn4_upd_sr_fp32read", "dyn4_lookahead_sr", "dyn4_upd_sr_fp32read_nextbc", "dyn4_la_upd_sr", "dyn4_proxy_la_upd_sr", "dyn4_vproxy_la_upd_sr", "dyn4_nz",
        "fp32",
    )
    _UPDATE_UNBIASED_SR_SCHEMES = {"lin4_upd_sr", "dyn4_upd_sr"}
    _UPDATE_SR_STORE_EDEN_SCHEMES = {"lin4_upd_sr_store_eden", "dyn4_upd_sr_store_eden"}
    _UPDATE_SR_FP32_READ_SCHEMES = {"lin4_upd_sr_fp32read", "dyn4_upd_sr_fp32read"}
    # `dyn4_upd_sr_fp32read_nextbc` is retained as the historical ablation alias.
    _UPDATE_SR_FP32_READ_NEXT_BIAS_SCHEMES = {
        "dyn4_lookahead_sr",
        "dyn4_upd_sr_fp32read_nextbc",
    }
    _ORACLE_LOOKAHEAD_UPDATE_SR_SCHEMES = {"dyn4_la_upd_sr"}
    _PROXY_LOOKAHEAD_UPDATE_SR_SCHEMES = {"dyn4_proxy_la_upd_sr"}
    _VPROXY_LOOKAHEAD_UPDATE_SR_SCHEMES = {"dyn4_vproxy_la_upd_sr"}
    _QUANT_RNG_STATE_KEY = "_quant_rng_state"
    _QUANT_RNG_OFFSETS = {
        "m1": 0x1A2B3C4D,
        "m2": 0x5E6F7789,
    }

    def __init__(self,
                 params: Iterator[torch.nn.Parameter],
                 lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.95),
                 eps: float = 1e-8,
                 weight_decay: float = 1e-1,
                 block_size: int | None = None,
                 quant_scheme: str = "linear",
                 use_eden_m2: bool = False,
                 m1_quant_scheme: str | None = None,
                 m2_quant_scheme: str | None = None,
                 quant_rng_seed: int | None = None,
                 telemetry=None,
                 update_sign_clip: float | None = None,
                 update_norm_clip: float | None = None,
                 record_update_direction_every: int | None = None,
                 record_preconditioner_every: int | None = None,
                 record_preconditioner_steps: tuple[int, ...] | list[int] | set[int] | None = None):


        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if quant_rng_seed is not None and (
            isinstance(quant_rng_seed, bool)
            or not isinstance(quant_rng_seed, int)
            or quant_rng_seed < 0
        ):
            raise ValueError(
                "quant_rng_seed must be a non-negative integer or None, "
                f"got {quant_rng_seed!r}"
            )
        if update_sign_clip is not None and not update_sign_clip > 0.0:
            raise ValueError(f"update_sign_clip must be positive or None, got {update_sign_clip}")
        if update_norm_clip is not None and not update_norm_clip > 0.0:
            raise ValueError(f"update_norm_clip must be positive or None, got {update_norm_clip}")
        if update_sign_clip is not None and update_norm_clip is not None:
            raise ValueError("Set at most one of update_sign_clip / update_norm_clip "
                             "(they are alternative update-clip mechanisms).")
        if record_update_direction_every is not None and record_update_direction_every <= 0:
            raise ValueError(
                "record_update_direction_every must be positive or None, "
                f"got {record_update_direction_every}"
            )
        if record_preconditioner_every is not None and record_preconditioner_every <= 0:
            raise ValueError(
                "record_preconditioner_every must be positive or None, "
                f"got {record_preconditioner_every}"
            )
        if record_preconditioner_steps is not None:
            record_preconditioner_steps = tuple(sorted(set(int(step) for step in record_preconditioner_steps)))
            if any(step <= 0 for step in record_preconditioner_steps):
                raise ValueError("record_preconditioner_steps must contain positive optimizer steps")

        m1_scheme = m1_quant_scheme or quant_scheme
        m2_scheme = m2_quant_scheme or quant_scheme
        for name, s in (("quant_scheme", quant_scheme), ("m1_quant_scheme", m1_scheme), ("m2_quant_scheme", m2_scheme)):
            if s not in self._VALID_SCHEMES:
                raise ValueError(f"{name} must be one of {self._VALID_SCHEMES}, got {s!r}")
        if use_eden_m2 and m2_scheme in {"lin4_sr", "dyn4_sr"}:
            raise ValueError("Vanilla stochastic-rounding m2 schemes should not be combined with use_eden_m2")
        if use_eden_m2 and m2_scheme in self._UPDATE_SR_STORE_EDEN_SCHEMES:
            raise ValueError("Do not set use_eden_m2 with *_upd_sr_store_eden; EDEN is built into the store path")
        if m2_scheme in (
            self._ORACLE_LOOKAHEAD_UPDATE_SR_SCHEMES
            | self._PROXY_LOOKAHEAD_UPDATE_SR_SCHEMES
            | self._VPROXY_LOOKAHEAD_UPDATE_SR_SCHEMES
        ):
            if m1_scheme not in {"fp32", "nf4"}:
                raise ValueError("lookahead update-SR currently supports m1_quant_scheme='fp32' or 'nf4'")
            if use_eden_m2:
                raise ValueError("lookahead update-SR should not be combined with use_eden_m2")
            if update_sign_clip is not None or update_norm_clip is not None:
                raise ValueError("lookahead update-SR should not be combined with update clipping")

        self.telemetry = telemetry
        self._m1_quant_scheme_overrides: dict[int, str] = {}
        self._quant_rng_seed = quant_rng_seed
        self._quant_generators: dict[
            str,
            dict[str, torch.Generator],
        ] = {"m1": {}, "m2": {}}
        self._pending_quant_rng_states: dict[
            str,
            dict[str, torch.Tensor],
        ] = {"m1": {}, "m2": {}}
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        block_size=block_size, quant_scheme=quant_scheme,
                        use_eden_m2=use_eden_m2,
                        m1_quant_scheme=m1_scheme, m2_quant_scheme=m2_scheme,
                        update_sign_clip=update_sign_clip,
                        update_norm_clip=update_norm_clip,
                        record_update_direction_every=record_update_direction_every,
                        record_preconditioner_every=record_preconditioner_every,
                        record_preconditioner_steps=record_preconditioner_steps)
        super().__init__(params, defaults)
        self._last_update_directions: dict[torch.Tensor, tuple[int, torch.Tensor]] = {}
        self._last_preconditioners: dict[torch.Tensor, tuple[int, torch.Tensor]] = {}
        self._last_effective_preconditioners: dict[torch.Tensor, tuple[int, torch.Tensor]] = {}

    def _m1_scheme_for(
        self,
        parameter: torch.Tensor,
        group_scheme: str,
    ) -> str:
        return self._m1_quant_scheme_overrides.get(
            id(parameter),
            group_scheme,
        )

    def set_m1_quant_scheme_for_parameters(
        self,
        parameters: Iterable[torch.nn.Parameter],
        scheme: str,
    ) -> None:
        """Retarget future m1 writes for existing optimizer parameters."""
        if scheme not in {"nf4", "nf4_sr"}:
            raise ValueError(
                "runtime m1 retargeting supports only nf4 and nf4_sr"
            )
        optimizer_parameter_ids = {
            id(parameter)
            for group in self.param_groups
            for parameter in group["params"]
        }
        selected = list(parameters)
        if not selected:
            raise ValueError(
                "runtime m1 retargeting requires parameters"
            )
        if any(
            id(parameter) not in optimizer_parameter_ids
            for parameter in selected
        ):
            raise ValueError(
                "runtime m1 retargeting received parameters outside optimizer"
            )
        for parameter in selected:
            self._m1_quant_scheme_overrides[id(parameter)] = scheme

    def _quant_generator(
        self,
        moment: str,
        device: torch.device,
    ) -> torch.Generator | None:
        if self._quant_rng_seed is None:
            return None
        if moment not in self._QUANT_RNG_OFFSETS:
            raise ValueError(
                f"Unsupported quantization RNG stream: {moment!r}"
            )
        device_key = str(device)
        generators = self._quant_generators[moment]
        if device_key in generators:
            return generators[device_key]

        generator = torch.Generator(device=device)
        max_seed = (1 << 63) - 1
        generator.manual_seed(
            (
                self._quant_rng_seed
                + self._QUANT_RNG_OFFSETS[moment]
            )
            % max_seed
        )
        pending = self._pending_quant_rng_states[moment]
        state = pending.pop(device_key, None)
        if state is None:
            compatible_keys = [
                key
                for key in pending
                if key.split(":", maxsplit=1)[0] == device.type
            ]
            if len(compatible_keys) == 1:
                state = pending.pop(compatible_keys[0])
            elif pending:
                raise ValueError(
                    "Unable to restore quantization RNG state for "
                    f"{moment} on {device}; saved devices={sorted(pending)}"
                )
        if state is not None:
            generator.set_state(state.detach().cpu())
        generators[device_key] = generator
        return generator

    def state_dict(self):
        result = super().state_dict()
        # Keep checkpoints compatible with torch.load(weights_only=True) without
        # replacing the runtime QuantState objects or copying their tensors.
        result["state"] = {
            index: {
                key: vars(value).copy() if isinstance(value, QuantState) else value
                for key, value in parameter_state.items()
            }
            for index, parameter_state in result["state"].items()
        }
        if self._quant_rng_seed is None:
            return result
        states = {
            moment: {
                device: state.detach().cpu().clone()
                for device, state in self._pending_quant_rng_states[
                    moment
                ].items()
            }
            for moment in self._QUANT_RNG_OFFSETS
        }
        for moment, generators in self._quant_generators.items():
            states[moment].update(
                {
                    device: (
                        generator.get_state().detach().cpu().clone()
                    )
                    for device, generator in generators.items()
                }
            )
        result[self._QUANT_RNG_STATE_KEY] = {
            "version": 1,
            "seed": self._quant_rng_seed,
            "states": states,
        }
        return result

    def load_state_dict(self, state_dict):
        """Restore FP32 moments and integer codes on each parameter's device."""
        rng_payload = state_dict.get(self._QUANT_RNG_STATE_KEY)
        optimizer_state = {
            key: value
            for key, value in state_dict.items()
            if key != self._QUANT_RNG_STATE_KEY
        }

        def capture_state(_optimizer, loaded_state):
            nonlocal optimizer_state
            optimizer_state = loaded_state

        def restore_moments(_optimizer):
            for group, saved_group in zip(self.param_groups, optimizer_state["param_groups"]):
                for parameter, saved_index in zip(group["params"], saved_group["params"]):
                    original = optimizer_state["state"].get(saved_index, {})
                    if not original:
                        continue
                    state = self.state[parameter]
                    # Use original tensors: converting PyTorch's downcast copies
                    # back to FP32 would retain the lost checkpoint precision.
                    for key in (
                        "m1", "m2", "lookahead_pending_exp_avg_sq",
                        "m2_read_preconditioner",
                    ):
                        if key in original:
                            state[key] = original[key].to(
                                device=parameter.device, dtype=torch.float32,
                            )
                    for key in ("m1_quant_state", "m2_quant_state"):
                        if key in original:
                            value = original[key]
                            # Accept both portable dictionaries and legacy objects.
                            fields = dict(value) if isinstance(value, Mapping) else vars(value).copy()
                            fields["absmax"] = fields["absmax"].to(
                                device=parameter.device, dtype=torch.float32,
                            )
                            state[key] = QuantState(**fields)
                    schemes = {
                        "m1": self._m1_scheme_for(parameter, group["m1_quant_scheme"]),
                        "m2": group["m2_quant_scheme"],
                    }
                    for key, scheme in schemes.items():
                        code_key = f"{key}_code"
                        if code_key in original:
                            code_dtype = torch.int8 if scheme == "linear" else torch.uint8
                            state[code_key] = original[code_key].to(
                                device=parameter.device, dtype=code_dtype,
                            )

        # Capture after caller pre-hooks, then repair before caller post-hooks.
        capture_hook = self.register_load_state_dict_pre_hook(capture_state)
        restore_hook = self.register_load_state_dict_post_hook(restore_moments, prepend=True)
        try:
            result = super().load_state_dict(optimizer_state)
        finally:
            capture_hook.remove()
            restore_hook.remove()
        self._quant_generators = {"m1": {}, "m2": {}}
        self._pending_quant_rng_states = {"m1": {}, "m2": {}}
        if rng_payload is not None:
            if not isinstance(rng_payload, Mapping):
                raise ValueError(
                    "Invalid quantization RNG checkpoint payload"
                )
            if rng_payload.get("version") != 1:
                raise ValueError(
                    "Unsupported quantization RNG checkpoint version: "
                    f"{rng_payload.get('version')!r}"
                )
            if self._quant_rng_seed is None:
                raise ValueError(
                    "Checkpoint contains separated quantization RNG "
                    "state, but quant_rng_seed is disabled"
                )
            saved_seed = rng_payload.get("seed")
            if saved_seed != self._quant_rng_seed:
                raise ValueError(
                    "Quantization RNG seed mismatch: "
                    f"checkpoint={saved_seed!r}, "
                    f"configured={self._quant_rng_seed!r}"
                )
            saved_states = rng_payload.get("states")
            if not isinstance(saved_states, Mapping):
                raise ValueError(
                    "Quantization RNG checkpoint states are missing"
                )
            for moment in self._QUANT_RNG_OFFSETS:
                moment_states = saved_states.get(moment, {})
                if not isinstance(moment_states, Mapping):
                    raise ValueError(
                        f"Invalid quantization RNG states for {moment}"
                    )
                for device, state in moment_states.items():
                    if (
                        not isinstance(device, str)
                        or not torch.is_tensor(state)
                    ):
                        raise ValueError(
                            "Invalid quantization RNG state entry for "
                            f"{moment}"
                        )
                    self._pending_quant_rng_states[moment][device] = (
                        state.detach().cpu().clone()
                    )
        return result

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            block_size = group["block_size"]
            use_eden_m2 = group["use_eden_m2"]
            m1_scheme = group["m1_quant_scheme"]
            m2_scheme = group["m2_quant_scheme"]
            update_sign_clip = group["update_sign_clip"]
            update_norm_clip = group["update_norm_clip"]
            record_update_direction_every = group["record_update_direction_every"]
            record_preconditioner_every = group["record_preconditioner_every"]
            record_preconditioner_steps = group["record_preconditioner_steps"]

            # Resolve m1 per parameter because the runtime schedule can switch
            # selected parameters between NF4 RTN and NF4-SR.
            m2_bs = block_size if block_size is not None else default_block_size(m2_scheme)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Optimizer math and any TorchAO-ineligible fallback state stay
                # FP32 even if a caller supplies lower-precision parameters.
                g = p.grad.to(dtype=torch.float32)
                st = self.state[p]
                m1_generator = self._quant_generator("m1", g.device)
                m2_generator = self._quant_generator("m2", g.device)
                m1_parameter_scheme = self._m1_scheme_for(p, m1_scheme)
                m1_parameter_bs = (
                    block_size
                    if block_size is not None
                    else default_block_size(m1_parameter_scheme)
                )
                m1_effective_scheme = self._effective_scheme(
                    p,
                    m1_parameter_scheme,
                    m1_parameter_bs,
                )
                m2_effective_scheme = self._effective_scheme(p, m2_scheme, m2_bs)

                # State initialization ("fp32" or TorchAO-ineligible schemes keep fp32 moments)
                if "step" not in st:
                    st["step"] = 0
                    self._store(
                        st,
                        "m1",
                        torch.zeros_like(g, dtype=torch.float32),
                        m1_effective_scheme,
                        m1_parameter_bs,
                        generator=m1_generator,
                    )
                    if m2_effective_scheme in self._ORACLE_LOOKAHEAD_UPDATE_SR_SCHEMES:
                        st["lookahead_pending_exp_avg_sq"] = torch.zeros_like(g, dtype=torch.float32)
                    else:
                        self._store(
                            st,
                            "m2",
                            torch.zeros_like(g, dtype=torch.float32),
                            m2_effective_scheme,
                            m2_bs,
                            eden=use_eden_m2,
                            generator=m2_generator,
                        )

                st["step"] += 1

                # Load momentum and variance
                m1 = self._load(st, "m1", m1_effective_scheme, m1_scheme)
                if m2_effective_scheme in self._ORACLE_LOOKAHEAD_UPDATE_SR_SCHEMES:
                    m2 = st["lookahead_pending_exp_avg_sq"]
                else:
                    m2 = self._load(st, "m2", m2_effective_scheme, m2_scheme)

                # AdamW update
                m1.lerp_(g, 1 - beta1)

                bias_c1 = 1 - beta1 ** st["step"]
                bias_c2 = 1 - beta2 ** st["step"]

                # Standard AdamW: θ = θ - α * m̂ / (√v̂ + ε)
                # where m̂ = m / (1 - β₁ᵗ) and v̂ = v / (1 - β₂ᵗ)
                step_size = lr / bias_c1
                m2_code = None
                m2_quant_state = None
                if m2_effective_scheme in self._ORACLE_LOOKAHEAD_UPDATE_SR_SCHEMES:
                    st.pop("m2_read_preconditioner", None)
                    _, _, sampled_prev_m2 = quantize_dyn4_la_upd_sr(
                        m2,
                        g,
                        beta2=beta2,
                        bias_correction=bias_c2,
                        eps=eps,
                        block_size=m2_bs,
                        generator=m2_generator,
                    )
                    m2 = sampled_prev_m2.mul(beta2).addcmul(g, g, value=1 - beta2)
                    denom = (m2.sqrt() / math.sqrt(bias_c2)).add_(eps)
                elif m2_effective_scheme in self._PROXY_LOOKAHEAD_UPDATE_SR_SCHEMES:
                    # Memory-efficient causal approximation: use the current
                    # temporary v_t for this update, then store v_t with
                    # probabilities targeted at the next EMA transition, using
                    # the latest gradient as a proxy for g_{t+1}.
                    st.pop("m2_read_preconditioner", None)
                    m2.lerp_(g.square(), 1 - beta2)
                    denom = (m2.sqrt() / math.sqrt(bias_c2)).add_(eps)
                    bias_c2_next = 1 - beta2 ** (st["step"] + 1)
                    m2_code, m2_quant_state, _ = quantize_dyn4_la_upd_sr(
                        m2,
                        g,
                        beta2=beta2,
                        bias_correction=bias_c2_next,
                        eps=eps,
                        block_size=m2_bs,
                        generator=m2_generator,
                    )
                elif m2_effective_scheme in self._VPROXY_LOOKAHEAD_UPDATE_SR_SCHEMES:
                    # Self-proxy lookahead: approximate the next fresh
                    # contribution with the current temporary second moment.
                    st.pop("m2_read_preconditioner", None)
                    m2.lerp_(g.square(), 1 - beta2)
                    denom = (m2.sqrt() / math.sqrt(bias_c2)).add_(eps)
                    bias_c2_next = 1 - beta2 ** (st["step"] + 1)
                    m2_code, m2_quant_state, _ = quantize_dyn4_la_upd_sr(
                        m2,
                        g,
                        beta2=beta2,
                        bias_correction=bias_c2_next,
                        eps=eps,
                        block_size=m2_bs,
                        generator=m2_generator,
                        fresh_proxy=(1 - beta2) * m2,
                    )
                elif m2_effective_scheme in self._UPDATE_SR_STORE_EDEN_SCHEMES:
                    m2.lerp_(g.square(), 1 - beta2)
                    # Use the raw update-unbiased sample for the current step,
                    # but store the same sampled codes with only EDEN's
                    # recurrent-state block-scale correction applied.
                    if self.telemetry is not None:
                        self.telemetry(p, m2, m2_effective_scheme, m2_bs, eps, True, st["step"], beta2)
                    m2_code, raw_m2_quant_state, m2_for_update = quantize_with_dequantized(
                        m2,
                        block_size=m2_bs,
                        mode=m2_effective_scheme,
                        eden_correction=False,
                        bias_correction=bias_c2,
                        eps=eps,
                        generator=m2_generator,
                    )
                    denom = (m2_for_update.sqrt() / math.sqrt(bias_c2)).add_(eps)
                    st["m2_read_preconditioner"] = denom.reciprocal().detach()
                    m2_quant_state = apply_eden_correction_to_quant_state(
                        m2,
                        m2_code,
                        raw_m2_quant_state,
                        m2_effective_scheme,
                    )
                elif m2_effective_scheme in self._UPDATE_UNBIASED_SR_SCHEMES:
                    m2.lerp_(g.square(), 1 - beta2)
                    # These schemes choose SR probabilities to make the current
                    # Adam preconditioner unbiased, so the sampled m2 must be the
                    # same one used in the denominator and stored for the next step.
                    st.pop("m2_read_preconditioner", None)
                    if self.telemetry is not None:
                        self.telemetry(p, m2, m2_effective_scheme, m2_bs, eps, use_eden_m2, st["step"], beta2)
                    m2_code, m2_quant_state, m2_for_update = quantize_with_dequantized(
                        m2,
                        block_size=m2_bs,
                        mode=m2_effective_scheme,
                        eden_correction=use_eden_m2,
                        bias_correction=bias_c2,
                        eps=eps,
                        generator=m2_generator,
                    )
                    denom = (m2_for_update.sqrt() / math.sqrt(bias_c2)).add_(eps)
                elif m2_effective_scheme in (
                    self._UPDATE_SR_FP32_READ_SCHEMES
                    | self._UPDATE_SR_FP32_READ_NEXT_BIAS_SCHEMES
                ):
                    # Keep the current Adam denominator on temporary fp32 v_t.
                    # Lookahead-SR changes only the storage target from c_t to
                    # c_{t+1}; it does not apply another EMA transition to the
                    # candidate codebook values.
                    st.pop("m2_read_preconditioner", None)
                    m2.lerp_(g.square(), 1 - beta2)
                    if self.telemetry is not None:
                        self.telemetry(p, m2, m2_effective_scheme, m2_bs, eps, use_eden_m2, st["step"], beta2)
                    denom = (m2.sqrt() / math.sqrt(bias_c2)).add_(eps)
                    storage_bias_c2 = (
                        1 - beta2 ** (st["step"] + 1)
                        if m2_effective_scheme in self._UPDATE_SR_FP32_READ_NEXT_BIAS_SCHEMES
                        else bias_c2
                    )
                    m2_code, m2_quant_state = quantize(
                        m2,
                        block_size=m2_bs,
                        mode=m2_effective_scheme,
                        eden_correction=use_eden_m2,
                        bias_correction=storage_bias_c2,
                        eps=eps,
                        generator=m2_generator,
                    )
                else:
                    m2.lerp_(g.square(), 1 - beta2)
                    st.pop("m2_read_preconditioner", None)
                    m2_for_update = m2
                    denom = (m2_for_update.sqrt() / math.sqrt(bias_c2)).add_(eps)

                # Weight decay and param update
                p.data.mul_(1 - lr * wd)
                should_record_update = (
                    record_update_direction_every is not None
                    and st["step"] % record_update_direction_every == 0
                )
                should_record_preconditioner = (
                    (
                        record_preconditioner_every is not None
                        and st["step"] % record_preconditioner_every == 0
                    )
                    or (
                        record_preconditioner_steps is not None
                        and st["step"] in record_preconditioner_steps
                    )
                )
                if should_record_preconditioner:
                    raw_preconditioner = denom.reciprocal()
                    self._last_preconditioners[p] = (st["step"], raw_preconditioner.detach().clone())
                else:
                    self._last_preconditioners.pop(p, None)
                    self._last_effective_preconditioners.pop(p, None)
                if update_sign_clip is not None:
                    # Element-wise clamp of u = m̂1/(√m̂2+ε) to ±update_sign_clip: bounds a
                    # collapsed-m2 (tiny denominator) step per coordinate, but caps each coordinate
                    # independently so the direction shifts toward signSGD. (m1/bias_c1 is a fresh
                    # tensor; m1 is untouched.)
                    m1_hat = m1 / bias_c1
                    u_raw = m1_hat.div(denom)
                    u = u_raw.clamp(-update_sign_clip, update_sign_clip)
                    if should_record_preconditioner:
                        active = u_raw.abs() > update_sign_clip
                        m1_hat_abs = m1_hat.abs()
                        effective_preconditioner = torch.where(
                            active,
                            update_sign_clip / m1_hat_abs.clamp_min(torch.finfo(m1_hat.dtype).tiny),
                            raw_preconditioner,
                        )
                        self._last_effective_preconditioners[p] = (
                            st["step"],
                            effective_preconditioner.detach().clone(),
                        )
                    p.data.add_(u, alpha=-lr)
                elif update_norm_clip is not None:
                    self._last_effective_preconditioners.pop(p, None)
                    # Direction-preserving alternative: rescale each m2-quant block of u so its RMS
                    # <= update_norm_clip (gradient-clipping analog), keeping the block's direction.
                    u = _block_rms_clip((m1 / bias_c1).div_(denom), m2_bs, update_norm_clip)
                    p.data.add_(u, alpha=-lr)
                else:
                    self._last_effective_preconditioners.pop(p, None)
                    if should_record_update:
                        u = (m1 / bias_c1).div(denom)
                    p.data.addcdiv_(m1, denom, value=-step_size)
                if should_record_update:
                    self._last_update_directions[p] = (st["step"], u.detach().clone())
                else:
                    self._last_update_directions.pop(p, None)

                # Diagnostics on the m2 about to be quantized (quantized m2 only)
                if (
                    self.telemetry is not None
                    and m2_effective_scheme != "fp32"
                    and m2_effective_scheme not in self._ORACLE_LOOKAHEAD_UPDATE_SR_SCHEMES
                    and m2_effective_scheme not in self._PROXY_LOOKAHEAD_UPDATE_SR_SCHEMES
                    and m2_effective_scheme not in self._VPROXY_LOOKAHEAD_UPDATE_SR_SCHEMES
                    and m2_effective_scheme not in self._UPDATE_SR_STORE_EDEN_SCHEMES
                    and m2_effective_scheme not in self._UPDATE_UNBIASED_SR_SCHEMES
                    and m2_effective_scheme not in self._UPDATE_SR_FP32_READ_SCHEMES
                    and m2_effective_scheme not in self._UPDATE_SR_FP32_READ_NEXT_BIAS_SCHEMES
                ):
                    self.telemetry(p, m2, m2_effective_scheme, m2_bs, eps, use_eden_m2, st["step"], beta2)

                # Re-quantize / store state
                self._store(
                    st,
                    "m1",
                    m1,
                    m1_effective_scheme,
                    m1_parameter_bs,
                    generator=m1_generator,
                )
                if (
                    m2_effective_scheme in self._ORACLE_LOOKAHEAD_UPDATE_SR_SCHEMES
                ):
                    st["lookahead_pending_exp_avg_sq"] = m2.detach()
                    st.pop("m2", None)
                    st.pop("m2_code", None)
                    st.pop("m2_quant_state", None)
                elif (
                    m2_effective_scheme in self._UPDATE_SR_STORE_EDEN_SCHEMES
                    or m2_effective_scheme in self._UPDATE_UNBIASED_SR_SCHEMES
                    or m2_effective_scheme in self._UPDATE_SR_FP32_READ_SCHEMES
                    or m2_effective_scheme in self._UPDATE_SR_FP32_READ_NEXT_BIAS_SCHEMES
                    or m2_effective_scheme in self._PROXY_LOOKAHEAD_UPDATE_SR_SCHEMES
                    or m2_effective_scheme in self._VPROXY_LOOKAHEAD_UPDATE_SR_SCHEMES
                ):
                    self._store_prequantized(
                        st,
                        "m2",
                        m2_code,
                        m2_quant_state,
                        m2_effective_scheme,
                    )
                else:
                    self._store(
                        st,
                        "m2",
                        m2,
                        m2_effective_scheme,
                        m2_bs,
                        eden=use_eden_m2,
                        generator=m2_generator,
                    )

        return loss

    @staticmethod
    def _eligible_for_quantized_state(p: torch.Tensor, block_size: int) -> bool:
        """Mirror TorchAO optimizer-state eligibility for quantized state tensors."""
        n = p.numel()
        return n >= _TORCHAO_MIN_QUANT_NUMEL and n % block_size == 0

    @classmethod
    def _effective_scheme(cls, p: torch.Tensor, scheme: str, block_size: int) -> str:
        if scheme == "fp32" or cls._eligible_for_quantized_state(p, block_size):
            return scheme
        return "fp32"

    @staticmethod
    def _store(
        st,
        key,
        value,
        scheme,
        bs,
        eden=False,
        generator=None,
    ):
        """Save a moment buffer under `key`, quantized per `scheme` ('fp32' = raw)."""
        if scheme == "fp32":
            st[key] = value.to(dtype=torch.float32)
            st.pop(f"{key}_code", None)
            st.pop(f"{key}_quant_state", None)
        else:
            codes, quant_state = quantize(
                value,
                block_size=bs,
                mode=scheme,
                eden_correction=eden,
                generator=generator,
            )
            st[f"{key}_code"], st[f"{key}_quant_state"] = (
                pack_codes_for_storage(codes, quant_state, mode=scheme)
            )
            st.pop(key, None)

    @staticmethod
    def _store_prequantized(st, key, codes, quant_state, scheme):
        """Save an already-sampled quantized moment without re-quantizing."""
        st[f"{key}_code"], st[f"{key}_quant_state"] = (
            pack_codes_for_storage(codes, quant_state, mode=scheme)
        )
        st.pop(key, None)

    @staticmethod
    def _load(st, key, scheme, decode_scheme=None):
        """Load a moment buffer (returns a tensor that may be mutated in place)."""
        if scheme == "fp32":
            if key not in st and decode_scheme is not None and f"{key}_code" in st:
                return dequantize(st[f"{key}_code"], st[f"{key}_quant_state"], mode=decode_scheme)
            return st[key]
        return dequantize(st[f"{key}_code"], st[f"{key}_quant_state"], mode=scheme)


class AdamW8bit(QuantizedAdamW):
    """Compatibility name retaining the original 8-bit linear defaults."""


class ZIPSRAdamW4Bit(QuantizedAdamW):
    """NF4 first moments and zero-inclusive Dyn4 preconditioner-space SR."""

    def __init__(self, params, **kwargs):
        super().__init__(
            params,
            m1_quant_scheme="nf4",
            m2_quant_scheme="dyn4_upd_sr_fp32read",
            use_eden_m2=False,
            **kwargs,
        )


class ZEEDENAdamW4Bit(QuantizedAdamW):
    """NF4 first moments and zero-exclusive Dyn4 second moments with EDEN."""

    def __init__(self, params, **kwargs):
        super().__init__(
            params,
            m1_quant_scheme="nf4",
            m2_quant_scheme="dyn4_nz",
            use_eden_m2=True,
            **kwargs,
        )
