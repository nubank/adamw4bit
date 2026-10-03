import pytest
import torch

import adamw4bit.adamw as adamw_8bit_module
from adamw4bit import AdamW8bit
from adamw4bit.quantization import (
    default_block_size,
    dequantize,
    quantize,
    quantize_dyn4_la_upd_sr,
)


def _preconditioner(
    prev: torch.Tensor,
    grad: torch.Tensor,
    *,
    beta2: float,
    bias_correction: float,
    eps: float,
) -> torch.Tensor:
    fresh = (1.0 - beta2) * grad.square()
    return 1.0 / (((beta2 * prev + fresh) / bias_correction).sqrt() + eps)


def test_dyn4_la_upd_sr_uses_dyn4_block_scale() -> None:
    z_prev = torch.linspace(1e-4, 0.5, steps=128, dtype=torch.float32)
    grad = torch.linspace(-0.2, 0.2, steps=128, dtype=torch.float32)

    _, dyn4_state = quantize(z_prev, block_size=128, mode="dyn4")
    _, la_state, _ = quantize_dyn4_la_upd_sr(
        z_prev,
        grad,
        beta2=0.95,
        bias_correction=0.25,
        eps=1e-8,
        block_size=128,
        generator=torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(la_state.absmax, dyn4_state.absmax)


def test_dyn4_la_upd_sr_monte_carlo_preconditioner_unbiased() -> None:
    z_prev = torch.linspace(0.02, 0.5, steps=128, dtype=torch.float32)
    grad = torch.linspace(0.05, 0.25, steps=128, dtype=torch.float32)
    beta2 = 0.95
    bias_correction = 0.4
    eps = 1e-8
    n_samples = 1000

    sample_sum = torch.zeros_like(z_prev, dtype=torch.float64)
    for seed in range(n_samples):
        _, _, sampled = quantize_dyn4_la_upd_sr(
            z_prev,
            grad,
            beta2=beta2,
            bias_correction=bias_correction,
            eps=eps,
            block_size=128,
            generator=torch.Generator().manual_seed(seed),
        )
        sample_sum += _preconditioner(
            sampled.double(),
            grad.double(),
            beta2=beta2,
            bias_correction=bias_correction,
            eps=eps,
        )

    sample_mean = sample_sum / n_samples
    reference = _preconditioner(
        z_prev.double(),
        grad.double(),
        beta2=beta2,
        bias_correction=bias_correction,
        eps=eps,
    )
    torch.testing.assert_close(sample_mean.mean(), reference.mean(), rtol=0.015, atol=0.015)


def test_adamw8bit_dyn4_la_upd_sr_first_step_matches_fp32_adamw() -> None:
    values = torch.linspace(-1.0, 1.0, steps=4096)
    grad = torch.linspace(-0.5, 0.5, steps=4096)
    p_oracle = torch.nn.Parameter(values.clone())
    p_fp32 = torch.nn.Parameter(values.clone())

    kwargs = {
        "lr": 1e-3,
        "weight_decay": 0.0,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    }
    opt_oracle = AdamW8bit(
        [p_oracle],
        **kwargs,
        m1_quant_scheme="fp32",
        m2_quant_scheme="dyn4_la_upd_sr",
        block_size=128,
    )
    opt_fp32 = torch.optim.AdamW([p_fp32], **kwargs)

    p_oracle.grad = grad.clone()
    p_fp32.grad = grad.clone()
    opt_oracle.step()
    opt_fp32.step()

    torch.testing.assert_close(p_oracle, p_fp32, rtol=1e-6, atol=1e-6)
    state = opt_oracle.state[p_oracle]
    assert "lookahead_pending_exp_avg_sq" in state
    assert "m2_code" not in state
    assert "m2_quant_state" not in state
    expected_m2 = (1.0 - kwargs["betas"][1]) * grad.square()
    torch.testing.assert_close(state["lookahead_pending_exp_avg_sq"], expected_m2)


def test_adamw8bit_dyn4_la_upd_sr_accepts_nf4_m1_full4bit() -> None:
    p = torch.nn.Parameter(torch.linspace(-1.0, 1.0, steps=4096))
    opt = AdamW8bit(
        [p],
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        m1_quant_scheme="nf4",
        m2_quant_scheme="dyn4_la_upd_sr",
        block_size=128,
    )

    p.grad = torch.linspace(-0.5, 0.5, steps=4096)
    opt.step()

    state = opt.state[p]
    assert "m1_code" in state
    assert "m1_quant_state" in state
    assert "m1" not in state
    assert "lookahead_pending_exp_avg_sq" in state
    assert "m2_code" not in state
    assert "m2_quant_state" not in state
    assert torch.isfinite(p).all()


@pytest.mark.parametrize(
    "scheme",
    [
        "lin4_upd_sr_fp32read",
        "dyn4_upd_sr_fp32read",
        "dyn4_lookahead_sr",
        "dyn4_upd_sr_fp32read_nextbc",
    ],
)
def test_adamw8bit_upd_sr_fp32read_first_step_matches_fp32_and_stores_quantized_m2(
    scheme: str,
) -> None:
    values = torch.linspace(-1.0, 1.0, steps=4096)
    grad = torch.linspace(-0.5, 0.5, steps=4096)
    p_fp32read = torch.nn.Parameter(values.clone())
    p_fp32 = torch.nn.Parameter(values.clone())

    kwargs = {
        "lr": 1e-3,
        "weight_decay": 0.0,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    }
    opt_fp32read = AdamW8bit(
        [p_fp32read],
        **kwargs,
        m1_quant_scheme="fp32",
        m2_quant_scheme=scheme,
        block_size=128,
    )
    opt_fp32 = torch.optim.AdamW([p_fp32], **kwargs)

    p_fp32read.grad = grad.clone()
    p_fp32.grad = grad.clone()
    opt_fp32read.step()
    opt_fp32.step()

    torch.testing.assert_close(p_fp32read, p_fp32, rtol=1e-6, atol=1e-6)
    state = opt_fp32read.state[p_fp32read]
    assert "lookahead_pending_exp_avg_sq" not in state
    assert "m2_code" in state
    assert "m2_quant_state" in state
    assert "m2" not in state
    saved_m2 = dequantize(
        state["m2_code"],
        state["m2_quant_state"],
        mode=scheme,
    )
    assert saved_m2.shape == p_fp32read.shape
    assert torch.isfinite(saved_m2).all()


@pytest.mark.parametrize(
    "scheme",
    [
        "lin4_upd_sr_fp32read",
        "dyn4_upd_sr_fp32read",
        "dyn4_lookahead_sr",
        "dyn4_upd_sr_fp32read_nextbc",
    ],
)
@pytest.mark.parametrize("m1_scheme", ["nf4", "sdyn4"])
def test_adamw8bit_upd_sr_fp32read_accepts_4bit_m1_full4bit(
    scheme: str,
    m1_scheme: str,
) -> None:
    p = torch.nn.Parameter(torch.linspace(-1.0, 1.0, steps=4096))
    opt = AdamW8bit(
        [p],
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        m1_quant_scheme=m1_scheme,
        m2_quant_scheme=scheme,
        block_size=128,
    )

    p.grad = torch.linspace(-0.5, 0.5, steps=4096)
    opt.step()

    state = opt.state[p]
    assert "m1_code" in state
    assert "m1_quant_state" in state
    assert "m1" not in state
    assert "m2_code" in state
    assert "m2_quant_state" in state
    assert "m2" not in state
    assert torch.isfinite(p).all()


@pytest.mark.parametrize(
    ("scheme", "bias_step"),
    [
        ("dyn4_upd_sr_fp32read", 1),
        ("dyn4_lookahead_sr", 2),
        ("dyn4_upd_sr_fp32read_nextbc", 2),
    ],
)
def test_adamw8bit_upd_sr_fp32read_uses_configured_storage_bias_correction(
    monkeypatch: pytest.MonkeyPatch,
    scheme: str,
    bias_step: int,
) -> None:
    observed_bias_corrections: list[float | None] = []
    original_quantize = adamw_8bit_module.quantize

    def capture_quantize(*args, **kwargs):
        if kwargs.get("mode") == scheme:
            observed_bias_corrections.append(kwargs.get("bias_correction"))
        return original_quantize(*args, **kwargs)

    monkeypatch.setattr(adamw_8bit_module, "quantize", capture_quantize)

    beta2 = 0.95
    p = torch.nn.Parameter(torch.linspace(-1.0, 1.0, steps=4096))
    opt = AdamW8bit(
        [p],
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, beta2),
        m1_quant_scheme="fp32",
        m2_quant_scheme=scheme,
        block_size=128,
    )

    p.grad = torch.linspace(-0.5, 0.5, steps=4096)
    opt.step()

    # Initialization quantizes exact zeros without a bias target. At t=1,
    # Ordinary fp32-read storage targets c_t while lookahead-SR (and its
    # historical nextbc alias) targets c_{t+1}.
    assert observed_bias_corrections[-1] == pytest.approx(1 - beta2**bias_step)


@pytest.mark.parametrize("scheme", ["dyn4_proxy_la_upd_sr", "dyn4_vproxy_la_upd_sr"])
def test_adamw8bit_memory_efficient_lookahead_first_step_matches_fp32_and_stores_quantized_m2(scheme: str) -> None:
    values = torch.linspace(-1.0, 1.0, steps=4096)
    grad = torch.linspace(-0.5, 0.5, steps=4096)
    p_proxy = torch.nn.Parameter(values.clone())
    p_fp32 = torch.nn.Parameter(values.clone())

    kwargs = {
        "lr": 1e-3,
        "weight_decay": 0.0,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    }
    opt_proxy = AdamW8bit(
        [p_proxy],
        **kwargs,
        m1_quant_scheme="fp32",
        m2_quant_scheme=scheme,
        block_size=128,
    )
    opt_fp32 = torch.optim.AdamW([p_fp32], **kwargs)

    p_proxy.grad = grad.clone()
    p_fp32.grad = grad.clone()
    opt_proxy.step()
    opt_fp32.step()

    torch.testing.assert_close(p_proxy, p_fp32, rtol=1e-6, atol=1e-6)
    state = opt_proxy.state[p_proxy]
    assert "lookahead_pending_exp_avg_sq" not in state
    assert "m2_code" in state
    assert "m2_quant_state" in state
    assert "m2" not in state
    saved_m2 = dequantize(
        state["m2_code"],
        state["m2_quant_state"],
        mode=scheme,
    )
    assert saved_m2.shape == p_proxy.shape
    assert torch.isfinite(saved_m2).all()


@pytest.mark.parametrize("scheme", ["dyn4_proxy_la_upd_sr", "dyn4_vproxy_la_upd_sr"])
def test_adamw8bit_memory_efficient_lookahead_accepts_nf4_m1_full4bit(scheme: str) -> None:
    p = torch.nn.Parameter(torch.linspace(-1.0, 1.0, steps=4096))
    opt = AdamW8bit(
        [p],
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        m1_quant_scheme="nf4",
        m2_quant_scheme=scheme,
        block_size=128,
    )

    p.grad = torch.linspace(-0.5, 0.5, steps=4096)
    opt.step()

    state = opt.state[p]
    assert "m1_code" in state
    assert "m1_quant_state" in state
    assert "m1" not in state
    assert "m2_code" in state
    assert "m2_quant_state" in state
    assert "m2" not in state
    assert torch.isfinite(p).all()


@pytest.mark.parametrize("scheme", ["dyn4_la_upd_sr", "dyn4_proxy_la_upd_sr", "dyn4_vproxy_la_upd_sr"])
def test_adamw8bit_lookahead_sr_rejects_extra_corrections(scheme: str) -> None:
    p = torch.nn.Parameter(torch.linspace(-1.0, 1.0, steps=4096))

    with pytest.raises(ValueError, match="supports m1_quant_scheme"):
        AdamW8bit([p], m1_quant_scheme="sdyn4", m2_quant_scheme=scheme)
    with pytest.raises(ValueError, match="use_eden_m2"):
        AdamW8bit([p], m1_quant_scheme="fp32", m2_quant_scheme=scheme, use_eden_m2=True)
    with pytest.raises(ValueError, match="update clipping"):
        AdamW8bit([p], m1_quant_scheme="fp32", m2_quant_scheme=scheme, update_norm_clip=1.0)


@pytest.mark.parametrize("scheme", ["dyn4_la_upd_sr", "dyn4_proxy_la_upd_sr", "dyn4_vproxy_la_upd_sr"])
def test_lookahead_sr_defaults_to_128_element_blocks(scheme: str) -> None:
    assert default_block_size(scheme) == 128


@pytest.mark.parametrize(
    "scheme",
    [
        "lin4_upd_sr_fp32read",
        "dyn4_upd_sr_fp32read",
        "dyn4_lookahead_sr",
        "dyn4_upd_sr_fp32read_nextbc",
    ],
)
def test_upd_sr_fp32read_defaults_to_128_element_blocks(scheme: str) -> None:
    assert default_block_size(scheme) == 128
