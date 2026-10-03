import torch

from adamw4bit import AdamW8bit


def _step_once(opt: AdamW8bit, p: torch.nn.Parameter) -> None:
    p.grad = torch.linspace(-0.5, 0.5, steps=p.numel(), device=p.device, dtype=p.dtype).reshape_as(p)
    opt.step()


def test_adamw8bit_keeps_small_quantized_moments_in_fp32() -> None:
    p = torch.nn.Parameter(torch.linspace(-1.0, 1.0, steps=768))
    opt = AdamW8bit(
        [p],
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        m1_quant_scheme="sdyn4",
        m2_quant_scheme="lin4_nz",
        block_size=128,
    )

    _step_once(opt, p)

    state = opt.state[p]
    assert "m1" in state
    assert "m2" in state
    assert "m1_code" not in state
    assert "m2_code" not in state
    assert state["m1"].shape == p.shape
    assert state["m2"].shape == p.shape


def test_adamw8bit_fp32_fallback_is_independent_of_parameter_dtype() -> None:
    p = torch.nn.Parameter(torch.linspace(-1.0, 1.0, steps=768, dtype=torch.bfloat16))
    opt = AdamW8bit(
        [p],
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        m1_quant_scheme="sdyn4",
        m2_quant_scheme="lin4_nz",
        block_size=128,
    )

    _step_once(opt, p)

    state = opt.state[p]
    assert state["m1"].dtype is torch.float32
    assert state["m2"].dtype is torch.float32
    assert p.dtype is torch.bfloat16


def test_adamw8bit_quantizes_large_block_aligned_moments() -> None:
    p = torch.nn.Parameter(torch.linspace(-1.0, 1.0, steps=4096))
    opt = AdamW8bit(
        [p],
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        m1_quant_scheme="sdyn4",
        m2_quant_scheme="lin4_nz",
        block_size=128,
    )

    _step_once(opt, p)

    state = opt.state[p]
    assert "m1_code" in state
    assert "m2_code" in state
    assert "m1" not in state
    assert "m2" not in state
    assert state["m1_code"].numel() == p.numel() // 2
    assert state["m2_code"].numel() == p.numel() // 2
    assert state["m1_quant_state"].shape == p.shape
    assert state["m2_quant_state"].shape == p.shape
    assert state["m1_quant_state"].packed is True
    assert state["m2_quant_state"].packed is True


def test_adamw8bit_keeps_large_non_aligned_moments_in_fp32() -> None:
    p = torch.nn.Parameter(torch.linspace(-1.0, 1.0, steps=4097))
    opt = AdamW8bit(
        [p],
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        m1_quant_scheme="sdyn4",
        m2_quant_scheme="dyn4_nz",
        block_size=128,
    )

    _step_once(opt, p)

    state = opt.state[p]
    assert "m1" in state
    assert "m2" in state
    assert "m1_code" not in state
    assert "m2_code" not in state


def test_adamw8bit_m2only_keeps_m1_fp32_and_quantizes_eligible_m2() -> None:
    p = torch.nn.Parameter(torch.linspace(-1.0, 1.0, steps=4096))
    opt = AdamW8bit(
        [p],
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        m1_quant_scheme="fp32",
        m2_quant_scheme="dyn4_nz",
        block_size=128,
    )

    _step_once(opt, p)

    state = opt.state[p]
    assert "m1" in state
    assert "m1_code" not in state
    assert "m2_code" in state
    assert "m2" not in state
    assert state["m1"].shape == p.shape
    assert state["m2_code"].numel() == p.numel() // 2
    assert state["m2_quant_state"].shape == p.shape
    assert state["m2_quant_state"].packed is True


def test_adamw8bit_records_true_preconditioner_on_requested_steps() -> None:
    p = torch.nn.Parameter(torch.linspace(-1.0, 1.0, steps=4096))
    opt = AdamW8bit(
        [p],
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        m1_quant_scheme="fp32",
        m2_quant_scheme="dyn4_nz",
        block_size=128,
        record_preconditioner_steps=[1],
    )

    _step_once(opt, p)

    step, preconditioner = opt._last_preconditioners[p]
    expected = 1.0 / (p.grad.abs() + opt.param_groups[0]["eps"])
    assert step == 1
    torch.testing.assert_close(preconditioner, expected, rtol=1e-5, atol=1e-5)

    _step_once(opt, p)

    assert p not in opt._last_preconditioners


def test_adamw8bit_records_sign_effective_preconditioner_on_requested_steps() -> None:
    p = torch.nn.Parameter(torch.linspace(-1.0, 1.0, steps=4))
    update_sign_clip = 0.25
    opt = AdamW8bit(
        [p],
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        m1_quant_scheme="fp32",
        m2_quant_scheme="fp32",
        update_sign_clip=update_sign_clip,
        record_preconditioner_steps=[1],
    )

    _step_once(opt, p)

    step, raw_preconditioner = opt._last_preconditioners[p]
    effective_step, effective_preconditioner = opt._last_effective_preconditioners[p]
    expected_raw = 1.0 / (p.grad.abs() + opt.param_groups[0]["eps"])
    expected_effective = update_sign_clip / p.grad.abs()

    assert step == 1
    assert effective_step == 1
    torch.testing.assert_close(raw_preconditioner, expected_raw)
    torch.testing.assert_close(effective_preconditioner, expected_effective)
