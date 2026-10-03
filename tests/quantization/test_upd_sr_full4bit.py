import pytest
import torch

from adamw4bit import AdamW8bit


@pytest.mark.parametrize("m2_quant_scheme", ["dyn4_upd_sr", "lin4_upd_sr", "dyn4_nz", "lin4_nz"])
def test_adamw8bit_accepts_sdyn4_m1_with_full4bit_m2(m2_quant_scheme: str) -> None:
    p = torch.nn.Parameter(torch.linspace(-1.0, 1.0, steps=4096))
    opt = AdamW8bit(
        [p],
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        m1_quant_scheme="sdyn4",
        m2_quant_scheme=m2_quant_scheme,
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
    assert state["m1_code"].numel() == p.numel() // 2
    assert state["m2_code"].numel() == p.numel() // 2
    assert state["m1_quant_state"].packed is True
    assert state["m2_quant_state"].packed is True
    assert torch.isfinite(p).all()


@pytest.mark.parametrize("m1_quant_scheme", ["sdyn4_sr", "nf4"])
def test_adamw8bit_accepts_new_m1_4bit_schemes(m1_quant_scheme: str) -> None:
    p = torch.nn.Parameter(torch.linspace(-1.0, 1.0, steps=4096))
    opt = AdamW8bit(
        [p],
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        m1_quant_scheme=m1_quant_scheme,
        m2_quant_scheme="dyn4_nz",
        block_size=128,
    )

    p.grad = torch.linspace(-0.5, 0.5, steps=4096)
    opt.step()

    state = opt.state[p]
    assert "m1_code" in state
    assert "m1_quant_state" in state
    assert "m1" not in state
    assert state["m1_code"].numel() == p.numel() // 2
    assert state["m1_quant_state"].packed is True
    assert torch.isfinite(p).all()
