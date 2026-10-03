import pytest
import torch

from adamw4bit import AdamW8bit
from adamw4bit.quantization import (
    _block,
    apply_eden_correction_to_quant_state,
    dequantize,
    quantize,
)


@pytest.mark.parametrize(
    "mode",
    ["lin4_upd_sr_store_eden", "dyn4_upd_sr_store_eden"],
)
def test_upd_sr_store_eden_applies_eden_to_saved_state(mode: str) -> None:
    v = torch.tensor([0.0, 0.015, 0.05, 0.2, 0.001, 0.07, 0.11, 0.35])
    block_size = 4
    codes, raw_state = quantize(
        v,
        block_size=block_size,
        mode=mode,
        bias_correction=0.25,
        eps=1e-8,
        generator=torch.Generator().manual_seed(0),
    )

    eden_state = apply_eden_correction_to_quant_state(v, codes, raw_state, mode)
    saved = dequantize(codes, eden_state, mode=mode)
    blocked_v, _ = _block(v, block_size)
    blocked_saved, _ = _block(saved, block_size)

    numerator = (blocked_v.float() * blocked_v.float()).sum(dim=1)
    denominator = (blocked_v.float() * blocked_saved.float()).sum(dim=1)
    torch.testing.assert_close(denominator, numerator, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize(
    "mode",
    ["lin4_upd_sr_store_eden", "dyn4_upd_sr_store_eden"],
)
def test_adamw8bit_upd_sr_store_eden_stores_preconditioner(mode: str) -> None:
    p = torch.nn.Parameter(torch.linspace(-1.0, 1.0, steps=4096))
    opt = AdamW8bit(
        [p],
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        m1_quant_scheme="fp32",
        m2_quant_scheme=mode,
        block_size=128,
    )

    p.grad = torch.linspace(-0.5, 0.5, steps=4096)
    opt.step()

    state = opt.state[p]
    assert "m2_code" in state
    assert "m2_quant_state" in state
    assert "m2_read_preconditioner" in state
    assert "m2" not in state
    assert state["m2_read_preconditioner"].shape == p.shape
    assert torch.isfinite(p).all()


@pytest.mark.parametrize(
    "mode",
    ["lin4_upd_sr_store_eden", "dyn4_upd_sr_store_eden"],
)
def test_upd_sr_store_eden_rejects_use_eden_m2(mode: str) -> None:
    p = torch.nn.Parameter(torch.tensor([1.0]))

    with pytest.raises(ValueError, match="EDEN is built into the store path"):
        AdamW8bit(
            [p],
            m1_quant_scheme="fp32",
            m2_quant_scheme=mode,
            use_eden_m2=True,
        )
