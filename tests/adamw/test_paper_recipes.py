import pytest
import torch

from adamw4bit import ZEEDENAdamW4Bit, ZIPSRAdamW4Bit


@pytest.mark.parametrize("optimizer_class", [ZEEDENAdamW4Bit, ZIPSRAdamW4Bit])
def test_paper_recipe_runs_with_packed_full_state(optimizer_class) -> None:
    parameter = torch.nn.Parameter(torch.linspace(-1.0, 1.0, 4096))
    optimizer = optimizer_class(
        [parameter],
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    parameter.grad = torch.linspace(-0.5, 0.5, parameter.numel())

    optimizer.step()

    state = optimizer.state[parameter]
    assert state["m1_quant_state"].packed is True
    assert state["m2_quant_state"].packed is True
    assert state["m1_code"].numel() == parameter.numel() // 2
    assert state["m2_code"].numel() == parameter.numel() // 2
    assert torch.isfinite(parameter).all()
