import copy

import pytest
import torch

from adamw4bit import AdamW8bit
from adamw4bit.quantization import unpack_4bit_codes


def _optimizer(parameter: torch.nn.Parameter) -> AdamW8bit:
    return AdamW8bit(
        [parameter],
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        m1_quant_scheme="sdyn4",
        m2_quant_scheme="dyn4_nz",
        block_size=128,
    )


def _step(
    optimizer: AdamW8bit,
    parameter: torch.nn.Parameter,
    scale: float,
) -> None:
    parameter.grad = torch.linspace(
        -scale,
        scale,
        steps=parameter.numel(),
    )
    optimizer.step()


def test_packed_4bit_optimizer_state_dict_roundtrip() -> None:
    first_parameter = torch.nn.Parameter(torch.linspace(-1.0, 1.0, 4096))
    first_optimizer = _optimizer(first_parameter)
    _step(first_optimizer, first_parameter, 0.5)
    checkpoint = copy.deepcopy(first_optimizer.state_dict())

    second_parameter = torch.nn.Parameter(first_parameter.detach().clone())
    second_optimizer = _optimizer(second_parameter)
    second_optimizer.load_state_dict(checkpoint)

    _step(first_optimizer, first_parameter, 0.25)
    _step(second_optimizer, second_parameter, 0.25)

    torch.testing.assert_close(second_parameter, first_parameter)
    second_state = second_optimizer.state[second_parameter]
    assert second_state["m1_quant_state"].packed is True
    assert second_state["m2_quant_state"].packed is True


def test_legacy_unpacked_optimizer_state_is_read_and_repacked() -> None:
    parameter = torch.nn.Parameter(torch.linspace(-1.0, 1.0, 4096))
    optimizer = _optimizer(parameter)
    _step(optimizer, parameter, 0.5)
    state = optimizer.state[parameter]

    for key in ("m1", "m2"):
        quant_state = state[f"{key}_quant_state"]
        state[f"{key}_code"] = unpack_4bit_codes(
            state[f"{key}_code"],
            numel=parameter.numel(),
        ).reshape(parameter.shape)
        quant_state.packed = False

    _step(optimizer, parameter, 0.25)

    assert torch.isfinite(parameter).all()
    assert state["m1_code"].numel() == parameter.numel() // 2
    assert state["m2_code"].numel() == parameter.numel() // 2
    assert state["m1_quant_state"].packed is True
    assert state["m2_quant_state"].packed is True


@pytest.mark.parametrize(
    ("scheme", "code_dtype"),
    [("linear", torch.int8), ("dynamic", torch.uint8)],
)
def test_loading_8bit_state_preserves_integer_code_dtype(
    scheme: str,
    code_dtype: torch.dtype,
) -> None:
    first_parameter = torch.nn.Parameter(torch.linspace(-1.0, 1.0, 4096))
    first_optimizer = AdamW8bit(
        [first_parameter],
        lr=1e-3,
        m1_quant_scheme=scheme,
        m2_quant_scheme=scheme,
        block_size=256,
    )
    _step(first_optimizer, first_parameter, 0.5)

    second_parameter = torch.nn.Parameter(first_parameter.detach().clone())
    second_optimizer = AdamW8bit(
        [second_parameter],
        lr=1e-3,
        m1_quant_scheme=scheme,
        m2_quant_scheme=scheme,
        block_size=256,
    )
    second_optimizer.load_state_dict(copy.deepcopy(first_optimizer.state_dict()))

    state = second_optimizer.state[second_parameter]
    assert state["m1_code"].dtype is code_dtype
    assert state["m2_code"].dtype is code_dtype
    assert state["m1_quant_state"].packed is False
    assert state["m2_quant_state"].packed is False
