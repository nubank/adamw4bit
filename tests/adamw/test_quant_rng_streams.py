import copy

import torch

from adamw4bit import AdamW8bit


def _optimizer(parameter: torch.nn.Parameter) -> AdamW8bit:
    return AdamW8bit(
        [parameter],
        lr=1e-3,
        weight_decay=0.0,
        block_size=128,
        m1_quant_scheme="nf4",
        m2_quant_scheme="dyn4_upd_sr_fp32read",
        quant_rng_seed=43,
    )


def _set_gradient(parameter: torch.nn.Parameter, step: int) -> None:
    parameter.grad = torch.linspace(
        -0.5 + 0.01 * step,
        0.5 + 0.01 * step,
        parameter.numel(),
    )


def test_runtime_m1_sr_isolates_m2_rng_and_resumes() -> None:
    rtn_parameter = torch.nn.Parameter(
        torch.linspace(-1.0, 1.0, 4096)
    )
    sr_parameter = torch.nn.Parameter(rtn_parameter.detach().clone())
    rtn = _optimizer(rtn_parameter)
    sr = _optimizer(sr_parameter)
    sr.set_m1_quant_scheme_for_parameters(
        [sr_parameter],
        "nf4_sr",
    )

    for step in range(2):
        _set_gradient(rtn_parameter, step)
        _set_gradient(sr_parameter, step)
        rtn.step()
        sr.step()

    torch.testing.assert_close(
        rtn.state[rtn_parameter]["m2_code"],
        sr.state[sr_parameter]["m2_code"],
    )

    saved_state = copy.deepcopy(sr.state_dict())
    resumed_parameter = torch.nn.Parameter(
        sr_parameter.detach().clone()
    )
    resumed = _optimizer(resumed_parameter)
    resumed.set_m1_quant_scheme_for_parameters(
        [resumed_parameter],
        "nf4_sr",
    )
    resumed.load_state_dict(saved_state)

    _set_gradient(sr_parameter, 2)
    _set_gradient(resumed_parameter, 2)
    sr.step()
    resumed.step()

    torch.testing.assert_close(sr_parameter, resumed_parameter)
    for key in ("m1_code", "m2_code"):
        torch.testing.assert_close(
            sr.state[sr_parameter][key],
            resumed.state[resumed_parameter][key],
        )
