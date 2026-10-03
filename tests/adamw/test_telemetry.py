import pytest
import torch

from adamw4bit import ZEEDENAdamW4Bit, ZIPSRAdamW4Bit
from adamw4bit.quantization import dequantize


@pytest.mark.parametrize(
    "optimizer_class,mode,eden",
    [
        (ZEEDENAdamW4Bit, "dyn4_nz", True),
        (ZIPSRAdamW4Bit, "dyn4_upd_sr_fp32read", False),
    ],
)
@pytest.mark.parametrize("numel", [768, 4096])
def test_telemetry_reports_working_second_moment_once_per_step(
    optimizer_class, mode, eden, numel,
):
    parameter = torch.nn.Parameter(torch.ones(numel, dtype=torch.bfloat16))
    calls = []

    def telemetry(param, moment, scheme, block_size, eps, use_eden, step, beta2):
        calls.append((param, moment.clone(), scheme, block_size, eps, use_eden, step, beta2))

    beta2 = 0.8
    eps = 3e-7
    optimizer = optimizer_class(
        [parameter], betas=(0.9, beta2), eps=eps, telemetry=telemetry,
        quant_rng_seed=43,
    )
    previous_moment = torch.zeros(numel, dtype=torch.float32)
    for step in (1, 2):
        parameter.grad = torch.linspace(-0.5, 0.5, numel, dtype=parameter.dtype) * step
        expected = beta2 * previous_moment + (1 - beta2) * parameter.grad.float().square()
        optimizer.step()
        if numel < 4096:
            assert calls == []
            continue

        assert len(calls) == step
        param, moment, scheme, block_size, observed_eps, use_eden, observed_step, observed_beta2 = calls[-1]
        assert param is parameter
        assert moment.dtype == torch.float32
        torch.testing.assert_close(moment, expected, rtol=1e-6, atol=1e-8)
        assert (scheme, block_size, observed_eps, use_eden, observed_step, observed_beta2) == (
            mode, 128, eps, eden, step, beta2,
        )
        state = optimizer.state[parameter]
        previous_moment = dequantize(state["m2_code"], state["m2_quant_state"], mode=mode)
