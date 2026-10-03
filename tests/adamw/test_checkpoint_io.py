import copy
import io

import pytest
import torch

from adamw4bit import ZEEDENAdamW4Bit, ZIPSRAdamW4Bit
from adamw4bit.quantization import QuantState


RECIPES = [ZEEDENAdamW4Bit, ZIPSRAdamW4Bit]
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _step(optimizer, parameter, step):
    generator = torch.Generator().manual_seed(123 + step)
    parameter.grad = torch.randn(parameter.shape, generator=generator).to(parameter)
    optimizer.step()


def _assert_state_equal(first, second):
    assert first.keys() == second.keys()
    for key, value in first.items():
        other = second[key]
        if isinstance(value, QuantState):
            assert isinstance(other, QuantState)
            assert value.shape == other.shape
            assert value.dtype == other.dtype
            assert value.blocksize == other.blocksize
            assert value.packed == other.packed
            torch.testing.assert_close(value.absmax, other.absmax, rtol=0, atol=0)
        elif torch.is_tensor(value):
            torch.testing.assert_close(value, other, rtol=0, atol=0)
        else:
            assert value == other


@pytest.mark.parametrize("optimizer_class", RECIPES)
@pytest.mark.parametrize(
    "device,dtype,numel",
    [
        ("cpu", dtype, numel)
        for dtype in (torch.float32, torch.bfloat16, torch.float16)
        for numel in (768, 4096)
    ] + [pytest.param("cuda", torch.bfloat16, 4096, marks=CUDA)],
)
def test_serialized_checkpoint_resumes_exactly(optimizer_class, device, dtype, numel):
    parameter = torch.nn.Parameter(torch.ones(numel, device=device, dtype=dtype))
    optimizer = optimizer_class([parameter], quant_rng_seed=43)
    optimizer.set_m1_quant_scheme_for_parameters([parameter], "nf4_sr")
    for step in range(3):
        _step(optimizer, parameter, step)

    buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location="cpu")
    if numel == 4096:
        assert isinstance(checkpoint["state"][0]["m1_quant_state"], dict)
        assert isinstance(optimizer.state[parameter]["m1_quant_state"], QuantState)

    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = optimizer_class([resumed_parameter], quant_rng_seed=43)
    resumed.set_m1_quant_scheme_for_parameters([resumed_parameter], "nf4_sr")
    resumed.load_state_dict(checkpoint)
    # This also checks the original FP32 values, before another step can mask
    # precision lost by casting the checkpoint through BF16/FP16.
    _assert_state_equal(optimizer.state[parameter], resumed.state[resumed_parameter])

    for step in range(3, 5):
        _step(optimizer, parameter, step)
        _step(resumed, resumed_parameter, step)
        torch.testing.assert_close(parameter, resumed_parameter, rtol=0, atol=0)
        _assert_state_equal(optimizer.state[parameter], resumed.state[resumed_parameter])


@pytest.mark.parametrize("optimizer_class", RECIPES)
def test_legacy_quant_state_objects_still_load(optimizer_class):
    parameter = torch.nn.Parameter(torch.ones(4096, dtype=torch.bfloat16))
    optimizer = optimizer_class([parameter], quant_rng_seed=43)
    _step(optimizer, parameter, 0)
    checkpoint = copy.deepcopy(optimizer.state_dict())
    checkpoint["state"][0] = copy.deepcopy(optimizer.state[parameter])
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    # Old pickle files need a scoped allowlist when reading trusted checkpoints.
    with torch.serialization.safe_globals([QuantState]):
        checkpoint = torch.load(buffer)

    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = optimizer_class([resumed_parameter], quant_rng_seed=43)
    resumed.load_state_dict(checkpoint)
    _assert_state_equal(optimizer.state[parameter], resumed.state[resumed_parameter])
    _step(optimizer, parameter, 1)
    _step(resumed, resumed_parameter, 1)
    torch.testing.assert_close(parameter, resumed_parameter, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("optimizer_class", RECIPES)
def test_cpu_checkpoint_moves_quantization_scales_to_cuda(optimizer_class):
    parameter = torch.nn.Parameter(torch.ones(4096))
    optimizer = optimizer_class([parameter])
    _step(optimizer, parameter, 0)
    checkpoint = copy.deepcopy(optimizer.state_dict())
    resumed_parameter = torch.nn.Parameter(parameter.detach().cuda())
    resumed = optimizer_class([resumed_parameter])
    resumed.load_state_dict(checkpoint)

    for moment in ("m1", "m2"):
        state = resumed.state[resumed_parameter]
        assert state[f"{moment}_code"].device == resumed_parameter.device
        scales = state[f"{moment}_quant_state"].absmax
        assert scales.device == resumed_parameter.device
        assert scales.dtype == torch.float32
        torch.testing.assert_close(
            scales.cpu(), checkpoint["state"][0][f"{moment}_quant_state"]["absmax"],
            rtol=0, atol=0,
        )
    _step(resumed, resumed_parameter, 1)
    assert torch.isfinite(resumed_parameter).all()


def test_load_hooks_see_adapted_and_restored_state():
    small = torch.nn.Parameter(torch.ones(768, dtype=torch.bfloat16))
    large = torch.nn.Parameter(torch.ones(4096, dtype=torch.float16))
    optimizer = ZEEDENAdamW4Bit([{"params": [small]}, {"params": [large]}])
    small.grad = torch.ones_like(small)
    _step(optimizer, large, 0)
    checkpoint = copy.deepcopy(optimizer.state_dict())
    expected = torch.full((768,), 0.123456789)
    expected_scale = torch.full((32,), 0.234567891)
    calls = []

    def adapt(_optimizer, state_dict):
        calls.append("pre")
        adapted = copy.deepcopy(state_dict)
        adapted["state"][0]["m1"] = expected.clone()
        adapted["state"][1]["m1_quant_state"]["absmax"] = expected_scale.clone()
        return adapted

    def inspect(loaded_optimizer):
        calls.append("post")
        small_state = loaded_optimizer.state[small]
        large_state = loaded_optimizer.state[large]
        torch.testing.assert_close(small_state["m1"], expected, rtol=0, atol=0)
        assert large_state["m1_code"].dtype == torch.uint8
        assert isinstance(large_state["m1_quant_state"], QuantState)
        torch.testing.assert_close(
            large_state["m1_quant_state"].absmax, expected_scale, rtol=0, atol=0,
        )
        loaded_optimizer.param_groups[0]["lr"] = 0.007

    pre_hook = optimizer.register_load_state_dict_pre_hook(adapt)
    post_hook = optimizer.register_load_state_dict_post_hook(inspect)
    try:
        # PyTorch validates the groups; temporary repair hooks must also be
        # removed if validation fails, before a subsequent valid load.
        invalid = copy.deepcopy(checkpoint)
        invalid["param_groups"] = []
        with pytest.raises(ValueError, match="parameter groups"):
            optimizer.load_state_dict(invalid)
        calls.clear()
        optimizer.load_state_dict(checkpoint)
    finally:
        pre_hook.remove()
        post_hook.remove()
    assert calls == ["pre", "post"]
    assert optimizer.param_groups[0]["lr"] == 0.007


def test_ze_eden_zero_initialization_matches_adamw_first_step():
    parameter = torch.nn.Parameter(torch.zeros(4096))
    reference = torch.nn.Parameter(parameter.detach().clone())
    optimizer = ZEEDENAdamW4Bit([parameter], lr=1e-3, weight_decay=0.0)
    adamw = torch.optim.AdamW(
        [reference], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
    )
    parameter.grad = torch.full_like(parameter, 1e-8)
    reference.grad = parameter.grad.clone()
    optimizer.step()
    adamw.step()
    torch.testing.assert_close(parameter, reference, rtol=1e-6, atol=0)
    torch.testing.assert_close(parameter, torch.full_like(parameter, -0.0005), rtol=1e-6, atol=0)
