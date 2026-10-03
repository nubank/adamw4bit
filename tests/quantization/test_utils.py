import pytest
import torch

from adamw4bit.quantization import (
    _STRATEGIES,
    _block,
    codes_for_dequantization,
    default_block_size,
    dequantize,
    pack_4bit_codes,
    pack_codes_for_storage,
    quantize,
    quantize_with_dequantized,
    unpack_4bit_codes,
)


FOUR_BIT_SCHEMES = [
    "sdyn4",
    "nf4",
    "nf4_sr",
    "lin4",
    "lin4_sr",
    "lin4_upd_sr",
    "lin4_upd_sr_store_eden",
    "lin4_upd_sr_fp32read",
    "lin4_nz",
    "dyn4",
    "dyn4_sr",
    "dyn4_upd_sr",
    "dyn4_upd_sr_store_eden",
    "dyn4_upd_sr_fp32read",
    "dyn4_lookahead_sr",
    "dyn4_upd_sr_fp32read_nextbc",
    "dyn4_proxy_la_upd_sr",
    "dyn4_vproxy_la_upd_sr",
    "dyn4_nz",
]


def test_nf4_sr_reuses_nf4_codebook_and_is_reproducible() -> None:
    values = torch.linspace(-1.0, 1.0, 256)
    nf4_codes, nf4_state = quantize(
        values,
        block_size=128,
        mode="nf4",
    )
    nf4_values = dequantize(
        nf4_codes,
        nf4_state,
        mode="nf4",
    )
    nf4_sr_values = dequantize(
        nf4_codes,
        nf4_state,
        mode="nf4_sr",
    )
    torch.testing.assert_close(nf4_sr_values, nf4_values)

    first_codes, _ = quantize(
        values,
        block_size=128,
        mode="nf4_sr",
        generator=torch.Generator().manual_seed(123),
    )
    second_codes, _ = quantize(
        values,
        block_size=128,
        mode="nf4_sr",
        generator=torch.Generator().manual_seed(123),
    )
    torch.testing.assert_close(first_codes, second_codes)


@pytest.mark.parametrize(
    "codes",
    [
        torch.tensor([], dtype=torch.uint8),
        torch.tensor([0], dtype=torch.uint8),
        torch.tensor([1, 2, 15], dtype=torch.uint8),
        torch.arange(16, dtype=torch.uint8),
    ],
)
def test_pack_4bit_codes_roundtrip(codes: torch.Tensor) -> None:
    packed = pack_4bit_codes(codes)
    unpacked = unpack_4bit_codes(packed, numel=codes.numel())

    torch.testing.assert_close(unpacked, codes)
    assert packed.numel() == (codes.numel() + 1) // 2


def test_pack_4bit_codes_uses_low_then_high_nibble_order() -> None:
    codes = torch.tensor([1, 2, 15], dtype=torch.uint8)

    packed = pack_4bit_codes(codes)

    torch.testing.assert_close(
        packed,
        torch.tensor([0x21, 0x0F], dtype=torch.uint8),
    )


@pytest.mark.parametrize("mode", ["nf4", "dyn4_nz"])
def test_dequantize_packed_codes_matches_legacy_unpacked_state(mode: str) -> None:
    x = torch.linspace(-1.0, 1.0, 256).abs() if mode == "dyn4_nz" else torch.linspace(-1.0, 1.0, 256)
    codes, quant_state = quantize(x, block_size=128, mode=mode)
    expected = dequantize(codes, quant_state, mode=mode)

    packed, packed_state = pack_codes_for_storage(
        codes,
        quant_state,
        mode=mode,
    )
    actual = dequantize(packed, packed_state, mode=mode)

    assert packed_state.packed is True
    assert packed.numel() == codes.numel() // 2
    torch.testing.assert_close(actual, expected)


def test_legacy_unpacked_codes_remain_readable() -> None:
    x = torch.linspace(0.0, 1.0, 128)
    codes, quant_state = quantize(x, block_size=128, mode="dyn4_nz")

    logical_codes = codes_for_dequantization(
        codes,
        quant_state,
        mode="dyn4_nz",
    )

    assert quant_state.packed is False
    torch.testing.assert_close(logical_codes, codes)


def test_8bit_codes_are_not_packed_for_storage() -> None:
    x = torch.linspace(-1.0, 1.0, 256)
    codes, quant_state = quantize(x, block_size=256, mode="dynamic")

    stored_codes, stored_state = pack_codes_for_storage(
        codes,
        quant_state,
        mode="dynamic",
    )

    assert stored_state.packed is False
    assert stored_codes.data_ptr() == codes.data_ptr()
    assert stored_codes.numel() == x.numel()


def test_block_uses_view_when_tensor_is_block_aligned() -> None:
    x = torch.arange(256, dtype=torch.float32)

    blocked, shape = _block(x, block_size=128)

    assert shape == x.shape
    assert blocked.shape == (2, 128)
    assert blocked.data_ptr() == x.data_ptr()


@pytest.mark.parametrize("mode", FOUR_BIT_SCHEMES)
def test_all_4bit_schemes_default_to_128_element_blocks(mode: str) -> None:
    assert default_block_size(mode) == 128


def test_dyn4_nz_floors_zero_code_to_smallest_positive_code() -> None:
    x = torch.tensor([0.0, 1e-6, 0.01, 1.0], dtype=torch.float32)

    codes, _ = quantize(x, block_size=4, mode="dyn4_nz")

    assert int(codes.min().item()) >= 1
    codebook = _STRATEGIES["dyn4_nz"]._get_codebook(torch.device("cpu"))  # type: ignore[attr-defined]
    assert codebook[0].item() == 0.0
    assert codebook[1].item() > 0.0


def test_lin4_nz_matches_torchao_unsigned_4bit_qmap() -> None:
    codebook = _STRATEGIES["lin4_nz"]._get_codebook(torch.device("cpu"))  # type: ignore[attr-defined]
    expected = torch.linspace(0.0, 1.0, 17, dtype=torch.float32)[1:]

    torch.testing.assert_close(codebook, expected)


def test_sdyn4_matches_torchao_signed_4bit_qmap() -> None:
    codebook = _STRATEGIES["sdyn4"]._get_codebook(torch.device("cpu"))  # type: ignore[attr-defined]
    expected = torch.tensor(
        [
            -0.8875, -0.6625, -0.4375, -0.2125,
            -0.0775, -0.0325, -0.0055, 0.0,
            0.0055, 0.0325, 0.0775, 0.2125,
            0.4375, 0.6625, 0.8875, 1.0,
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(codebook, expected)


def test_dyn4_matches_bnb_style_unsigned_dynamic_4bit_qmap() -> None:
    codebook = _STRATEGIES["dyn4"]._get_codebook(torch.device("cpu"))  # type: ignore[attr-defined]
    expected = torch.tensor(
        [
            0.0, 0.00325, 0.00775, 0.02125,
            0.04375, 0.06625, 0.08875, 0.15625,
            0.26875, 0.38125, 0.49375, 0.60625,
            0.71875, 0.83125, 0.94375, 1.0,
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(codebook, expected)


@pytest.mark.parametrize(
    ("fp32read_mode", "base_mode"),
    [
        ("lin4_upd_sr_fp32read", "lin4_upd_sr"),
        ("dyn4_upd_sr_fp32read", "dyn4_upd_sr"),
        ("dyn4_lookahead_sr", "dyn4_upd_sr"),
        ("dyn4_upd_sr_fp32read_nextbc", "dyn4_upd_sr"),
    ],
)
def test_upd_sr_fp32read_uses_matching_family_codebook(
    fp32read_mode: str,
    base_mode: str,
) -> None:
    device = torch.device("cpu")
    fp32read_codebook = _STRATEGIES[fp32read_mode]._get_codebook(device)  # type: ignore[attr-defined]
    base_codebook = _STRATEGIES[base_mode]._get_codebook(device)  # type: ignore[attr-defined]

    torch.testing.assert_close(fp32read_codebook, base_codebook)


def test_torchao_aligned_4bit_schemes_round_midpoint_ties_up() -> None:
    values = torch.tensor([0.09375, 0.96875, 1.0, 0.0], dtype=torch.float32)

    lin_codes, _ = quantize(values, block_size=4, mode="lin4_nz")

    assert lin_codes.tolist() == [1, 15, 15, 0]

    signed_values = torch.tensor([0.00275, -0.00275, 1.0, 0.0], dtype=torch.float32)
    signed_codes, _ = quantize(signed_values, block_size=4, mode="sdyn4")

    assert signed_codes.tolist() == [8, 7, 15, 7]


@pytest.mark.parametrize("mode", FOUR_BIT_SCHEMES)
def test_custom_4bit_schemes_use_torchao_scale_floor(mode: str) -> None:
    _, state = quantize(torch.zeros(128, dtype=torch.float32), block_size=128, mode=mode)

    torch.testing.assert_close(state.absmax, torch.tensor([1e-12], dtype=torch.float32))


@pytest.mark.parametrize("mode", ["linear", "dynamic"])
def test_8bit_schemes_keep_legacy_scale_floor(mode: str) -> None:
    _, state = quantize(torch.zeros(256, dtype=torch.float32), block_size=256, mode=mode)

    torch.testing.assert_close(state.absmax, torch.tensor([1e-8], dtype=torch.float32))


@pytest.mark.parametrize(
    "mode",
    ["lin4_upd_sr", "dyn4_upd_sr", "lin4_upd_sr_store_eden", "dyn4_upd_sr_store_eden"],
)
def test_quantize_with_dequantized_matches_quantize_then_dequantize(mode: str) -> None:
    x = torch.linspace(0.0, 0.5, 17, dtype=torch.float32)
    kwargs = {
        "block_size": 8,
        "mode": mode,
        "bias_correction": 0.25,
        "eps": 1e-8,
    }

    codes, state = quantize(x, **kwargs, generator=torch.Generator().manual_seed(123))
    expected = dequantize(codes, state, mode=mode)

    helper_codes, helper_state, helper_dequantized = quantize_with_dequantized(
        x,
        **kwargs,
        generator=torch.Generator().manual_seed(123),
    )

    torch.testing.assert_close(helper_codes, codes)
    torch.testing.assert_close(helper_state.absmax, state.absmax)
    assert helper_state.shape == state.shape
    assert helper_state.dtype == state.dtype
    assert helper_state.blocksize == state.blocksize
    torch.testing.assert_close(helper_dequantized, expected)
