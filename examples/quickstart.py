#!/usr/bin/env python3
"""Train a tiny, locally initialized GPT-2 with a quantized AdamW optimizer."""

import argparse
import json

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from adamw4bit import QuantizedAdamW, ZEEDENAdamW4Bit, ZIPSRAdamW4Bit


def build_optimizer(
    model: torch.nn.Module,
    recipe: str,
) -> QuantizedAdamW:
    """Create one of the 4-bit optimizer configurations from the paper."""
    if recipe == "ze-eden":
        optimizer_class = ZEEDENAdamW4Bit
    elif recipe == "zip-sr":
        optimizer_class = ZIPSRAdamW4Bit
    else:
        raise ValueError(f"unsupported recipe: {recipe}")

    return optimizer_class(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )


def run_quickstart(
    recipe: str = "ze-eden",
    *,
    steps: int = 3,
    device: str = "cpu",
) -> dict[str, bool | float | int | str]:
    """Run an offline optimizer smoke test on synthetic token IDs."""
    if steps <= 0:
        raise ValueError("steps must be positive")

    torch.manual_seed(42)
    resolved_device = torch.device(device)
    config = GPT2Config(
        vocab_size=128,
        n_positions=64,
        n_embd=64,
        n_layer=2,
        n_head=4,
        n_inner=256,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        tie_word_embeddings=False,
        use_cache=False,
        loss_type="ForCausalLM",
    )
    model = GPT2LMHeadModel(config).to(resolved_device)
    model.loss_type = "ForCausalLM"
    optimizer = build_optimizer(model, recipe)
    initial_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    model.train()
    for _ in range(steps):
        input_ids = torch.randint(0, config.vocab_size, (2, 32)).to(resolved_device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids, labels=input_ids).loss
        loss.backward()
        optimizer.step()

    if not torch.isfinite(loss):
        raise RuntimeError("training produced a non-finite loss")
    parameters_updated = any(
        not torch.equal(initial, parameter.detach())
        for initial, parameter in zip(
            initial_parameters,
            model.parameters(),
            strict=True,
        )
    )
    if not parameters_updated:
        raise RuntimeError("optimizer did not update model parameters")

    packed_states = sum(
        int(bool(getattr(value, "packed", False)))
        for state in optimizer.state.values()
        for key, value in state.items()
        if key.endswith("_quant_state")
    )
    return {
        "recipe": recipe,
        "steps": steps,
        "final_loss": float(loss.detach()),
        "parameters_updated": parameters_updated,
        "packed_state_tensors": packed_states,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        choices=("ze-eden", "zip-sr"),
        default="ze-eden",
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(
        json.dumps(
            run_quickstart(args.recipe, steps=args.steps, device=args.device),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
