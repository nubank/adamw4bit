#!/usr/bin/env python3
"""Compare FP32 AdamW, ZE-EDEN, and ZIP-SR on FineWeb-Edu with Hugging Face.

This example uses GPT-2 position embeddings, individually truncated/padded
text documents, and first-moment RTN throughout. It does not reproduce the
paper's full training protocol. Use --smoke-test for a reduced local run.
"""

import argparse
import gc
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    set_seed,
)

from adamw4bit import ZEEDENAdamW4Bit, ZIPSRAdamW4Bit


DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
TOKENIZER_NAME = "gpt2"
SEED = 42
TRAIN_SAMPLES = 1_582_031
VALIDATION_SAMPLES = 50_000
SEQUENCE_LENGTH = 2_048
TRAINING_STEPS = 6_179
PER_DEVICE_BATCH_SIZE = 32


def build_model(*, smoke_test: bool) -> GPT2LMHeadModel:
    """Build the GPT-small shape, or a reduced local smoke-test shape."""
    config = GPT2Config(
        vocab_size=50_257,
        n_positions=128 if smoke_test else SEQUENCE_LENGTH,
        n_embd=128 if smoke_test else 768,
        n_layer=2 if smoke_test else 12,
        n_head=4 if smoke_test else 12,
        n_inner=512 if smoke_test else 3_072,
        activation_function="gelu",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=50_256,
        eos_token_id=50_256,
        pad_token_id=50_256,
        tie_word_embeddings=False,
        use_cache=False,
        loss_type="ForCausalLM",
    )
    model = GPT2LMHeadModel(config)
    model.loss_type = "ForCausalLM"
    return model


def build_optimizer(
    model: torch.nn.Module,
    recipe: str,
    *,
    learning_rate: float = 1e-3,
) -> torch.optim.Optimizer:
    """Create fp32 AdamW, ZE-EDEN, or ZIP-SR."""
    common = {
        "lr": learning_rate,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.1,
    }
    if recipe == "fp32":
        return torch.optim.AdamW(model.parameters(), **common)
    if recipe == "ze-eden":
        return ZEEDENAdamW4Bit(model.parameters(), **common)
    if recipe == "zip-sr":
        return ZIPSRAdamW4Bit(model.parameters(), **common)
    raise ValueError(f"unsupported recipe: {recipe}")


def build_datasets(tokenizer, *, smoke_test: bool, seed: int):
    """Select and tokenize the same public FineWeb-Edu source dataset."""
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
        streaming=True,
    )
    validation_samples = 16 if smoke_test else VALIDATION_SAMPLES
    train_samples = 64 if smoke_test else TRAIN_SAMPLES
    sequence_length = 128 if smoke_test else SEQUENCE_LENGTH

    validation = dataset.take(validation_samples)
    train = dataset.skip(validation_samples).take(train_samples)
    train = train.shuffle(seed=seed, buffer_size=min(10_000, train_samples))

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=sequence_length,
        )

    remove_columns = dataset.column_names
    return (
        train.map(tokenize, batched=True, remove_columns=remove_columns),
        validation.map(tokenize, batched=True, remove_columns=remove_columns),
    )


def wsd_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """10% warmup, stable phase, then the original 10% 1-sqrt cooldown."""
    warmup_steps = max(1, round(0.1 * training_steps))
    cooldown_start = round(0.9 * training_steps)

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        if step < cooldown_start:
            return 1.0
        progress = min(
            1.0,
            (step - cooldown_start) / max(1, training_steps - cooldown_start),
        )
        return 1.0 - math.sqrt(progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        choices=("fp32", "ze-eden", "zip-sr"),
        default="ze-eden",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=PER_DEVICE_BATCH_SIZE,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gpt-small"))
    parser.add_argument(
        "--result-file",
        type=Path,
        help="Write the rank-zero run summary as JSON for experiment aggregation.",
    )
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.per_device_batch_size <= 0:
        parser.error("--per-device-batch-size must be positive")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient-accumulation-steps must be positive")

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset, validation_dataset = build_datasets(
        tokenizer,
        smoke_test=args.smoke_test,
        seed=args.seed,
    )
    model = build_model(smoke_test=args.smoke_test)
    optimizer = build_optimizer(
        model,
        args.recipe,
        learning_rate=args.learning_rate,
    )
    training_steps = args.max_steps or (
        2 if args.smoke_test else TRAINING_STEPS
    )
    scheduler = wsd_scheduler(optimizer, training_steps=training_steps)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir / args.recipe,
            max_steps=training_steps,
            per_device_train_batch_size=(
                2 if args.smoke_test else args.per_device_batch_size
            ),
            per_device_eval_batch_size=(
                2 if args.smoke_test else args.per_device_batch_size
            ),
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=0.1,
            max_grad_norm=1.0,
            bf16=not args.smoke_test,
            eval_strategy="steps",
            eval_steps=max(1, round(0.05 * training_steps)),
            logging_steps=1 if args.smoke_test else 20,
            save_strategy="no",
            report_to="none",
            seed=args.seed,
            data_seed=args.seed,
            ddp_find_unused_parameters=False,
        ),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer,
        optimizers=(optimizer, scheduler),
    )
    train_result = trainer.train()
    validation = trainer.evaluate()
    if trainer.is_world_process_zero():
        result = {
            "recipe": args.recipe,
            "seed": args.seed,
            "steps": training_steps,
            "train_loss": train_result.training_loss,
            "validation_loss": validation["eval_loss"],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "dataset": f"{DATASET_NAME}/{DATASET_CONFIG}",
            "tokenizer": TOKENIZER_NAME,
            "sequence_length": 128 if args.smoke_test else SEQUENCE_LENGTH,
            "world_size": trainer.args.world_size,
            "learning_rate": args.learning_rate,
            "per_device_batch_size": (
                2 if args.smoke_test else args.per_device_batch_size
            ),
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        }
        rendered_result = json.dumps(result, indent=2)
        print(rendered_result)
        if args.result_file is not None:
            args.result_file.parent.mkdir(parents=True, exist_ok=True)
            temporary_result = args.result_file.with_suffix(
                f"{args.result_file.suffix}.tmp"
            )
            temporary_result.write_text(f"{rendered_result}\n")
            temporary_result.replace(args.result_file)
    trainer.accelerator.end_training()


if __name__ == "__main__":
    main()
    # Release streaming dataset cycles before Arrow/Python interpreter shutdown.
    gc.collect()
