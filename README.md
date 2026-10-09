# adamw4bit

`adamw4bit` provides PyTorch implementations of ZIP-SR and ZE-EDEN for
4-bit AdamW moment storage. Both use NF4 for the first moment. ZIP-SR uses
zero-inclusive Dyn4 with stochastic rounding in preconditioner space for the
second moment; ZE-EDEN uses zero-exclusive Dyn4 with EDEN scale calibration.
The methods are described in *Rounding in Preconditioner Space: Redesigning
4-bit AdamW Optimizer-State Quantization*.

## Installation

Requires Python 3.11 or 3.12 and PyTorch 2.13 or newer.

```bash
pip install .           # the optimizer
pip install '.[all]'    # plus the Hugging Face dependencies used by the examples
```

## Usage

Use `ZEEDENAdamW4Bit` or `ZIPSRAdamW4Bit` like any PyTorch optimizer:

```python
import torch

from adamw4bit import ZEEDENAdamW4Bit, ZIPSRAdamW4Bit

model = torch.nn.Linear(64, 64)
optimizer = ZEEDENAdamW4Bit(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)
```

For ZIP-SR, use `ZIPSRAdamW4Bit(model.parameters(), lr=1e-3)` instead.
Both constructors select the method's first- and second-moment quantizers and
EDEN setting. Learning rate, betas, epsilon and weight decay remain configurable.

`ZIPSRAdamW4Bit` and `ZEEDENAdamW4Bit` are thin subclasses of `QuantizedAdamW`, the
shared implementation for 4-bit, 8-bit and FP32 moment storage. For ablations,
use `QuantizedAdamW` with explicit `m1_quant_scheme`, `m2_quant_scheme` and
`use_eden_m2` settings. Its defaults use 8-bit linear quantization;
`AdamW8bit` remains a compatibility wrapper with its original defaults.

Four-bit blocks default to 128 values; tensors with fewer than 4,096 values or
sizes not divisible by the block size remain in FP32. This quantizes optimizer
moments, not model weights or gradients.

The paper's complete recipes also switch only the LM-head first moment to
NF4 stochastic rounding during the final 10% of training. The training loop
must identify the head and apply this schedule before each optimizer update:

```python
boundary = round(0.9 * total_optimizer_steps)
head = model.get_output_embeddings()
scheme = "nf4_sr" if completed_optimizer_steps >= boundary else "nf4"
optimizer.set_m1_quant_scheme_for_parameters(head.parameters(), scheme)
```

For 6,179 updates, SR begins at update 5,562. Tied embeddings share the same
parameter, so use untied weights for a head-only switch. The examples below
keep first-moment RTN throughout and demonstrate the quantization settings;
they do not implement the full paper training protocol.

## Optional memory optimization

The reference implementation remains the default. Enable the bounded-workspace
backend explicitly:

```python
optimizer = ZEEDENAdamW4Bit(model.parameters(), optimized=True)
# ZIPSRAdamW4Bit and QuantizedAdamW accept the same flag.
```

For supported contiguous tensors, the backend updates packed moments in place
and processes temporary FP32 values in chunks of at most 1,048,576 elements.
Chunk boundaries preserve quantization blocks and packed-code boundaries.
Persistent moments, parameters, and gradients still scale with model size.
CPU uses PyTorch operations; supported FP32 CUDA rounding is compiled lazily
with `torch.compile`, so the first update includes compilation time.

Eight-bit schemes, noncontiguous tensors, telemetry, recorded diagnostics,
optimizer-level update clipping, unsupported research read variants, and
incompatible or overlapping state storage use the reference update. Those fallbacks can require full-tensor
workspace. Ordinary gradient clipping before `optimizer.step()` is unaffected.

This option prioritizes peak optimizer memory. The current chunked implementation
can be slower than the reference, and optimizer memory savings may not reduce a
training peak dominated by activations. Measure both memory and time on your
workload.

Chunking and compiled arithmetic can change floating-point results and stochastic
rounding samples. Resume with the same `optimized` setting and quantization seed;
use matching settings across distributed replicas. Optimized checkpoints retain
their chunk and fallback policy. A reference checkpoint loaded with
`optimized=True` adopts the current chunk limit. With `quant_rng_seed=None`, also
save and restore the global PyTorch RNG state as part of the training checkpoint.

## Checkpoints

Save optimizer state alongside model state. New optimizer checkpoints load with
PyTorch's restricted loader:

```python
torch.save(optimizer.state_dict(), "optimizer.pt")
optimizer.load_state_dict(torch.load("optimizer.pt", weights_only=True))
```

For a trusted older `adamw4bit` checkpoint containing `QuantState` objects, use
a scoped allowlist, then save again to use the current format:

```python
from adamw4bit.quantization import QuantState

with torch.serialization.safe_globals([QuantState]):
    state = torch.load("optimizer.pt", weights_only=True)
optimizer.load_state_dict(state)
torch.save(optimizer.state_dict(), "optimizer.pt")
```

## Paper reproduction

ZE-EDEN starts with an exactly zero second moment, matching the paper's
initialization. This corrects the historical implementation's approximately
`3.25e-15` initial value on eligible tensors and can change training trajectories.
For the six smaller pretraining sizes, FP32/TorchAO use the final scheduled
validation while ZE-EDEN/ZIP-SR use terminal validation, so their evaluation
steps differ slightly.
The corrected Qwen3-8B SFT runs use separate LM-head/non-head optimizer groups
under ZeRO-2; comparisons to earlier one-group runs retain that topology difference.

## Examples

Run from the repository root after `pip install '.[all]'`. Each script accepts
`--recipe` (`ze-eden` or `zip-sr`; the GPT-small script also accepts `fp32`).

- `examples/quickstart.py` trains a tiny GPT-2 on synthetic tokens and does not
  download anything:

  ```bash
  python examples/quickstart.py --recipe zip-sr
  ```

- `examples/train_gpt_small.py` trains GPT-small on FineWeb-Edu with the
  Hugging Face Trainer. Use `--smoke-test` for a reduced run, or launch the full
  run on eight GPUs:

  ```bash
  python examples/train_gpt_small.py --recipe ze-eden --smoke-test
  torchrun --standalone --nproc-per-node=8 \
    examples/train_gpt_small.py --recipe ze-eden --seed 42
  ```

- `examples/run_gpt_small_experiment.sh` runs the full paired comparison of all
  three recipes over three seeds. See [`examples/README.md`](examples/README.md).

## Development

```bash
uv sync
uv run pytest
```
## Status
Security fixes are provided for the
latest release, as described in [`SECURITY.md`](SECURITY.md). Questions and
non-security bugs belong in GitHub issues. The project is maintained by
[Nubank](https://nubank.com.br).

## License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE).
