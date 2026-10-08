# Optimizer memory and time

Compare the reference and opt-in implementations from the same checkout on a
CUDA device. The benchmark uses synthetic parameters and preallocated gradients;
it does not download models or data.

```bash
python benchmarks/profile_memory.py \
  --source . --cases 40m model --methods ze-eden zip-sr \
  --output ../benchmark-results/reference.json

python benchmarks/profile_memory.py \
  --source . --cases 40m model --methods ze-eden zip-sr --optimized \
  --output ../benchmark-results/optimized.json
```

Without `--optimized`, the optimizer uses its unchanged reference default.
The flag applies only to library optimizers. Native controls are available as
`adamw-single`, `adamw-default`, and `adamw-foreach`; `fp32-state` uses the library
with FP32 moments. Use `--dry-run` to inspect case sizes without initializing CUDA.

Cases `1m`, `16m`, and `40m` contain one tensor with that many multiples of
1,048,576 elements. `model` contains 100 GPT-small-shaped tensors, with
162,167,808 parameters and two large vocabulary matrices. `many` contains
128 matrices of 4,096 elements and 24 smaller fallback tensors. These cases
separate total moment storage from workspace driven by the largest tensor and
per-parameter dispatch overhead. They are not full training workloads.

Each case and method runs in a fresh subprocess. JSON records source and script
hashes, runtime versions, backend selection, tensor counts, and persistent state
bytes. FP32 parameters and gradients are the default. With lower-precision
parameters, native AdamW follows parameter dtype while this library keeps FP32
working/fallback moments; that is not an FP32-state versus 4-bit-state comparison.

## Interpreting measurements

The first step is measured separately, including state initialization and any
CUDA compilation. Warmup precedes repeated steady measurements. CUDA events and
synchronized wall time are reported separately. Input/gradient creation is
outside measurement. Timing covers optimizer updates only.

All JSON memory values are bytes; the text summary uses MiB (1,048,576 bytes).
Persistent state counts unique backing storage, including block scales. Peaks
include live parameters, gradients, state, and temporary buffers. The transient
excess metric subtracts the larger of pre-step and post-step residency; it is
not an exact buffer attribution. Reserved allocator memory is reported separately
from live allocated memory. CUDA context and other driver allocations are not
included in PyTorch's allocated-memory counters.

For eligible tensors with 128-element blocks, both packed moments require
1.0625 bytes per parameter versus 8 bytes for FP32 moments. With FP32 parameters
and gradients, ideal resident storage is 9.0625 versus 16 bytes per parameter,
before fallback tensors, allocator overhead, and workspace. A lower optimizer
peak does not guarantee a lower forward/backward peak or faster training.

Add `--head-sr` to use NF4 stochastic rounding on the final eligible tensor from
the first benchmark step. This measures the SR policy, not the paper's late
training switch. Native and FP32-state controls are unaffected.

## Optional profiling

Add `--profile-dir ../benchmark-results/traces` to profile a separate step after
the ordinary measurements. Profiling is excluded from reported timing and peak
measurements. Allocation totals do not represent simultaneously live memory;
use allocation lifetimes when attributing peaks. Keep raw traces outside the
source tree because they can contain local paths and runtime details.
