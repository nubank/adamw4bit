# GPT-small OSS comparison

This experiment checks that the public `adamw4bit` library behaves as intended
in a Hugging Face GPT-small training run. It compares full-precision AdamW,
ZE-EDEN, and ZIP-SR using paired seeds.

This is a relative optimizer comparison using GPT-2 position embeddings,
individually truncated/padded FineWeb-Edu documents, and first-moment RTN
throughout. The paper uses a different model and data pipeline, and switches
only the LM-head first moment to stochastic rounding for the final 10% of
training. This example does not reproduce the paper's full training protocol;
compare paired results within this experiment rather than against Table 3.

## Configuration

The default run uses:

- recipes: `fp32`, `ze-eden`, and `zip-sr`;
- paired seeds: `42`, `43`, and `44`;
- one node with 8 GPUs;
- per-device batch size 32 and gradient accumulation 1;
- learning rate `1e-3`;
- 6,179 optimizer steps;
- FineWeb-Edu `sample-10BT`, GPT-2 tokenizer, and sequence length 2,048;
- bf16 mixed precision and the WSD schedule used by the training example.

Install the package and Hugging Face dependencies in the cluster environment:

```bash
pip install '.[all]'
```

Run commands from the repository root.

## Check the launch commands

Print every command without starting training:

```bash
DRY_RUN=1 ./examples/run_gpt_small_experiment.sh
```

The output should contain nine runs: three recipes for each of the three seeds.

## Smoke test

Use a separate output directory so smoke-test results cannot be mistaken for
full-run results:

```bash
SMOKE_TEST=1 \
NPROC_PER_NODE=1 \
OUTPUT_DIR=outputs/gpt-small-smoke \
./examples/run_gpt_small_experiment.sh
```

## Full experiment

On one 8-GPU node:

```bash
./examples/run_gpt_small_experiment.sh
```

To place artifacts on persistent cluster storage:

```bash
OUTPUT_DIR=/path/to/persistent-storage/gpt-small-oss \
./examples/run_gpt_small_experiment.sh
```

The launcher runs jobs sequentially. Completed result files are reused, making
it safe to restart the same experiment after interruption. Set `OVERWRITE=1`
to rerun every selected recipe and seed.

## Overrides

Configuration is passed through environment variables:

```bash
SEEDS="42 43 44" \
RECIPES="fp32 ze-eden zip-sr" \
NPROC_PER_NODE=8 \
LEARNING_RATE=0.001 \
PER_DEVICE_BATCH_SIZE=32 \
GRADIENT_ACCUMULATION_STEPS=1 \
OUTPUT_DIR=/path/to/persistent-storage/gpt-small-oss \
./examples/run_gpt_small_experiment.sh
```

Additional controls:

- `PYTHON`: Python executable; default `python`.
- `MAX_STEPS`: optional positive step limit.
- `SMOKE_TEST=1`: use the reduced model and dataset.
- `DRY_RUN=1`: print commands only.
- `OVERWRITE=1`: replace existing selected results.

Use a new `OUTPUT_DIR`, or set `OVERWRITE=1`, whenever training parameters
change. Otherwise the launcher will reuse existing result files for matching
recipe and seed names.

## Output layout

With the default output directory:

```text
outputs/gpt-small-experiment/
├── logs/
│   ├── fp32-seed42.log
│   ├── ze-eden-seed42.log
│   └── ...
├── results/
│   ├── fp32-seed42.json
│   ├── ze-eden-seed42.json
│   ├── zip-sr-seed42.json
│   └── ...
└── runs/
    ├── seed-42/
    ├── seed-43/
    └── seed-44/
```

Each result JSON contains the recipe, seed, step count, train loss, validation
loss, parameter count, dataset, tokenizer, world size, learning rate, batch
size, and gradient accumulation.

## Gather results

Install `jq` if it is not already available. Set the artifact location used by
the experiment:

```bash
OUTPUT_DIR=${OUTPUT_DIR:-outputs/gpt-small-experiment}
```

Confirm that all nine result files exist:

```bash
ls "${OUTPUT_DIR}"/results/{fp32,ze-eden,zip-sr}-seed{42,43,44}.json
```

Create a CSV containing each validation loss and its same-seed gap to FP32:

```bash
{
  echo "recipe,seed,validation_loss,paired_gap_to_fp32"
  for seed in 42 43 44; do
    fp32_loss=$(jq -r '.validation_loss' \
      "${OUTPUT_DIR}/results/fp32-seed${seed}.json")
    for recipe in fp32 ze-eden zip-sr; do
      loss=$(jq -r '.validation_loss' \
        "${OUTPUT_DIR}/results/${recipe}-seed${seed}.json")
      gap=$(awk -v loss="${loss}" -v fp32="${fp32_loss}" \
        'BEGIN { printf "%.10f", loss - fp32 }')
      echo "${recipe},${seed},${loss},${gap}"
    done
  done
} > "${OUTPUT_DIR}/paired-results.csv"

column -s, -t "${OUTPUT_DIR}/paired-results.csv"
```

Summarize the FP32 absolute validation loss and each 4-bit method's paired gap
as mean and sample standard deviation:

```bash
awk -F, '
  NR == 1 { next }
  {
    value = ($1 == "fp32") ? $3 : $4
    count[$1]++
    sum[$1] += value
    sumsq[$1] += value * value
  }
  END {
    print "recipe,metric,mean,sample_sd"
    for (recipe in count) {
      n = count[recipe]
      mean = sum[recipe] / n
      sd = (n > 1) ? sqrt((sumsq[recipe] - sum[recipe]^2 / n) / (n - 1)) : 0
      metric = (recipe == "fp32") ? "validation_loss" : "paired_gap_to_fp32"
      printf "%s,%s,%.6f,%.6f\n", recipe, metric, mean, sd
    }
  }
' "${OUTPUT_DIR}/paired-results.csv" \
  | tee "${OUTPUT_DIR}/summary.csv"
```

Interpretation:

- FP32 is reported as absolute mean validation loss.
- ZE-EDEN and ZIP-SR are reported as paired gaps to the matching FP32 seed.
- Smaller paired gaps are better.
- These results measure this example's configuration; they do not establish
  reproduction of the paper's reported losses.
