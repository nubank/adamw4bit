#!/usr/bin/env bash
set -euo pipefail

# Paired OSS-library validation experiment. Override any setting through the
# environment, for example: SEEDS="42" DRY_RUN=1 ./examples/run_gpt_small_experiment.sh
PYTHON=${PYTHON:-python}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
SEEDS=${SEEDS:-"42 43 44"}
RECIPES=${RECIPES:-"fp32 ze-eden zip-sr"}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/gpt-small-experiment}
MAX_STEPS=${MAX_STEPS:-}
LEARNING_RATE=${LEARNING_RATE:-0.001}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-32}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
SMOKE_TEST=${SMOKE_TEST:-0}
OVERWRITE=${OVERWRITE:-0}
DRY_RUN=${DRY_RUN:-0}

case "${NPROC_PER_NODE}" in
  ''|*[!0-9]*|0) echo "NPROC_PER_NODE must be a positive integer" >&2; exit 2 ;;
esac
if [[ -n "${MAX_STEPS}" && ! "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_STEPS must be empty or a positive integer" >&2
  exit 2
fi
for positive_integer in PER_DEVICE_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS; do
  value=${!positive_integer}
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${positive_integer} must be a positive integer" >&2
    exit 2
  fi
done
for toggle in SMOKE_TEST OVERWRITE DRY_RUN; do
  value=${!toggle}
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${toggle} must be 0 or 1" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/results" "${OUTPUT_DIR}/runs"

for seed in ${SEEDS}; do
  if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed: ${seed}" >&2
    exit 2
  fi

  for recipe in ${RECIPES}; do
    case "${recipe}" in
      fp32|ze-eden|zip-sr) ;;
      *) echo "Unsupported recipe: ${recipe}" >&2; exit 2 ;;
    esac

    result_file="${OUTPUT_DIR}/results/${recipe}-seed${seed}.json"
    log_file="${OUTPUT_DIR}/logs/${recipe}-seed${seed}.log"
    run_dir="${OUTPUT_DIR}/runs/seed-${seed}"

    if [[ -f "${result_file}" && "${OVERWRITE}" != "1" ]]; then
      echo "Reusing ${result_file}"
      continue
    fi

    command=(
      "${PYTHON}" -m torch.distributed.run
      --standalone
      "--nproc-per-node=${NPROC_PER_NODE}"
      examples/train_gpt_small.py
      --recipe "${recipe}"
      --seed "${seed}"
      --learning-rate "${LEARNING_RATE}"
      --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}"
      --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
      --output-dir "${run_dir}"
      --result-file "${result_file}"
    )
    if [[ -n "${MAX_STEPS}" ]]; then
      command+=(--max-steps "${MAX_STEPS}")
    fi
    if [[ "${SMOKE_TEST}" == "1" ]]; then
      command+=(--smoke-test)
    fi

    printf 'Running %s seed %s:' "${recipe}" "${seed}"
    printf ' %q' "${command[@]}"
    printf '\n'
    if [[ "${DRY_RUN}" == "1" ]]; then
      continue
    fi

    "${command[@]}" 2>&1 | tee "${log_file}"
    if [[ ! -s "${result_file}" ]]; then
      echo "Run completed without writing ${result_file}" >&2
      exit 1
    fi
  done
done

echo "Experiment complete. Per-run metrics: ${OUTPUT_DIR}/results/"
