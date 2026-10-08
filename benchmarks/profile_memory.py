#!/usr/bin/env python3
"""Profile optimizer memory and step time on synthetic, preallocated gradients.

Each case/method runs in a fresh subprocess to isolate allocator caches. This
measures optimizer steps only: no forward, backward, or gradient generation is
inside the measurement. Requires one CUDA device and the package dependencies.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


METHODS = (
    "adamw-single", "fp32-state", "ze-eden", "zip-sr",
    "adamw-default", "adamw-foreach",
)
LIBRARY_METHODS = ("fp32-state", "ze-eden", "zip-sr")
CASES = ("1m", "16m", "40m", "model", "many")


def make_optimizer(parameters, method: str, optimized: bool = False):
    """Keep constructor defaults intact unless a library override was requested."""
    import torch

    common = dict(lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    if optimized:
        if method not in LIBRARY_METHODS:
            raise ValueError("optimized applies only to library methods")
        common["optimized"] = True
    if method == "adamw-single":
        return torch.optim.AdamW(parameters, foreach=False, fused=False, **common)
    if method == "adamw-foreach":
        return torch.optim.AdamW(parameters, foreach=True, fused=False, **common)
    if method == "adamw-default":
        return torch.optim.AdamW(parameters, **common)

    from adamw4bit import QuantizedAdamW, ZEEDENAdamW4Bit, ZIPSRAdamW4Bit

    if method == "fp32-state":
        return QuantizedAdamW(
            parameters, m1_quant_scheme="fp32", m2_quant_scheme="fp32", **common
        )
    optimizer_class = ZEEDENAdamW4Bit if method == "ze-eden" else ZIPSRAdamW4Bit
    return optimizer_class(parameters, **common)


def shapes_for(case: str) -> list[tuple[int, ...]]:
    if case in ("1m", "16m", "40m"):
        return [(int(case[:-1]) * 1024 * 1024,)]
    if case == "many":
        return [(64, 64)] * 128 + [(768,)] * 24
    # Generic GPT-small-shaped tensors: untied vocabulary matrices, twelve
    # attention/MLP blocks, and small normalization tensors. No model download.
    width = 768
    shapes = [(50_257, width)]
    for _ in range(12):
        shapes.extend([
            (3 * width, width), (width, width),
            (4 * width, width), (width, 4 * width),
            (width,), (width,), (width,), (width,),
        ])
    return shapes + [(width,), (width,), (50_257, width)]


def source_directory(path: str) -> Path:
    root = Path(path).resolve()
    for candidate in (root / "src", root):
        if (candidate / "adamw4bit" / "adamw.py").is_file():
            return candidate
    raise ValueError(f"No adamw4bit source package under {root}")


def source_files(source: Path) -> list[str]:
    package = source / "adamw4bit"
    return sorted(path.relative_to(package).as_posix()
                  for path in package.rglob("*.py") if path.is_file())


def source_hash(source: Path) -> str:
    digest = hashlib.sha256()
    for name in source_files(source):
        digest.update(name.encode() + b"\0")
        digest.update((source / "adamw4bit" / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def unique_tensor_bytes(values, device_type: str) -> int:
    """Count actual unique backing storage, including QuantState.absmax."""
    import torch

    seen = set()
    total = 0

    def visit(value):
        nonlocal total
        if torch.is_tensor(value):
            if value.device.type != device_type:
                return
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif hasattr(value, "absmax"):
            visit(value.absmax)

    visit(values)
    return total


def measure(optimizer, steps: int, device) -> dict:
    import torch

    torch.cuda.synchronize(device)
    pre_allocated = torch.cuda.memory_allocated(device)
    pre_reserved = torch.cuda.memory_reserved(device)
    pre_requested = torch.cuda.memory_stats(device).get("requested_bytes.all.current")
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(steps):
        optimizer.step()
    end.record()
    torch.cuda.synchronize(device)
    wall_ms = 1000 * (time.perf_counter() - wall_start) / steps
    post_allocated = torch.cuda.memory_allocated(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    memory_stats = torch.cuda.memory_stats(device)
    return {
        "steps": steps,
        "cuda_ms_per_step": start.elapsed_time(end) / steps,
        "wall_ms_per_step": wall_ms,
        "pre_allocated_bytes": pre_allocated,
        "post_allocated_bytes": post_allocated,
        "peak_allocated_bytes": peak_allocated,
        "pre_requested_bytes": pre_requested,
        "post_requested_bytes": memory_stats.get("requested_bytes.all.current"),
        "peak_requested_bytes": memory_stats.get("requested_bytes.all.peak"),
        "incremental_peak_bytes": peak_allocated - pre_allocated,
        # Cold allocation includes newly-created moment state. This separate
        # number measures excess above both before/after resident footprints.
        "peak_excess_over_resident_bytes": max(
            0, peak_allocated - max(pre_allocated, post_allocated)
        ),
        "pre_reserved_bytes": pre_reserved,
        "post_reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def worker(config: dict) -> dict:
    import torch

    source = source_directory(config["source"])
    initial_source_files = source_files(source)
    initial_source_hash = source_hash(source)
    benchmark_path = Path(__file__).resolve()
    benchmark_hash = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
    sys.path.insert(0, str(source))

    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Memory profiling requires an available CUDA device")
    torch.cuda.set_device(device)
    torch.manual_seed(config["seed"])
    generator = torch.Generator(device=device).manual_seed(config["seed"] + 1)
    dtype = getattr(torch, config["dtype"])
    shapes = shapes_for(config["case"])
    parameters = [
        torch.nn.Parameter(torch.randn(shape, device=device, dtype=dtype, generator=generator) * 0.01)
        for shape in shapes
    ]
    for parameter in parameters:
        parameter.grad = torch.randn(
            parameter.shape, device=device, dtype=dtype, generator=generator
        ) * 0.01
    method = config["method"]
    optimized = config.get("optimized", False)
    optimizer = make_optimizer(parameters, method, optimized)
    head_sr = config["head_sr"] and method in ("ze-eden", "zip-sr")
    if head_sr:
        eligible_parameters = [
            p for p in parameters if p.numel() >= 4096 and p.numel() % 128 == 0
        ]
        optimizer.set_m1_quant_scheme_for_parameters(eligible_parameters[-1:], "nf4_sr")

    # Release scratch allocations from input construction. Keep parameters and
    # gradients resident. Never empty the cache between timed optimizer steps.
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    cold = measure(optimizer, 1, device)
    for _ in range(config["warmup"]):
        optimizer.step()
    torch.cuda.synchronize(device)
    steady = [measure(optimizer, config["steps"], device) for _ in range(config["repeats"])]
    state_values = list(optimizer.state.values())
    param_bytes = unique_tensor_bytes(parameters, "cuda")
    grad_bytes = unique_tensor_bytes([p.grad for p in parameters], "cuda")
    state_bytes = unique_tensor_bytes(state_values, "cuda")
    cpu_state_bytes = unique_tensor_bytes(state_values, "cpu")
    numel = sum(p.numel() for p in parameters)
    eligible = [p for p in parameters if p.numel() >= 4096 and p.numel() % 128 == 0]
    eligible_numel = sum(p.numel() for p in eligible)
    fallback_numel = numel - eligible_numel
    predicted_4bit = eligible_numel + (eligible_numel // 128) * 8 + fallback_numel * 8
    # Native AdamW follows parameter dtype; the library keeps working/fallback
    # states in FP32. The 8-byte reference is specifically FP32 moment storage.
    expected_for_method = (
        predicted_4bit if method in ("ze-eden", "zip-sr")
        else 8 * numel if method == "fp32-state"
        else 2 * numel * torch.empty((), dtype=dtype).element_size()
    )
    if state_bytes != expected_for_method:
        raise RuntimeError(
            f"Persistent CUDA optimizer state is {state_bytes} bytes; "
            f"expected {expected_for_method} bytes for {method}"
        )
    resident_memory_stats = torch.cuda.memory_stats(device)
    result = {
        "case": config["case"], "method": method, "dtype": config["dtype"],
        "head_sr": head_sr,
        "optimized": optimized,
        "effective_group_chunk_numel": [
            group.get("max_chunk_numel") for group in optimizer.param_groups
        ],
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        "source_sha256": initial_source_hash,
        "source_hash_files": initial_source_files,
        "benchmark_sha256": benchmark_hash,
        "seed": config["seed"],
        "warmup": config["warmup"], "steps": config["steps"],
        "repeats": config["repeats"],
        "parameter_tensors": len(parameters), "parameter_numel": numel,
        "largest_parameter_numel": max(p.numel() for p in parameters),
        "eligible_tensor_count": len(eligible),
        "fallback_tensor_count": len(parameters) - len(eligible),
        "eligible_numel": eligible_numel, "fallback_numel": fallback_numel,
        "parameter_bytes": param_bytes, "gradient_bytes": grad_bytes,
        "predicted_fp32_moment_bytes": 8 * numel,
        "predicted_4bit_moment_bytes": predicted_4bit,
        "predicted_method_moment_bytes": expected_for_method,
        "persistent_state_matches_prediction": True,
        "persistent_optimizer_cuda_bytes": state_bytes,
        "persistent_optimizer_cpu_bytes": cpu_state_bytes,
        "resident_cuda_bytes": torch.cuda.memory_allocated(device),
        "resident_requested_cuda_bytes": resident_memory_stats.get("requested_bytes.all.current"),
        "other_resident_cuda_bytes": torch.cuda.memory_allocated(device) - param_bytes - grad_bytes - state_bytes,
        "cold": cold, "steady": steady,
        "steady_cuda_ms_median": statistics.median(x["cuda_ms_per_step"] for x in steady),
        "steady_wall_ms_median": statistics.median(x["wall_ms_per_step"] for x in steady),
        "steady_peak_allocated_max": max(x["peak_allocated_bytes"] for x in steady),
        "steady_workspace_peak_max": max(x["peak_excess_over_resident_bytes"] for x in steady),
    }
    if config.get("profile_dir"):
        # Profiling is an additional step after every measurement above. Its
        # retained trace buffers must not affect the timing or peak results.
        output = Path(config["profile_dir"])
        output.mkdir(parents=True, exist_ok=True)
        suffix = "-head-sr" if head_sr else ""
        backend = "optimized" if optimized else "reference"
        stem = output / f"{config['case']}-{method}{suffix}-{backend}"
        torch.cuda.synchronize(device)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
        ) as profile:
            with torch.profiler.record_function("optimizer_step"):
                optimizer.step()
            torch.cuda.synchronize(device)
        profile.export_chrome_trace(str(stem.with_suffix(".trace.json")))
        operator_rows = []
        for event in profile.key_averages(group_by_input_shape=True):
            operator_rows.append({
                "operator": event.key,
                "input_shapes": event.input_shapes,
                "calls": event.count,
                "self_device_memory_bytes": event.self_device_memory_usage,
                "device_memory_bytes": event.device_memory_usage,
                "self_cpu_memory_bytes": event.self_cpu_memory_usage,
            })
        operator_rows.sort(key=lambda item: item["self_device_memory_bytes"], reverse=True)
        stem.with_suffix(".operators.json").write_text(json.dumps(operator_rows, indent=2) + "\n")
        result["profile_trace_file"] = stem.with_suffix(".trace.json").name
        result["profile_operator_file"] = stem.with_suffix(".operators.json").name
    if source_files(source) != initial_source_files or source_hash(source) != initial_source_hash:
        raise RuntimeError("Package source changed during profiling; discard this result")
    if hashlib.sha256(benchmark_path.read_bytes()).hexdigest() != benchmark_hash:
        raise RuntimeError("Benchmark script changed during profiling; discard this result")
    result["source_hash_unchanged"] = True
    result["benchmark_hash_unchanged"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=["1m", "16m", "40m", "model"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS[:4]))
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head-sr", action="store_true",
                        help="Use NF4-SR on the last eligible tensor of each 4-bit preset")
    parser.add_argument("--optimized", action="store_true",
                        help="Opt in to the optimized library implementation; native AdamW controls do not use it")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile-dir", type=Path,
                        help="Record a separate extra step after timing; keep raw traces outside source repositories")
    parser.add_argument("--dry-run", action="store_true", help="Print tensor counts without importing torch or starting CUDA")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.loads(args.worker))))
        return
    if args.warmup < 0 or args.steps < 1 or args.repeats < 1:
        parser.error("warmup must be nonnegative; steps and repeats must be positive")
    if args.optimized and any(
        method not in LIBRARY_METHODS for method in args.methods
    ):
        parser.error("--optimized requires only library methods: "
                     "fp32-state ze-eden zip-sr; native AdamW controls do not use it")
    source_directory(args.source)
    if args.dry_run:
        for case in args.cases:
            import math
            sizes = [math.prod(shape) for shape in shapes_for(case)]
            print(json.dumps({"case": case, "tensors": len(sizes), "numel": sum(sizes),
                              "largest": max(sizes),
                              "optimized": args.optimized}))
        return
    results = []
    for case in args.cases:
        for method in args.methods:
            print(f"Profiling {case} / {method}", file=sys.stderr, flush=True)
            config = dict(source=args.source, device=args.device, case=case, method=method,
                          dtype=args.dtype, seed=args.seed, warmup=args.warmup,
                          steps=args.steps, repeats=args.repeats,
                          head_sr=args.head_sr,
                          optimized=args.optimized,
                          profile_dir=str(args.profile_dir) if args.profile_dir else None)
            run = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(config)],
                capture_output=True, text=True,
            )
            if run.returncode:
                sys.stderr.write(run.stderr)
                raise SystemExit(run.returncode)
            if run.stderr:
                sys.stderr.write(run.stderr)
            results.append(json.loads(run.stdout))
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps({"results": results}, indent=2) + "\n")
    if args.output:
        print(f"{'case':<7} {'method':<16} {'state MiB':>10} {'resident MiB':>13} "
              f"{'cold peak MiB':>13} {'steady peak MiB':>15} "
              f"{'steady extra MiB':>16} {'steady CUDA ms':>14} {'steady wall ms':>14}")
        for result in results:
            mib = 1024 ** 2
            print(f"{result['case']:<7} {result['method']:<16} "
                  f"{result['persistent_optimizer_cuda_bytes'] / mib:10.2f} "
                  f"{result['resident_cuda_bytes'] / mib:13.2f} "
                  f"{result['cold']['peak_allocated_bytes'] / mib:13.2f} "
                  f"{result['steady_peak_allocated_max'] / mib:15.2f} "
                  f"{result['steady_workspace_peak_max'] / mib:16.2f} "
                  f"{result['steady_cuda_ms_median']:14.3f} "
                  f"{result['steady_wall_ms_median']:14.3f}")
        print(f"Full measurements: {args.output}")
    else:
        print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
