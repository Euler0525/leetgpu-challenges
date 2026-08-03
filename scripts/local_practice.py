#!/usr/bin/env python3
"""Local correctness, benchmark, and profiler runner for LeetGPU challenges."""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
CHALLENGES_ROOT = REPO_ROOT / "challenges"
BUILD_ROOT = REPO_ROOT / "out" / "local_practice"
BACKENDS = ("pytorch", "triton", "cuda")
STARTER_FILES = {
    "pytorch": "starter.pytorch.py",
    "triton": "starter.triton.py",
    "cuda": "starter.cu",
}
SOLUTION_FILES = {
    "pytorch": "solution.py",
    "triton": "solution.py",
    "cuda": "solution.cu",
}

os.environ.setdefault("TRITON_CACHE_DIR", str(BUILD_ROOT / "triton_cache"))


class PracticeError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_utf8(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")


def run_command(command: list[str], timeout: int = 20) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unavailable"
    output = (result.stdout or result.stderr).strip()
    return output if output else f"exit code {result.returncode}"


def discover_challenges() -> list[Path]:
    return sorted(path.parent for path in CHALLENGES_ROOT.glob("*/*/challenge.py"))


def resolve_challenge_path(value: str | Path) -> Path:
    raw = Path(value)
    candidates = [raw, REPO_ROOT / raw, CHALLENGES_ROOT / raw]
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "challenge.py").is_file():
            return resolved
    raise PracticeError(f"Cannot find challenge.py under: {value}")


def challenge_relative_path(challenge_dir: Path) -> str:
    return challenge_dir.relative_to(REPO_ROOT).as_posix()


def report_paths(challenge_dir: Path, backend: str) -> dict[str, Path]:
    backend_dir = challenge_dir / "solution" / backend
    return {
        "json": backend_dir / "report.json",
        "markdown": backend_dir / "report.md",
        "trace": backend_dir / "profile_trace.json",
    }


def initial_report(challenge_dir: Path, backend: str) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "challenge": challenge_relative_path(challenge_dir),
        "backend": backend,
        "status": "not_run",
        "updated_at": None,
        "message": "Run scripts/local_practice.py to generate this report.",
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Local practice report: {report.get('challenge', 'unknown')}",
        "",
        f"- Backend: `{report.get('backend', 'unknown')}`",
        f"- Status: **{report.get('status', 'unknown')}**",
        f"- Updated: {report.get('updated_at') or 'not run'}",
    ]

    correctness = report.get("correctness")
    if correctness:
        lines.extend(
            [
                "",
                "## Correctness",
                "",
                f"Passed {correctness['passed']}/{correctness['total']} cases.",
                "",
                "| Case | Status | Max absolute error | Max relative error |",
                "|---|---:|---:|---:|",
            ]
        )
        for case in correctness["cases"]:
            lines.append(
                "| {name} | {status} | {absolute} | {relative} |".format(
                    name=case["name"],
                    status=case["status"],
                    absolute=format_number(case.get("max_abs_error")),
                    relative=format_number(case.get("max_rel_error")),
                )
            )
            if case.get("message"):
                lines.append(f"\n`{case['name']}`: {case['message']}\n")

    benchmark = report.get("benchmark")
    if benchmark:
        lines.extend(["", "## Benchmark", ""])
        metric_rows = [
            ("GPU latency p50", benchmark["gpu_latency_ms"]["p50"], "ms"),
            ("GPU latency p90", benchmark["gpu_latency_ms"]["p90"], "ms"),
            ("GPU latency p99", benchmark["gpu_latency_ms"]["p99"], "ms"),
            ("End-to-end latency p50", benchmark["wall_latency_ms"]["p50"], "ms"),
            ("Host/sync overhead p50", benchmark["host_sync_overhead_ms_p50"], "ms"),
            (
                "Host/sync overhead ratio p50",
                benchmark["host_sync_overhead_percent_p50"],
                "%",
            ),
            ("Call throughput", benchmark["calls_per_second"], "calls/s"),
            (
                "Output element throughput",
                benchmark["output_throughput_elements_per_second"],
                "elements/s",
            ),
            ("Reference speedup", benchmark["speedup_vs_reference"], "x"),
            ("Reference GPU latency p50", benchmark["reference_gpu_latency_ms"]["p50"], "ms"),
            (
                "Reference effective bandwidth p50",
                benchmark["reference_effective_memory_bandwidth_gbps"]["p50"],
                "GB/s",
            ),
        ]
        lines.extend(["| Metric | Value |", "|---|---:|"])
        for name, value, unit in metric_rows:
            lines.append(f"| {name} | {format_number(value)} {unit} |")

        traffic = benchmark["memory_traffic_bytes"]
        bandwidth = benchmark["effective_memory_bandwidth_gbps"]
        lines.extend(
            [
                "",
                "### Memory traffic and effective bandwidth",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Input reads | {format_number(traffic['input_read'])} bytes |",
                f"| Output writes | {format_number(traffic['output_write'])} bytes |",
                f"| In-place reads | {format_number(traffic['inout_read'])} bytes |",
                f"| In-place writes | {format_number(traffic['inout_write'])} bytes |",
                f"| Minimum total traffic | {format_number(traffic['minimum_total'])} bytes |",
                f"| Effective bandwidth, best | {format_number(bandwidth['best'])} GB/s |",
                f"| Effective bandwidth, p50 | {format_number(bandwidth['p50'])} GB/s |",
                f"| Effective bandwidth, mean | {format_number(bandwidth['mean'])} GB/s |",
                f"| Effective bandwidth, p90 latency | {format_number(bandwidth['p90'])} GB/s |",
                "| Arithmetic throughput | n/a (challenge has no generic FLOP count) |",
            ]
        )

        memory = benchmark["memory"]
        performance_correctness = benchmark["performance_correctness"]
        lines.extend(
            [
                "",
                "### Performance-case validation and memory",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Performance output status | {performance_correctness['status']} |",
                "| Performance max absolute error | "
                f"{format_number(performance_correctness['max_abs_error'])} |",
                "| Performance max relative error | "
                f"{format_number(performance_correctness['max_rel_error'])} |",
                "| Resident PyTorch allocation | "
                f"{format_number(memory['resident_allocated_bytes'])} bytes |",
                "| Resident PyTorch reservation | "
                f"{format_number(memory['resident_reserved_bytes'])} bytes |",
                "| Incremental peak allocation | "
                f"{format_number(memory['incremental_peak_allocated_bytes'])} bytes |",
                "| Incremental peak reservation | "
                f"{format_number(memory['incremental_peak_reserved_bytes'])} bytes |",
                "| Peak PyTorch allocation | "
                f"{format_number(memory['peak_allocated_bytes'])} bytes |",
                "| Peak PyTorch reservation | "
                f"{format_number(memory['peak_reserved_bytes'])} bytes |",
            ]
        )
        lines.extend(
            [
                "",
                f"Warmup iterations: {benchmark['warmup']}; measured iterations: "
                f"{benchmark['repeat']}.",
                "",
                "Effective memory bandwidth uses the minimum bytes implied by input/output "
                "arguments, not hardware-counter DRAM traffic. PyTorch allocator peaks do not "
                "include allocations made directly inside native CUDA code.",
            ]
        )

    profile = report.get("profile")
    if profile:
        lines.extend(
            [
                "",
                "## Profiler",
                "",
                f"Chrome trace: `{profile.get('trace', 'not generated')}`",
            ]
        )

    errors = report.get("errors", [])
    if errors:
        lines.extend(["", "## Errors", ""])
        for error in errors:
            lines.append(f"- `{error['stage']}`: {error['message']}")

    environment = report.get("environment")
    if environment:
        lines.extend(
            [
                "",
                "## Environment",
                "",
                f"- GPU: {environment.get('gpu', 'unavailable')}",
                f"- PyTorch: {environment.get('torch', 'unavailable')}",
                f"- Triton: {environment.get('triton', 'unavailable')}",
                f"- CUDA runtime: {environment.get('torch_cuda', 'unavailable')}",
                f"- Python: {environment.get('python', 'unavailable')}",
                f"- Platform: {environment.get('platform', 'unavailable')}",
            ]
        )

    return "\n".join(lines) + "\n"


def format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.6g}"
    return str(value)


def write_report(challenge_dir: Path, backend: str, report: dict[str, Any]) -> None:
    paths = report_paths(challenge_dir, backend)
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    write_utf8(
        paths["json"],
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    write_utf8(paths["markdown"], render_markdown(report))


def read_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return report if isinstance(report, dict) else {}


def scaffold_challenge(challenge_dir: Path) -> tuple[int, int]:
    created = 0
    skipped = 0
    for backend in BACKENDS:
        starter = challenge_dir / "starter" / STARTER_FILES[backend]
        if not starter.is_file():
            skipped += 1
            continue
        backend_dir = challenge_dir / "solution" / backend
        backend_dir.mkdir(parents=True, exist_ok=True)
        solution = backend_dir / SOLUTION_FILES[backend]
        if not solution.exists():
            shutil.copyfile(starter, solution)
            created += 1

        paths = report_paths(challenge_dir, backend)
        if not paths["json"].exists():
            report = initial_report(challenge_dir, backend)
            write_utf8(paths["json"], json.dumps(report, indent=2, sort_keys=True) + "\n")
            created += 1
        if not paths["markdown"].exists():
            write_utf8(paths["markdown"], render_markdown(initial_report(challenge_dir, backend)))
            created += 1
        if not paths["trace"].exists():
            write_utf8(paths["trace"], '{"traceEvents": []}\n')
            created += 1
    return created, skipped


def scaffold(challenge_value: str | None) -> int:
    challenge_dirs = (
        [resolve_challenge_path(challenge_value)] if challenge_value else discover_challenges()
    )
    created = 0
    unsupported = 0
    for challenge_dir in challenge_dirs:
        challenge_created, challenge_unsupported = scaffold_challenge(challenge_dir)
        created += challenge_created
        unsupported += challenge_unsupported
    print(
        f"Initialized {len(challenge_dirs)} challenge(s); created {created} file(s); "
        f"unsupported backend slots: {unsupported}."
    )
    return 0


def load_challenge(challenge_dir: Path) -> Any:
    challenges_path = str(CHALLENGES_ROOT)
    if challenges_path not in sys.path:
        sys.path.insert(0, challenges_path)
    module_name = (
        "local_practice_challenge_" + hashlib.sha256(str(challenge_dir).encode()).hexdigest()[:12]
    )
    spec = importlib.util.spec_from_file_location(module_name, challenge_dir / "challenge.py")
    if spec is None or spec.loader is None:
        raise PracticeError(f"Cannot import {challenge_dir / 'challenge.py'}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Challenge(device="cuda")


def load_python_solution(solution_path: Path) -> Callable[..., Any]:
    module_name = (
        "local_practice_solution_"
        + hashlib.sha256(str(solution_path.resolve()).encode()).hexdigest()[:12]
    )
    spec = importlib.util.spec_from_file_location(module_name, solution_path)
    if spec is None or spec.loader is None:
        raise PracticeError(f"Cannot import {solution_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    solve = getattr(module, "solve", None)
    if not callable(solve):
        raise PracticeError(f"No callable solve function in {solution_path}")
    return solve


def bind_signature(function: Callable[..., Any], signature: Dict[str, tuple]) -> Callable[..., Any]:
    parameter_names = tuple(signature)

    def invoke(**case: Any) -> Any:
        return function(*(case[name] for name in parameter_names))

    return invoke


def cuda_library_suffix() -> str:
    return ".dll" if os.name == "nt" else ".so"


def compile_cuda_solution(challenge_dir: Path, solution_path: Path) -> tuple[Path, str]:
    if shutil.which("nvcc") is None:
        raise PracticeError("nvcc is not available on PATH")
    capability = torch.cuda.get_device_capability()
    arch = f"sm_{capability[0]}{capability[1]}"
    source = solution_path.read_bytes()
    build_key = hashlib.sha256(source + arch.encode() + platform.platform().encode()).hexdigest()[
        :16
    ]
    build_dir = BUILD_ROOT / challenge_dir.parent.name / challenge_dir.name / "cuda" / build_key
    build_dir.mkdir(parents=True, exist_ok=True)
    library_path = build_dir / f"solution{cuda_library_suffix()}"
    command = ["nvcc", "-O3", "-lineinfo", f"-arch={arch}", "--shared"]
    if os.name == "nt":
        command.extend(["-Xlinker", "/EXPORT:solve"])
    else:
        command.extend(["-Xcompiler", "-fPIC"])
    command.extend(["-o", str(library_path), str(solution_path)])
    if not library_path.exists():
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            details = (result.stdout + "\n" + result.stderr).strip()
            raise PracticeError(f"CUDA compilation failed:\n{details}")
    return library_path, " ".join(command)


def is_pointer_ctype(value: Any) -> bool:
    try:
        return issubclass(value, ctypes._Pointer)  # type: ignore[attr-defined]
    except TypeError:
        return False


def load_cuda_solution(
    challenge_dir: Path, solution_path: Path, signature: Dict[str, tuple]
) -> tuple[Callable[..., Any], dict[str, Any]]:
    library_path, compile_command = compile_cuda_solution(challenge_dir, solution_path)
    library = ctypes.CDLL(str(library_path))
    native_solve = library.solve
    argtypes = []
    for ctype, _ in signature.values():
        argtypes.append(ctypes.c_void_p if is_pointer_ctype(ctype) else ctype)
    native_solve.argtypes = argtypes
    native_solve.restype = None

    def invoke(**case: Any) -> None:
        args = []
        for name, (ctype, _) in signature.items():
            value = case[name]
            if isinstance(value, torch.Tensor):
                args.append(ctypes.c_void_p(value.data_ptr()))
            elif is_pointer_ctype(ctype):
                raise PracticeError(f"CUDA argument {name} is not a tensor")
            else:
                args.append(ctype(value))
        native_solve(*args)

    invoke._native_library = library  # type: ignore[attr-defined]
    return invoke, {"library": str(library_path), "compile_command": compile_command}


def materialize_value(value: Any, device: str = "cuda") -> Any:
    class_name = type(value).__name__
    if class_name not in {"RandTensor", "RandnTensor", "RandIntTensor", "FullTensor", "OutTensor"}:
        return value
    dtype = getattr(torch, value.dtype)
    if class_name == "RandTensor":
        return torch.empty(value.shape, device=device, dtype=dtype).uniform_(value.low, value.high)
    if class_name == "RandnTensor":
        return torch.empty(value.shape, device=device, dtype=dtype).normal_(value.mean, value.std)
    if class_name == "RandIntTensor":
        return torch.randint(value.low, value.high, value.shape, device=device, dtype=dtype)
    if class_name == "FullTensor":
        return torch.full(value.shape, value.value, device=device, dtype=dtype)
    return torch.empty(value.shape, device=device, dtype=dtype)


def materialize_case(case: dict[str, Any]) -> dict[str, Any]:
    return {name: materialize_value(value) for name, value in case.items()}


def clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, torch.nn.Module):
        return copy.deepcopy(value)
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_value(item) for item in value)
    return copy.deepcopy(value)


def clone_case(case: dict[str, Any]) -> dict[str, Any]:
    return {name: clone_value(value) for name, value in case.items()}


def synchronize() -> None:
    torch.cuda.synchronize()


def output_names(signature: Dict[str, tuple]) -> list[str]:
    return [name for name, (_, direction) in signature.items() if direction in {"out", "inout"}]


def reset_inout(case: dict[str, Any], initial: dict[str, Any]) -> None:
    for name, value in initial.items():
        target = case[name]
        if isinstance(target, torch.Tensor):
            target.copy_(value)
        else:
            case[name] = clone_value(value)


def inout_snapshot(case: dict[str, Any], signature: Dict[str, tuple]) -> dict[str, Any]:
    return {
        name: clone_value(case[name])
        for name, (_, direction) in signature.items()
        if direction == "inout"
    }


def tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    if actual.numel() == 0:
        return 0.0, 0.0
    if actual.is_floating_point() or actual.is_complex():
        actual_float = actual.detach().to(torch.float64)
        expected_float = expected.detach().to(actual.device, torch.float64)
        absolute = (actual_float - expected_float).abs()
        denominator = expected_float.abs().clamp_min(torch.finfo(torch.float64).tiny)
        relative = absolute / denominator
        return float(absolute.max().item()), float(relative.max().item())
    mismatches = int((actual != expected.to(actual.device)).sum().item())
    return float(mismatches), float(mismatches)


def assert_output_close(
    actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float
) -> tuple[float, float]:
    expected_device = expected.to(actual.device)
    torch.testing.assert_close(actual, expected_device, atol=atol, rtol=rtol, equal_nan=True)
    return tensor_error(actual, expected_device)


def verify_readonly_inputs(
    actual_case: dict[str, Any], original_case: dict[str, Any], signature: Dict[str, tuple]
) -> None:
    for name, (_, direction) in signature.items():
        if direction != "in":
            continue
        actual = actual_case[name]
        original = original_case[name]
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(actual, original, atol=0, rtol=0, equal_nan=True)


def check_case(
    challenge: Any,
    invoke: Callable[..., Any],
    base_case: dict[str, Any],
    signature: Dict[str, tuple],
    name: str,
    reference: Callable[..., Any],
) -> dict[str, Any]:
    expected_case = clone_case(base_case)
    actual_case = clone_case(base_case)
    result: dict[str, Any] = {"name": name, "status": "passed"}
    try:
        reference(**expected_case)
        synchronize()
        invoke(**actual_case)
        synchronize()
        maximum_absolute = 0.0
        maximum_relative = 0.0
        for output_name in output_names(signature):
            actual = actual_case[output_name]
            expected = expected_case[output_name]
            if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
                raise PracticeError(f"Output {output_name} is not a tensor")
            absolute, relative = assert_output_close(
                actual, expected, challenge.atol, challenge.rtol
            )
            maximum_absolute = max(maximum_absolute, absolute)
            maximum_relative = max(maximum_relative, relative)
        verify_readonly_inputs(actual_case, base_case, signature)
        result["max_abs_error"] = maximum_absolute
        result["max_rel_error"] = maximum_relative
    except Exception as exc:
        result.update(status="failed", message=str(exc))
    finally:
        del expected_case
        del actual_case
    return result


def run_correctness(
    challenge: Any, invoke: Callable[..., Any], signature: Dict[str, tuple]
) -> dict[str, Any]:
    reference = bind_signature(challenge.reference_impl, signature)
    cases = [("example", challenge.generate_example_test())]
    cases.extend(
        (f"functional_{index}", case)
        for index, case in enumerate(challenge.generate_functional_test(), start=1)
    )
    results = []
    for name, raw_case in cases:
        base_case = materialize_case(raw_case)
        results.append(check_case(challenge, invoke, base_case, signature, name, reference))
        del base_case
        torch.cuda.empty_cache()
    passed = sum(result["status"] == "passed" for result in results)
    return {"passed": passed, "total": len(results), "cases": results}


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "stddev": statistics.pstdev(values),
    }


def measure(
    function: Callable[..., Any],
    case: dict[str, Any],
    initial_inout: dict[str, Any],
    warmup: int,
    repeat: int,
) -> tuple[dict[str, float], dict[str, float]]:
    for _ in range(warmup):
        reset_inout(case, initial_inout)
        function(**case)
    synchronize()

    gpu_times = []
    wall_times = []
    for _ in range(repeat):
        reset_inout(case, initial_inout)
        synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start_event.record()
        function(**case)
        end_event.record()
        end_event.synchronize()
        wall_times.append((time.perf_counter() - wall_start) * 1000.0)
        gpu_times.append(float(start_event.elapsed_time(end_event)))
    return latency_stats(gpu_times), latency_stats(wall_times)


def value_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, torch.nn.Module):
        tensors: Iterable[torch.Tensor] = list(value.parameters()) + list(value.buffers())
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    return 0


def memory_traffic(case: dict[str, Any], signature: Dict[str, tuple]) -> dict[str, int]:
    traffic = {
        "input_read": 0,
        "output_write": 0,
        "inout_read": 0,
        "inout_write": 0,
    }
    for name, (_, direction) in signature.items():
        size = value_bytes(case[name])
        if direction == "in":
            traffic["input_read"] += size
        elif direction == "out":
            traffic["output_write"] += size
        elif direction == "inout":
            traffic["inout_read"] += size
            traffic["inout_write"] += size
    traffic["minimum_total"] = sum(traffic.values())
    return traffic


def output_element_count(case: dict[str, Any], signature: Dict[str, tuple]) -> int:
    return sum(
        case[name].numel()
        for name, (_, direction) in signature.items()
        if direction in {"out", "inout"} and isinstance(case[name], torch.Tensor)
    )


def per_second(units: int | float, latency_ms: float) -> float:
    if latency_ms <= 0:
        return math.inf
    return float(units) / (latency_ms / 1000.0)


def bandwidth_stats(traffic_bytes: int, latency: dict[str, float]) -> dict[str, float]:
    return {
        "best": per_second(traffic_bytes, latency["min"]) / 1e9,
        "p50": per_second(traffic_bytes, latency["p50"]) / 1e9,
        "mean": per_second(traffic_bytes, latency["mean"]) / 1e9,
        "p90": per_second(traffic_bytes, latency["p90"]) / 1e9,
    }


def describe_case(case: dict[str, Any]) -> dict[str, Any]:
    description = {}
    for name, value in case.items():
        if isinstance(value, torch.Tensor):
            description[name] = {
                "kind": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "bytes": value_bytes(value),
            }
        elif isinstance(value, torch.nn.Module):
            description[name] = {
                "kind": "module",
                "type": type(value).__name__,
                "bytes": value_bytes(value),
            }
        else:
            description[name] = value
    return description


def profile_solution(
    invoke: Callable[..., Any],
    case: dict[str, Any],
    initial_inout: dict[str, Any],
    trace_path: Path,
    backend: str,
) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    reset_inout(case, initial_inout)
    synchronize()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
    ) as profiler:
        with torch.autograd.profiler.record_function(f"leetgpu::{backend}::solve"):
            invoke(**case)
            synchronize()
    profiler.export_chrome_trace(str(trace_path))
    write_utf8(trace_path, trace_path.read_text(encoding="utf-8"))
    return {
        "trace": trace_path.name,
        "format": "PyTorch Profiler Chrome trace JSON",
        "range": f"leetgpu::{backend}::solve",
        "note": "Native CUDA kernel detail depends on CUPTI support in the local PyTorch build.",
    }


def run_benchmark(
    challenge: Any,
    invoke: Callable[..., Any],
    signature: Dict[str, tuple],
    backend: str,
    warmup: int,
    repeat: int,
    trace_path: Path,
    should_profile: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    base_case = materialize_case(challenge.generate_performance_test())
    case_description = describe_case(base_case)
    reference = bind_signature(challenge.reference_impl, signature)

    reference_case = clone_case(base_case)
    reference_initial = inout_snapshot(reference_case, signature)
    reference_gpu, reference_wall = measure(
        reference, reference_case, reference_initial, warmup, repeat
    )
    reset_inout(reference_case, reference_initial)
    reference(**reference_case)
    synchronize()
    expected_outputs = {
        name: reference_case[name].detach().cpu().clone() for name in output_names(signature)
    }
    del reference_case
    del reference_initial
    torch.cuda.empty_cache()

    solution_case = clone_case(base_case)
    del base_case
    initial_solution_inout = inout_snapshot(solution_case, signature)
    invoke(**solution_case)
    synchronize()
    performance_outputs = []
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for name, expected in expected_outputs.items():
        actual = solution_case[name]
        absolute, relative = assert_output_close(actual, expected, challenge.atol, challenge.rtol)
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
        performance_outputs.append(
            {
                "name": name,
                "max_abs_error": absolute,
                "max_rel_error": relative,
            }
        )
    del expected_outputs

    resident_allocated = torch.cuda.memory_allocated()
    resident_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    reset_inout(solution_case, initial_solution_inout)
    invoke(**solution_case)
    synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    solution_gpu, solution_wall = measure(
        invoke, solution_case, initial_solution_inout, warmup, repeat
    )
    traffic = memory_traffic(solution_case, signature)
    output_elements = output_element_count(solution_case, signature)
    effective_bandwidth = bandwidth_stats(traffic["minimum_total"], solution_gpu)
    reference_bandwidth = bandwidth_stats(traffic["minimum_total"], reference_gpu)
    host_sync_overhead = max(0.0, solution_wall["p50"] - solution_gpu["p50"])
    host_sync_overhead_percent = (
        host_sync_overhead / solution_wall["p50"] * 100.0 if solution_wall["p50"] > 0 else 0.0
    )
    benchmark = {
        "warmup": warmup,
        "repeat": repeat,
        "case": case_description,
        "performance_correctness": {
            "status": "passed",
            "max_abs_error": maximum_absolute,
            "max_rel_error": maximum_relative,
            "outputs": performance_outputs,
        },
        "gpu_latency_ms": solution_gpu,
        "wall_latency_ms": solution_wall,
        "reference_gpu_latency_ms": reference_gpu,
        "reference_wall_latency_ms": reference_wall,
        "speedup_vs_reference": reference_gpu["p50"] / solution_gpu["p50"],
        "host_sync_overhead_ms_p50": host_sync_overhead,
        "host_sync_overhead_percent_p50": host_sync_overhead_percent,
        "calls_per_second": per_second(1, solution_gpu["p50"]),
        "output_elements": output_elements,
        "output_throughput_elements_per_second": per_second(output_elements, solution_gpu["p50"]),
        "memory_traffic_bytes": traffic,
        "effective_memory_bandwidth_gbps": effective_bandwidth,
        "reference_effective_memory_bandwidth_gbps": reference_bandwidth,
        "arithmetic_throughput": {
            "status": "unavailable",
            "reason": "Challenge metadata does not define a generic operation count.",
        },
        "memory": {
            "resident_allocated_bytes": resident_allocated,
            "resident_reserved_bytes": resident_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "incremental_peak_allocated_bytes": max(0, peak_allocated - resident_allocated),
            "incremental_peak_reserved_bytes": max(0, peak_reserved - resident_reserved),
            "native_cuda_allocation_tracking": "not visible to the PyTorch allocator",
        },
        "io_bytes": traffic["minimum_total"],
        "effective_bandwidth_gbps": effective_bandwidth["p50"],
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }
    profile_result = (
        profile_solution(invoke, solution_case, initial_solution_inout, trace_path, backend)
        if should_profile
        else None
    )
    del solution_case
    del initial_solution_inout
    torch.cuda.empty_cache()
    return benchmark, profile_result


def environment_info() -> dict[str, Any]:
    triton_version = "unavailable"
    try:
        import triton

        triton_version = triton.__version__
    except ImportError:
        pass
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "unavailable"
    return {
        "gpu": gpu,
        "gpu_capability": (
            list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
        ),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton_version,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "nvcc": run_command(["nvcc", "--version"]),
        "nvidia_smi": run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }


def solution_path_for(challenge_dir: Path, backend: str, override: str | None = None) -> Path:
    path = (
        Path(override).resolve()
        if override
        else challenge_dir / "solution" / backend / SOLUTION_FILES[backend]
    )
    if not path.is_file():
        raise PracticeError(
            f"No {backend} solution at {path}. Run `python scripts/local_practice.py init`."
        )
    return path


def solution_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_invoker(
    challenge_dir: Path,
    backend: str,
    solution_path: Path,
    signature: Dict[str, tuple],
) -> tuple[Callable[..., Any], dict[str, Any]]:
    if backend in {"pytorch", "triton"}:
        return bind_signature(load_python_solution(solution_path), signature), {}
    return load_cuda_solution(challenge_dir, solution_path, signature)


def run_backend(
    challenge_dir: Path,
    backend: str,
    mode: str,
    warmup: int,
    repeat: int,
    should_profile: bool,
    solution_override: str | None,
    seed: int,
) -> bool:
    paths = report_paths(challenge_dir, backend)
    previous_report = read_report(paths["json"])
    report: dict[str, Any] = {
        "schema_version": 2,
        "challenge": challenge_relative_path(challenge_dir),
        "backend": backend,
        "mode": mode,
        "status": "failed",
        "updated_at": utc_now(),
        "environment": environment_info(),
        "errors": [],
    }
    try:
        if not torch.cuda.is_available():
            raise PracticeError("CUDA is not available to PyTorch")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        challenge = load_challenge(challenge_dir)
        signature = challenge.get_solve_signature()
        if not output_names(signature):
            raise PracticeError("Challenge signature has no out or inout tensor")
        solution_path = solution_path_for(challenge_dir, backend, solution_override)
        current_solution = {
            "path": (
                str(solution_path.relative_to(REPO_ROOT))
                if solution_path.is_relative_to(REPO_ROOT)
                else str(solution_path)
            ),
            "sha256": solution_hash(solution_path),
        }
        report["solution"] = current_solution
        same_solution = (
            previous_report.get("solution", {}).get("sha256") == current_solution["sha256"]
        )
        if same_solution:
            for section in ("correctness", "benchmark", "profile"):
                if section in previous_report:
                    report[section] = previous_report[section]
        elif mode == "test":
            write_utf8(paths["trace"], '{"traceEvents": []}\n')
        invoke, backend_details = create_invoker(challenge_dir, backend, solution_path, signature)
        if backend_details:
            report["backend_details"] = backend_details

        if mode in {"test", "run"}:
            correctness = run_correctness(challenge, invoke, signature)
            correctness["updated_at"] = report["updated_at"]
            report["correctness"] = correctness
            if correctness["passed"] != correctness["total"]:
                raise PracticeError("Correctness tests failed; benchmark was not run")

        if mode in {"benchmark", "run"}:
            benchmark, profile_result = run_benchmark(
                challenge,
                invoke,
                signature,
                backend,
                warmup,
                repeat,
                paths["trace"],
                should_profile,
            )
            benchmark["updated_at"] = report["updated_at"]
            report["benchmark"] = benchmark
            if profile_result:
                profile_result["updated_at"] = report["updated_at"]
                report["profile"] = profile_result

        report["status"] = "passed"
    except Exception as exc:
        report["errors"].append(
            {
                "stage": mode,
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    write_report(challenge_dir, backend, report)
    status = "PASS" if report["status"] == "passed" else "FAIL"
    print(f"[{status}] {challenge_relative_path(challenge_dir)} [{backend}]")
    print(f"  report: {paths['markdown'].relative_to(REPO_ROOT)}")
    if report.get("profile"):
        print(f"  trace:  {paths['trace'].relative_to(REPO_ROOT)}")
    if report["errors"]:
        print(f"  error:  {report['errors'][0]['message']}")
    return report["status"] == "passed"


def backends_for(challenge_dir: Path, requested: str) -> list[str]:
    candidates = list(BACKENDS) if requested == "all" else [requested]
    supported = []
    for backend in candidates:
        starter = challenge_dir / "starter" / STARTER_FILES[backend]
        if starter.is_file():
            supported.append(backend)
        elif requested != "all":
            raise PracticeError(
                f"{challenge_relative_path(challenge_dir)} does not support {backend}"
            )
        else:
            print(f"[SKIP] {challenge_relative_path(challenge_dir)} [{backend}] is unsupported")
    return supported


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("challenge", help="Challenge path, for example easy/1_vector_add")
    parser.add_argument("--backend", choices=("all",) + BACKENDS, default="all")
    parser.add_argument("--solution", help="Override solution file; requires one explicit backend")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Do not generate a PyTorch Profiler Chrome trace for benchmark/run",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LeetGPU solutions locally with correctness and performance reports."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init", help="Create non-destructive solution workspaces")
    init_parser.add_argument("challenge", nargs="?", help="Optional single challenge path")
    for command, help_text in (
        ("test", "Run example and functional correctness tests"),
        ("benchmark", "Run the performance case and profiler"),
        ("run", "Run correctness tests, benchmark, and profiler"),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        add_run_arguments(command_parser)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "init":
        return scaffold(args.challenge)
    if args.warmup < 0 or args.repeat < 1:
        parser.error("--warmup must be non-negative and --repeat must be at least 1")
    if args.solution and args.backend == "all":
        parser.error("--solution requires an explicit --backend")

    try:
        challenge_dir = resolve_challenge_path(args.challenge)
        selected_backends = backends_for(challenge_dir, args.backend)
    except PracticeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    success = True
    for backend in selected_backends:
        backend_success = run_backend(
            challenge_dir=challenge_dir,
            backend=backend,
            mode=args.command,
            warmup=args.warmup,
            repeat=args.repeat,
            should_profile=not args.no_profile,
            solution_override=args.solution,
            seed=args.seed,
        )
        success = backend_success and success
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
