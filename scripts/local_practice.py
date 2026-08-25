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
REPORT_SCHEMA_VERSION = 3
NCU_SETS = ("basic", "detailed", "full", "roofline")
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
        "schema_version": REPORT_SCHEMA_VERSION,
        "challenge": challenge_relative_path(challenge_dir),
        "backend": backend,
        "status": "not_run",
        "updated_at": None,
        "native_profiles": {},
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

    native_profiles = report.get("native_profiles", {})
    if native_profiles:
        lines.extend(
            [
                "",
                "## Native Nsight profiles",
                "",
                "| Tool | Status | File | Size | Version | Metric set | Source embedded |",
                "|---|---:|---|---:|---|---|---:|",
            ]
        )
        for tool in ("nsys", "ncu"):
            native = native_profiles.get(tool)
            if not native:
                continue
            source_embedded = native.get("source_embedded")
            embedded_text = "yes" if source_embedded else "no"
            lines.append(
                "| {tool} | {status} | `{file}` | {size} bytes | {version} | {metric_set} | "
                "{embedded} |".format(
                    tool=tool,
                    status=native.get("status", "unknown"),
                    file=native.get("file", "not generated"),
                    size=format_number(native.get("size_bytes")),
                    version=markdown_cell(native.get("tool_version", "unavailable")),
                    metric_set=native.get("metric_set") or "n/a",
                    embedded=embedded_text,
                )
            )
            if native.get("open_command"):
                lines.extend(["", f"Open {tool}: `{native['open_command']}`"])
            last_attempt = native.get("last_attempt")
            if last_attempt and last_attempt.get("status") == "failed":
                lines.extend(
                    [
                        "",
                        f"Latest {tool} generation attempt failed: "
                        f"{last_attempt.get('error', 'unknown error')}",
                    ]
                )
        if any(item.get("status") == "stale" for item in native_profiles.values()):
            lines.extend(
                [
                    "",
                    "A stale report was generated from a different solution SHA-256 and was "
                    "left on disk intentionally.",
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


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


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


def compile_cuda_solution(challenge_dir: Path, solution_path: Path) -> tuple[Path, dict[str, Any]]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise PracticeError("nvcc is not available on PATH")
    nvcc_path = Path(nvcc).resolve()
    nvcc_version = run_command([str(nvcc_path), "--version"])
    capability = torch.cuda.get_device_capability()
    arch = f"sm_{capability[0]}{capability[1]}"
    source = solution_path.read_bytes()
    compile_flags = ["-O3", "-lineinfo", f"-arch={arch}", "--shared"]
    if os.name == "nt":
        compile_flags.extend(["-Xlinker", "/EXPORT:solve"])
    else:
        compile_flags.extend(["-Xcompiler", "-fPIC"])
    cache_metadata = {
        "solution_path": str(solution_path.resolve()),
        "arch": arch,
        "platform": platform.platform(),
        "nvcc_path": str(nvcc_path),
        "nvcc_version": nvcc_version,
        "compile_flags": compile_flags,
    }
    cache_input = source + json.dumps(cache_metadata, sort_keys=True).encode("utf-8")
    build_key = hashlib.sha256(cache_input).hexdigest()[:16]
    build_dir = BUILD_ROOT / challenge_dir.parent.name / challenge_dir.name / "cuda" / build_key
    build_dir.mkdir(parents=True, exist_ok=True)
    library_path = build_dir / f"solution{cuda_library_suffix()}"
    command = [str(nvcc_path), *compile_flags]
    command.extend(["-o", str(library_path), str(solution_path)])
    if not library_path.exists():
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            details = (result.stdout + "\n" + result.stderr).strip()
            raise PracticeError(f"CUDA compilation failed:\n{details}")
    return library_path, {
        "library": str(library_path),
        "compile_command": command,
        "nvcc_path": str(nvcc_path),
        "nvcc_version": nvcc_version,
        "compile_flags": compile_flags,
        "build_cache_key": build_key,
    }


def is_pointer_ctype(value: Any) -> bool:
    try:
        return issubclass(value, ctypes._Pointer)  # type: ignore[attr-defined]
    except TypeError:
        return False


def make_cuda_invoker(library_path: Path, signature: Dict[str, tuple]) -> Callable[..., Any]:
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
    return invoke


def load_cuda_solution(
    challenge_dir: Path, solution_path: Path, signature: Dict[str, tuple]
) -> tuple[Callable[..., Any], dict[str, Any]]:
    library_path, compile_details = compile_cuda_solution(challenge_dir, solution_path)
    return make_cuda_invoker(library_path, signature), compile_details


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


def native_profile_path(solution_path: Path, tool: str) -> Path:
    return solution_path.with_suffix(f".{tool}-rep")


def find_native_tool(tool: str) -> Path | None:
    executable = shutil.which(tool)
    if executable is None and os.name == "nt":
        executable = shutil.which(f"{tool}.exe") or shutil.which(f"{tool}.bat")
    if executable is None:
        return None
    path = Path(executable).resolve()
    if tool == "ncu" and path.suffix.lower() in {".bat", ".cmd"}:
        native_executable = path.parent / "target" / "windows-desktop-win7-x64" / "ncu.exe"
        if native_executable.is_file():
            return native_executable.resolve()
    return path


def executable_prefix(path: Path) -> list[str]:
    if os.name == "nt" and path.suffix.lower() in {".bat", ".cmd"}:
        command_processor = os.environ.get("COMSPEC", "cmd.exe")
        return [command_processor, "/d", "/s", "/c", str(path)]
    return [str(path)]


def native_tool_version(path: Path) -> str:
    return run_command([*executable_prefix(path), "--version"], timeout=60)


def native_profile_error(tool: str, result: subprocess.CompletedProcess[str]) -> str:
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if "ERR_NVGPUCTRPERM" in output:
        return (
            "NCU cannot access NVIDIA GPU Performance Counters (ERR_NVGPUCTRPERM). "
            "Enable access in NVIDIA Control Panel > Desktop/Developer > Manage GPU "
            "Performance Counters, then allow access to all users, or run in a permitted "
            "administrator session."
        )
    tail = output[-4000:] if output else "no diagnostic output"
    return f"{tool} exited with code {result.returncode}: {tail}"


def native_target_command(
    challenge_dir: Path,
    solution_path: Path,
    library_path: Path,
    solution_sha256: str,
    warmup: int,
    seed: int,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_native-target",
        str(challenge_dir),
        "--solution",
        str(solution_path),
        "--library",
        str(library_path),
        "--solution-sha256",
        solution_sha256,
        "--warmup",
        str(warmup),
        "--seed",
        str(seed),
    ]


def generate_native_profile(
    tool: str,
    challenge_dir: Path,
    solution_path: Path,
    library_path: Path,
    solution_sha256: str,
    warmup: int,
    seed: int,
    ncu_set: str,
) -> dict[str, Any]:
    tool_path = find_native_tool(tool)
    if tool_path is None:
        display_name = "NVIDIA Nsight Systems" if tool == "nsys" else "NVIDIA Nsight Compute"
        raise PracticeError(f"{display_name} CLI ({tool}) is not available on PATH")

    output_dir = (
        BUILD_ROOT
        / challenge_dir.parent.name
        / challenge_dir.name
        / "native_profiles"
        / solution_sha256[:16]
        / tool
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_name = f"{solution_path.stem}-{os.getpid()}-{time.time_ns()}"
    temporary_base = output_dir / unique_name
    temporary_report = temporary_base.with_suffix(f".{tool}-rep")
    final_report = native_profile_path(solution_path, tool)
    target_command = native_target_command(
        challenge_dir,
        solution_path,
        library_path,
        solution_sha256,
        warmup,
        seed,
    )

    if tool == "nsys":
        profiler_arguments = [
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--cuda-memory-usage=true",
            "--force-overwrite=true",
            f"--output={temporary_base}",
        ]
        timeout = 300
    else:
        profiler_arguments = [
            "--set",
            ncu_set,
            "--target-processes",
            "application-only",
            "--profile-from-start",
            "off",
            "--import-source",
            "on",
            "--source-folders",
            str(solution_path.parent),
            "--export",
            str(temporary_base),
            "--force-overwrite",
        ]
        timeout = 1800

    command = [*executable_prefix(tool_path), *profiler_arguments, *target_command]
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        temporary_report.unlink(missing_ok=True)
        raise PracticeError(f"{tool} timed out after {timeout} seconds") from exc
    if result.returncode != 0:
        temporary_report.unlink(missing_ok=True)
        raise PracticeError(native_profile_error(tool, result))
    if not temporary_report.is_file() or temporary_report.stat().st_size == 0:
        temporary_report.unlink(missing_ok=True)
        raise PracticeError(f"{tool} did not produce a non-empty {temporary_report.name}")

    final_report.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary_report, final_report)
    return {
        "status": "passed",
        "format": "Nsight Systems report" if tool == "nsys" else "Nsight Compute report",
        "file": final_report.name,
        "size_bytes": final_report.stat().st_size,
        "solution_sha256": solution_sha256,
        "generated_at": utc_now(),
        "tool_path": str(tool_path),
        "tool_version": native_tool_version(tool_path),
        "metric_set": ncu_set if tool == "ncu" else None,
        "source_embedded": tool == "ncu",
        "open_command": f"{tool}-ui {final_report.name}",
        "error": None,
    }


def carry_native_profiles(
    previous_report: dict[str, Any], solution_path: Path, solution_sha256: str
) -> dict[str, Any]:
    carried = copy.deepcopy(previous_report.get("native_profiles", {}))
    if not isinstance(carried, dict):
        return {}
    for native in carried.values():
        if not isinstance(native, dict):
            continue
        filename = native.get("file")
        exists = bool(
            filename
            and (solution_path.parent / filename).is_file()
            and (solution_path.parent / filename).stat().st_size > 0
        )
        if native.get("solution_sha256") != solution_sha256:
            native["status"] = "stale"
        elif not exists:
            native["status"] = "missing"
        elif native.get("status") in {"stale", "missing"}:
            native["status"] = "passed"
    return carried


def record_native_failure(
    native_profiles: dict[str, Any],
    tool: str,
    solution_path: Path,
    solution_sha256: str,
    ncu_set: str,
    error: str,
) -> None:
    attempt = {
        "status": "failed",
        "generated_at": utc_now(),
        "file": native_profile_path(solution_path, tool).name,
        "solution_sha256": solution_sha256,
        "metric_set": ncu_set if tool == "ncu" else None,
        "error": error,
    }
    existing = native_profiles.get(tool)
    if isinstance(existing, dict) and existing.get("size_bytes"):
        existing["error"] = error
        existing["last_attempt"] = attempt
        return
    native_profiles[tool] = {
        "status": "failed",
        "format": "Nsight Systems report" if tool == "nsys" else "Nsight Compute report",
        "file": attempt["file"],
        "size_bytes": None,
        "solution_sha256": solution_sha256,
        "generated_at": None,
        "tool_version": "unavailable",
        "metric_set": attempt["metric_set"],
        "source_embedded": False,
        "error": error,
        "last_attempt": attempt,
    }


def cuda_profiler_call(name: str, result: Any) -> None:
    value = getattr(result, "value", result)
    if value is not None and int(value) != 0:
        raise PracticeError(f"{name} failed with CUDA error {value}")


def run_native_target(
    challenge_value: str,
    solution_value: str,
    library_value: str,
    expected_sha256: str,
    warmup: int,
    seed: int,
) -> int:
    challenge_dir = resolve_challenge_path(challenge_value)
    solution_path = Path(solution_value).resolve()
    library_path = Path(library_value).resolve()
    if not solution_path.is_file():
        raise PracticeError(f"CUDA solution does not exist: {solution_path}")
    if solution_hash(solution_path) != expected_sha256:
        raise PracticeError("CUDA solution changed after the precompiled library was created")
    if not library_path.is_file():
        raise PracticeError(f"Precompiled CUDA library does not exist: {library_path}")
    if not torch.cuda.is_available():
        raise PracticeError("CUDA is not available to PyTorch")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    challenge = load_challenge(challenge_dir)
    signature = challenge.get_solve_signature()
    invoke = make_cuda_invoker(library_path, signature)
    case = materialize_case(challenge.generate_performance_test())
    initial_inout = inout_snapshot(case, signature)
    for _ in range(warmup):
        reset_inout(case, initial_inout)
        invoke(**case)
    synchronize()
    reset_inout(case, initial_inout)
    synchronize()
    if solution_hash(solution_path) != expected_sha256:
        raise PracticeError("CUDA solution changed while the native profiling target warmed up")

    cudart = torch.cuda.cudart()
    profiler_started = False
    range_started = False
    try:
        cuda_profiler_call("cudaProfilerStart", cudart.cudaProfilerStart())
        profiler_started = True
        torch.cuda.nvtx.range_push("leetgpu::cuda::solve")
        range_started = True
        invoke(**case)
        synchronize()
    finally:
        if range_started:
            torch.cuda.nvtx.range_pop()
        if profiler_started:
            cuda_profiler_call("cudaProfilerStop", cudart.cudaProfilerStop())
    return 0


def run_backend(
    challenge_dir: Path,
    backend: str,
    mode: str,
    warmup: int,
    repeat: int,
    should_profile: bool,
    solution_override: str | None,
    seed: int,
    run_nsys: bool,
    run_ncu: bool,
    ncu_set: str,
) -> bool:
    paths = report_paths(challenge_dir, backend)
    previous_report = read_report(paths["json"])
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "challenge": challenge_relative_path(challenge_dir),
        "backend": backend,
        "mode": mode,
        "status": "failed",
        "updated_at": utc_now(),
        "environment": environment_info(),
        "errors": [],
        "native_profiles": {},
    }
    standard_success = False
    solution_path: Path | None = None
    current_solution: dict[str, str] | None = None
    backend_details: dict[str, Any] = {}
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
        report["native_profiles"] = carry_native_profiles(
            previous_report, solution_path, current_solution["sha256"]
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
        standard_success = True
    except Exception as exc:
        report["errors"].append(
            {
                "stage": mode,
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    if (
        standard_success
        and backend == "cuda"
        and mode in {"benchmark", "run"}
        and (run_nsys or run_ncu)
        and solution_path is not None
        and current_solution is not None
    ):
        native_success = True
        library_path = Path(backend_details["library"])
        for tool, requested in (("nsys", run_nsys), ("ncu", run_ncu)):
            if not requested:
                continue
            try:
                report["native_profiles"][tool] = generate_native_profile(
                    tool=tool,
                    challenge_dir=challenge_dir,
                    solution_path=solution_path,
                    library_path=library_path,
                    solution_sha256=current_solution["sha256"],
                    warmup=warmup,
                    seed=seed,
                    ncu_set=ncu_set,
                )
            except Exception as exc:
                native_success = False
                record_native_failure(
                    report["native_profiles"],
                    tool,
                    solution_path,
                    current_solution["sha256"],
                    ncu_set,
                    str(exc),
                )
                report["errors"].append(
                    {
                        "stage": tool,
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
        if not native_success:
            report["status"] = "partial"

    write_report(challenge_dir, backend, report)
    status = {"passed": "PASS", "partial": "PARTIAL"}.get(report["status"], "FAIL")
    print(f"[{status}] {challenge_relative_path(challenge_dir)} [{backend}]")
    print(f"  report: {paths['markdown'].relative_to(REPO_ROOT)}")
    if report.get("profile"):
        print(f"  trace:  {paths['trace'].relative_to(REPO_ROOT)}")
    for tool, native in report.get("native_profiles", {}).items():
        if native.get("status") == "passed":
            print(f"  {tool}:   {solution_path.parent / native['file']}")
    for error in report["errors"]:
        print(f"  error:  [{error['stage']}] {error['message']}")
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


def add_run_arguments(parser: argparse.ArgumentParser, allow_native_profiles: bool) -> None:
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
    if allow_native_profiles:
        parser.add_argument(
            "--nsys",
            action="store_true",
            help="Generate a CUDA Nsight Systems .nsys-rep beside the solution",
        )
        parser.add_argument(
            "--ncu",
            action="store_true",
            help="Generate a CUDA Nsight Compute .ncu-rep beside the solution",
        )
        parser.add_argument(
            "--ncu-set",
            choices=NCU_SETS,
            default=None,
            help="Nsight Compute metric set (default: detailed; requires --ncu)",
        )
    else:
        parser.set_defaults(nsys=False, ncu=False, ncu_set=None)


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
        add_run_arguments(command_parser, allow_native_profiles=command != "test")
    return parser


def build_native_target_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("challenge")
    parser.add_argument("--solution", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--solution-sha256", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "_native-target":
        native_args = build_native_target_parser().parse_args(sys.argv[2:])
        try:
            return run_native_target(
                challenge_value=native_args.challenge,
                solution_value=native_args.solution,
                library_value=native_args.library,
                expected_sha256=native_args.solution_sha256,
                warmup=native_args.warmup,
                seed=native_args.seed,
            )
        except Exception as exc:
            print(f"native target error: {exc}", file=sys.stderr)
            return 2

    parser = build_parser()
    args = parser.parse_args()
    if args.command == "init":
        return scaffold(args.challenge)
    if args.warmup < 0 or args.repeat < 1:
        parser.error("--warmup must be non-negative and --repeat must be at least 1")
    if args.solution and args.backend == "all":
        parser.error("--solution requires an explicit --backend")
    if (args.nsys or args.ncu) and args.backend in {"pytorch", "triton"}:
        parser.error("--nsys and --ncu require --backend cuda or --backend all")
    if args.ncu_set is not None and not args.ncu:
        parser.error("--ncu-set requires --ncu")
    ncu_set = args.ncu_set or "detailed"

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
            run_nsys=args.nsys and backend == "cuda",
            run_ncu=args.ncu and backend == "cuda",
            ncu_set=ncu_set,
        )
        success = backend_success and success
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
