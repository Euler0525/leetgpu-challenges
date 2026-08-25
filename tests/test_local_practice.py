from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "local_practice.py"
SPEC = importlib.util.spec_from_file_location("local_practice_test_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
local_practice = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(local_practice)


class LocalPracticeNativeProfileTests(unittest.TestCase):
    def test_variant_solution_controls_native_report_names(self) -> None:
        solution = Path("solution4.cu")

        self.assertEqual(
            local_practice.native_profile_path(solution, "nsys"), Path("solution4.nsys-rep")
        )
        self.assertEqual(
            local_practice.native_profile_path(solution, "ncu"), Path("solution4.ncu-rep")
        )

    @unittest.skipUnless(os.name == "nt", "Windows launcher behavior")
    def test_ncu_batch_launcher_resolves_real_executable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = root / "ncu.bat"
            executable = root / "target" / "windows-desktop-win7-x64" / "ncu.exe"
            executable.parent.mkdir(parents=True)
            batch.touch()
            executable.touch()

            with mock.patch.object(
                local_practice.shutil,
                "which",
                side_effect=lambda name: str(batch) if name == "ncu" else None,
            ):
                discovered = local_practice.find_native_tool("ncu")

        self.assertEqual(discovered, executable.resolve())

    def test_native_profile_becomes_stale_when_solution_hash_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            solution = Path(directory) / "solution.cu"
            report = solution.with_suffix(".nsys-rep")
            solution.write_text("old", encoding="utf-8")
            report.write_bytes(b"report")
            previous = {
                "native_profiles": {
                    "nsys": {
                        "status": "passed",
                        "file": report.name,
                        "size_bytes": report.stat().st_size,
                        "solution_sha256": "old-hash",
                    }
                }
            }

            carried = local_practice.carry_native_profiles(previous, solution, "new-hash")

        self.assertEqual(carried["nsys"]["status"], "stale")

    def test_failed_attempt_preserves_previous_success(self) -> None:
        native_profiles = {
            "ncu": {
                "status": "passed",
                "file": "solution.ncu-rep",
                "size_bytes": 123,
                "solution_sha256": "old",
            }
        }

        local_practice.record_native_failure(
            native_profiles,
            "ncu",
            Path("solution.cu"),
            "new",
            "detailed",
            "permission denied",
        )

        self.assertEqual(native_profiles["ncu"]["status"], "passed")
        self.assertEqual(native_profiles["ncu"]["size_bytes"], 123)
        self.assertEqual(native_profiles["ncu"]["last_attempt"]["status"], "failed")
        self.assertEqual(native_profiles["ncu"]["last_attempt"]["solution_sha256"], "new")

    def test_counter_permission_error_has_actionable_message(self) -> None:
        result = subprocess.CompletedProcess(
            ["ncu"], 1, stdout="", stderr="==ERROR== ERR_NVGPUCTRPERM"
        )

        message = local_practice.native_profile_error("ncu", result)

        self.assertIn("NVIDIA Control Panel", message)
        self.assertIn("ERR_NVGPUCTRPERM", message)

    def test_missing_tool_does_not_overwrite_previous_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            challenge = root / "easy" / "1_example"
            solution = challenge / "solution" / "cuda" / "solution.cu"
            library = root / "solution.dll"
            solution.parent.mkdir(parents=True)
            solution.write_text("source", encoding="utf-8")
            library.write_bytes(b"library")
            previous_report = solution.with_suffix(".nsys-rep")
            previous_report.write_bytes(b"previous success")

            with mock.patch.object(local_practice, "find_native_tool", return_value=None):
                with self.assertRaises(local_practice.PracticeError):
                    local_practice.generate_native_profile(
                        tool="nsys",
                        challenge_dir=challenge,
                        solution_path=solution,
                        library_path=library,
                        solution_sha256="a" * 64,
                        warmup=2,
                        seed=0,
                        ncu_set="detailed",
                    )

            self.assertEqual(previous_report.read_bytes(), b"previous success")

    def test_cuda_cache_key_includes_absolute_solution_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            challenge = root / "easy" / "1_example"
            first_solution = root / "first" / "solution.cu"
            second_solution = root / "second" / "solution.cu"
            compiler = root / "nvcc.exe"
            first_solution.parent.mkdir()
            second_solution.parent.mkdir()
            first_solution.write_text("same source", encoding="utf-8")
            second_solution.write_text("same source", encoding="utf-8")
            compiler.touch()

            def fake_compile(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                output = Path(command[command.index("-o") + 1])
                output.write_bytes(b"library")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with (
                mock.patch.object(local_practice, "BUILD_ROOT", root / "out"),
                mock.patch.object(local_practice.shutil, "which", return_value=str(compiler)),
                mock.patch.object(local_practice, "run_command", return_value="nvcc 13.3"),
                mock.patch.object(
                    local_practice.torch.cuda, "get_device_capability", return_value=(8, 9)
                ),
                mock.patch.object(local_practice.subprocess, "run", side_effect=fake_compile),
            ):
                first_library, first_details = local_practice.compile_cuda_solution(
                    challenge, first_solution
                )
                second_library, second_details = local_practice.compile_cuda_solution(
                    challenge, second_solution
                )

            self.assertNotEqual(first_library, second_library)
            self.assertNotEqual(first_details["build_cache_key"], second_details["build_cache_key"])
            self.assertEqual(first_details["nvcc_path"], str(compiler.resolve()))
            self.assertIn("-lineinfo", first_details["compile_flags"])

    def test_nsys_generation_atomically_places_nonempty_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            challenge = root / "easy" / "1_example"
            solution = challenge / "solution" / "cuda" / "solution4.cu"
            library = root / "solution.dll"
            solution.parent.mkdir(parents=True)
            solution.write_text("source", encoding="utf-8")
            library.write_bytes(b"library")
            tool_path = root / "nsys.exe"
            tool_path.touch()
            captured_command: list[str] = []

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                captured_command.extend(command)
                output_argument = next(item for item in command if item.startswith("--output="))
                temporary = Path(output_argument.split("=", 1)[1] + ".nsys-rep")
                temporary.write_bytes(b"native report")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            with (
                mock.patch.object(local_practice, "BUILD_ROOT", root / "out"),
                mock.patch.object(local_practice, "find_native_tool", return_value=tool_path),
                mock.patch.object(local_practice, "native_tool_version", return_value="2026.1"),
                mock.patch.object(local_practice.subprocess, "run", side_effect=fake_run),
            ):
                metadata = local_practice.generate_native_profile(
                    tool="nsys",
                    challenge_dir=challenge,
                    solution_path=solution,
                    library_path=library,
                    solution_sha256="a" * 64,
                    warmup=2,
                    seed=0,
                    ncu_set="detailed",
                )

            final_report = solution.with_suffix(".nsys-rep")
            self.assertEqual(final_report.read_bytes(), b"native report")
            self.assertEqual(metadata["file"], "solution4.nsys-rep")
            self.assertEqual(metadata["size_bytes"], len(b"native report"))
            self.assertIn("--capture-range=cudaProfilerApi", captured_command)
            self.assertIn("--cuda-memory-usage=true", captured_command)
            self.assertNotIn("nvcc", " ".join(captured_command).lower())

    def test_parser_rejects_native_profile_for_python_backend(self) -> None:
        arguments = [
            str(MODULE_PATH),
            "benchmark",
            "easy/52_silu",
            "--backend",
            "pytorch",
            "--nsys",
        ]

        with mock.patch.object(sys, "argv", arguments), self.assertRaises(SystemExit) as raised:
            local_practice.main()

        self.assertEqual(raised.exception.code, 2)

    def test_parser_rejects_ncu_set_without_ncu(self) -> None:
        arguments = [
            str(MODULE_PATH),
            "benchmark",
            "easy/52_silu",
            "--backend",
            "cuda",
            "--ncu-set",
            "full",
        ]

        with mock.patch.object(sys, "argv", arguments), self.assertRaises(SystemExit) as raised:
            local_practice.main()

        self.assertEqual(raised.exception.code, 2)

    def test_parser_accepts_no_profile_with_native_profiles(self) -> None:
        parser = local_practice.build_parser()

        arguments = parser.parse_args(
            [
                "benchmark",
                "easy/52_silu",
                "--backend",
                "cuda",
                "--no-profile",
                "--nsys",
                "--ncu",
            ]
        )

        self.assertTrue(arguments.no_profile)
        self.assertTrue(arguments.nsys)
        self.assertTrue(arguments.ncu)
        self.assertIsNone(arguments.ncu_set)

    def test_native_failure_marks_successful_benchmark_partial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            solution = root / "solution.cu"
            library = root / "solution.dll"
            solution.write_text("source", encoding="utf-8")
            library.write_bytes(b"library")
            captured_report: dict[str, object] = {}

            class FakeChallenge:
                def get_solve_signature(self) -> dict[str, tuple[object, str]]:
                    return {
                        "output": (
                            local_practice.ctypes.POINTER(local_practice.ctypes.c_float),
                            "out",
                        )
                    }

            def capture_report(_challenge: Path, _backend: str, report: dict[str, object]) -> None:
                captured_report.update(report)

            with (
                mock.patch.object(local_practice, "read_report", return_value={}),
                mock.patch.object(local_practice, "environment_info", return_value={}),
                mock.patch.object(local_practice.torch.cuda, "is_available", return_value=True),
                mock.patch.object(local_practice.torch, "manual_seed"),
                mock.patch.object(local_practice.torch.cuda, "manual_seed_all"),
                mock.patch.object(local_practice, "load_challenge", return_value=FakeChallenge()),
                mock.patch.object(local_practice, "solution_path_for", return_value=solution),
                mock.patch.object(
                    local_practice,
                    "create_invoker",
                    return_value=(lambda **_: None, {"library": str(library)}),
                ),
                mock.patch.object(
                    local_practice,
                    "run_benchmark",
                    return_value=({"performance_correctness": {"status": "passed"}}, None),
                ),
                mock.patch.object(
                    local_practice,
                    "generate_native_profile",
                    side_effect=local_practice.PracticeError("nsys missing"),
                ),
                mock.patch.object(local_practice, "write_report", side_effect=capture_report),
            ):
                success = local_practice.run_backend(
                    challenge_dir=local_practice.CHALLENGES_ROOT / "easy" / "52_silu",
                    backend="cuda",
                    mode="benchmark",
                    warmup=1,
                    repeat=1,
                    should_profile=False,
                    solution_override=None,
                    seed=0,
                    run_nsys=True,
                    run_ncu=False,
                    ncu_set="detailed",
                )

        self.assertFalse(success)
        self.assertEqual(captured_report["status"], "partial")
        self.assertIn("benchmark", captured_report)
        self.assertEqual(captured_report["native_profiles"]["nsys"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
