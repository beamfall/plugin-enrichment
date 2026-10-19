#!/usr/bin/env python3
"""Offline coverage and CI-depth contract gate."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FLOORS = ROOT / "testing/coverage-floor.json"
WORKFLOW = ROOT / ".github/workflows/conformance.yml"
GATE = ROOT / "script/gate.sh"


def load_floors(path: Path = FLOORS) -> dict[str, float]:
    data = json.loads(path.read_text())
    packages = data.get("packages")
    if data.get("schema_version") != 1 or not isinstance(packages, dict) or not packages:
        raise ValueError("coverage-floor.json must have schema_version 1 and packages")
    for package, floor in packages.items():
        if not isinstance(package, str) or not package or isinstance(floor, bool) or not isinstance(floor, (int, float)) or not 0 < floor <= 100:
            raise ValueError("coverage floors must map package names to numbers in (0, 100]")
    return {package: float(floor) for package, floor in packages.items()}


def coverage_errors(floors: dict[str, float], packages: set[str], measured: dict[str, float]) -> list[str]:
    errors = [f"discovered package has no coverage floor: {package}" for package in sorted(packages - floors.keys())]
    errors += [f"coverage floor targets missing package: {package}" for package in sorted(floors.keys() - packages)]
    for package in sorted(packages & floors.keys()):
        if package not in measured:
            errors.append(f"package has no measured coverage: {package}")
        elif measured[package] < floors[package]:
            errors.append(f"{package} coverage {measured[package]:.1f}% is below {floors[package]:.1f}%")
    return errors


def parse_total(output: str) -> float:
    match = re.search(r"^total:\s+\(statements\)\s+([0-9.]+)%$", output, re.MULTILINE)
    if not match:
        raise ValueError("go tool cover output has no total statement coverage")
    return float(match.group(1))


def run(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout


def check_coverage() -> None:
    floors = load_floors()
    packages = set(run("go", "list", "./...").splitlines())
    measured: dict[str, float] = {}
    with tempfile.TemporaryDirectory(prefix="enrichment-coverage-") as directory:
        for index, package in enumerate(sorted(packages)):
            profile = Path(directory) / f"coverage-{index}.out"
            subprocess.run(("go", "test", "-count=1", f"-coverprofile={profile}", package), cwd=ROOT, check=True)
            measured[package] = parse_total(run("go", "tool", "cover", f"-func={profile}"))
    errors = coverage_errors(floors, packages, measured)
    if errors:
        raise ValueError("\n".join(errors))
    for package in sorted(packages):
        print(f"coverage: {package} {measured[package]:.1f}% >= {floors[package]:.1f}%")


def job(text: str, name: str) -> str:
    match = re.search(rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [a-z][a-z0-9-]*:\n|\Z)", text)
    return match.group(1) if match else ""


def contract_errors(workflow: str, gate: str) -> list[str]:
    fast, race, required, fuzz = (job(workflow, name) for name in ("fast-conformance", "race", "conformance", "fuzz"))
    checks = (
        ("fast gate must run contracts", "run: script/gate.sh contracts" in fast),
        ("fast gate must run per-package coverage", "run: script/gate.sh coverage" in fast),
        ("race job must execute script/gate.sh race", "run: script/gate.sh race" in race),
        ("conformance must require fast and race jobs", "needs: [fast-conformance, race]" in required and "if: always()" in required),
        ("scheduled fuzz job must execute script/gate.sh fuzz", "github.event_name == 'schedule'" in fuzz and "run: script/gate.sh fuzz" in fuzz),
        ("local race mode must execute the race detector", "go test -race -count=1 ./..." in gate),
        ("fuzz mode must discover its required target", "go test -list '^Fuzz[A-Za-z0-9_]*$' ." in gate and "[ \"$targets\" != FuzzEmbedImageReader ]" in gate),
        ("fuzz mode must use a bounded target", "-fuzz '^FuzzEmbedImageReader$'" in gate and "-fuzztime=\"$fuzz_time\"" in gate),
    )
    return [error for error, ok in checks if not ok]


def check_contracts() -> None:
    workflow, gate = WORKFLOW.read_text(), GATE.read_text()
    errors = contract_errors(workflow, gate)
    if errors:
        raise ValueError("\n".join(errors))

    masking = coverage_errors({"high": 95, "low": 80}, {"high", "low"}, {"high": 100, "low": 70})
    if masking != ["low coverage 70.0% is below 80.0%"]:
        raise AssertionError("planted aggregate masking did not fail")
    if "race job must execute script/gate.sh race" not in contract_errors(workflow.replace("run: script/gate.sh race", "run: true", 1), gate):
        raise AssertionError("planted missing race execution did not fail")
    if "fuzz mode must discover its required target" not in contract_errors(workflow, gate.replace("go test -list '^Fuzz[A-Za-z0-9_]*$' .", "true", 1)):
        raise AssertionError("planted missing fuzz-target discovery did not fail")
    print("test-depth contracts: aggregate masking, race execution, and fuzz discovery mutations rejected")


def main() -> int:
    try:
        if len(sys.argv) != 2 or sys.argv[1] not in {"contracts", "coverage"}:
            raise ValueError("usage: test_depth_gate.py [contracts|coverage]")
        check_contracts() if sys.argv[1] == "contracts" else check_coverage()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"test-depth gate: FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
