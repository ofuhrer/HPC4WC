#!/usr/bin/env python3
"""Generate student and published solution files from .master folders."""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any


HPC4WC_METADATA_KEY = "hpc4wc"
SKIP_DIRS = {".ipynb_checkpoints", "__pycache__"}
STUDENT_BEGIN = "hpc4wc:student-begin"
STUDENT_LINE = "hpc4wc:student |"
STUDENT_END = "hpc4wc:student-end"
SOLUTION_BEGIN = "hpc4wc:solution-begin"
SOLUTION_END = "hpc4wc:solution-end"
DAY_DIR_RE = re.compile(r"day\d+$")


def load_notebook(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def notebook_bytes(notebook: dict[str, Any]) -> bytes:
    return (json.dumps(notebook, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def strip_hpc4wc_metadata(cell: dict[str, Any]) -> None:
    metadata = cell.get("metadata")
    if not isinstance(metadata, dict):
        return

    metadata.pop(HPC4WC_METADATA_KEY, None)
    if not metadata:
        cell["metadata"] = {}


def strip_top_level_hpc4wc_metadata(notebook: dict[str, Any]) -> None:
    metadata = notebook.get("metadata")
    if not isinstance(metadata, dict):
        return

    metadata.pop(HPC4WC_METADATA_KEY, None)


def strip_execution_state(cell: dict[str, Any]) -> None:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []


def notebook_student_mode(notebook: dict[str, Any]) -> str | None:
    metadata = notebook.get("metadata")
    if not isinstance(metadata, dict):
        return None
    hpc4wc = metadata.get(HPC4WC_METADATA_KEY) or {}
    if not isinstance(hpc4wc, dict):
        return None
    return hpc4wc.get("student")


def generate_student_notebook(master: dict[str, Any]) -> dict[str, Any]:
    student = copy.deepcopy(master)
    strip_top_level_hpc4wc_metadata(student)
    generated_cells = []

    for cell in student.get("cells", []):
        metadata = cell.get("metadata")
        hpc4wc = {}
        if isinstance(metadata, dict):
            hpc4wc = metadata.get(HPC4WC_METADATA_KEY) or {}

        if hpc4wc.get("student") == "remove":
            continue

        if "student_cell_type" in hpc4wc:
            cell["cell_type"] = hpc4wc["student_cell_type"]

        if "student_source" in hpc4wc:
            cell["source"] = hpc4wc["student_source"]

        strip_hpc4wc_metadata(cell)
        strip_execution_state(cell)
        generated_cells.append(cell)

    student["cells"] = generated_cells
    return student


def generate_solution_notebook(master: dict[str, Any]) -> dict[str, Any]:
    solution = copy.deepcopy(master)
    strip_top_level_hpc4wc_metadata(solution)
    for cell in solution.get("cells", []):
        strip_hpc4wc_metadata(cell)
    return solution


def split_marker_line(line: str, marker: str) -> tuple[str, str] | None:
    marker_index = line.find(marker)
    if marker_index < 0:
        return None

    prefix = line[:marker_index]
    suffix = line[marker_index + len(marker) :]
    return prefix, suffix


def student_payload(line: str) -> str:
    split = split_marker_line(line, STUDENT_LINE)
    if split is None:
        raise ValueError(f"malformed student marker line: {line.rstrip()}")

    prefix, suffix = split
    if "#" in prefix:
        prefix = prefix[: prefix.rfind("#")]
    elif "!" in prefix:
        prefix = prefix[: prefix.rfind("!")]
    elif "//" in prefix:
        prefix = prefix[: prefix.rfind("//")]
    if suffix.startswith(" "):
        suffix = suffix[1:]
    return prefix + suffix


def transform_marked_source(text: str, *, target: str) -> str:
    if target not in {"student", "solution"}:
        raise ValueError(f"unknown target: {target}")

    lines = text.splitlines(keepends=True)
    output: list[str] = []
    mode = "normal"

    for line in lines:
        if STUDENT_BEGIN in line:
            if mode != "normal":
                raise ValueError("nested hpc4wc source marker")
            mode = "student"
            continue

        if STUDENT_END in line:
            if mode != "student":
                raise ValueError("student-end without student-begin")
            mode = "normal"
            continue

        if SOLUTION_BEGIN in line:
            if mode != "normal":
                raise ValueError("nested hpc4wc source marker")
            mode = "solution"
            continue

        if SOLUTION_END in line:
            if mode != "solution":
                raise ValueError("solution-end without solution-begin")
            mode = "normal"
            continue

        if mode == "student":
            if target == "student":
                output.append(student_payload(line))
            elif STUDENT_LINE not in line:
                raise ValueError(f"unexpected line in student block: {line.rstrip()}")
            continue

        if mode == "solution":
            if target == "solution":
                output.append(line)
            continue

        output.append(line)

    if mode != "normal":
        raise ValueError(f"unterminated hpc4wc source marker: {mode}")

    return "".join(output)


def has_source_markers(path: Path) -> bool:
    if path.suffix == ".ipynb" or not path.is_file():
        return False

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    return STUDENT_BEGIN in text or SOLUTION_BEGIN in text


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def master_dir_for(day: Path) -> Path:
    master_dir = day / ".master"
    if not master_dir.is_dir():
        raise FileNotFoundError(f"missing master directory: {master_dir}")
    return master_dir


def resolve_day(day: Path) -> Path:
    if (day / ".master").is_dir():
        return day

    cwd = Path.cwd()
    if day.name == cwd.name and (cwd / ".master").is_dir():
        return cwd

    raise FileNotFoundError(f"missing master directory: {day / '.master'}")


def discover_days() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[1]
    days = [
        path
        for path in repo_root.iterdir()
        if path.is_dir() and DAY_DIR_RE.fullmatch(path.name) and (path / ".master").is_dir()
    ]
    if not days:
        raise FileNotFoundError(f"no day<n> directories with .master found in {repo_root}")
    return sorted(days)


def is_executable(path: Path) -> bool:
    return bool(path.stat().st_mode & 0o111)


def add_solution_outputs(
    source_path: Path,
    relative_path: Path,
    target_root: Path,
    outputs: dict[Path, bytes],
    executable_paths: set[Path],
) -> None:
    if should_skip(relative_path):
        return

    if source_path.is_symlink():
        resolved = source_path.resolve()
        if resolved.is_dir():
            for child in sorted(resolved.rglob("*")):
                child_relative = relative_path / child.relative_to(resolved)
                add_solution_outputs(child, child_relative, target_root, outputs, executable_paths)
        else:
            target_path = target_root / relative_path
            outputs[target_path] = resolved.read_bytes()
            if is_executable(resolved):
                executable_paths.add(target_path)
        return

    if source_path.is_dir():
        for child in sorted(source_path.iterdir()):
            add_solution_outputs(
                child,
                relative_path / child.name,
                target_root,
                outputs,
                executable_paths,
            )
        return

    target_path = target_root / relative_path
    if source_path.suffix == ".ipynb":
        outputs[target_path] = notebook_bytes(
            generate_solution_notebook(load_notebook(source_path))
        )
    elif has_source_markers(source_path):
        outputs[target_path] = transform_marked_source(
            source_path.read_text(encoding="utf-8"), target="solution"
        ).encode("utf-8")
    else:
        outputs[target_path] = source_path.read_bytes()

    if is_executable(source_path):
        executable_paths.add(target_path)


def expected_solution_outputs(day: Path) -> tuple[dict[Path, bytes], set[Path]]:
    master_dir = master_dir_for(day)
    outputs: dict[Path, bytes] = {}
    executable_paths: set[Path] = set()
    target_root = day / "solution"

    for master_path in sorted(master_dir.iterdir()):
        relative_path = master_path.relative_to(master_dir)
        add_solution_outputs(master_path, relative_path, target_root, outputs, executable_paths)

    return outputs, executable_paths


def expected_student_outputs(day: Path) -> tuple[dict[Path, bytes], set[Path]]:
    master_dir = master_dir_for(day)
    outputs: dict[Path, bytes] = {}
    executable_paths: set[Path] = set()

    for master_path in sorted(master_dir.rglob("*")):
        relative_path = master_path.relative_to(master_dir)
        if should_skip(relative_path) or master_path.is_dir():
            continue

        if master_path.suffix == ".ipynb":
            master_notebook = load_notebook(master_path)
            if notebook_student_mode(master_notebook) == "remove":
                continue
            outputs[day / relative_path] = notebook_bytes(
                generate_student_notebook(master_notebook)
            )
            continue

        if has_source_markers(master_path):
            target_path = day / relative_path
            outputs[target_path] = transform_marked_source(
                master_path.read_text(encoding="utf-8"), target="student"
            ).encode("utf-8")
            if is_executable(master_path):
                executable_paths.add(target_path)

    return outputs, executable_paths


def collect_files(root: Path) -> dict[Path, bytes]:
    if not root.exists():
        return {}

    if root.is_file():
        return {root: root.read_bytes()}

    return {
        path: path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and not should_skip(path.relative_to(root))
    }


def write_outputs(outputs: dict[Path, bytes], executable_paths: set[Path]) -> None:
    for path, content in sorted(outputs.items()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        if path in executable_paths:
            path.chmod(path.stat().st_mode | 0o111)


def output_conflicts(outputs: dict[Path, bytes]) -> list[str]:
    conflicts = []
    for path, content in sorted(outputs.items()):
        if path.exists() and path.is_dir():
            conflicts.append(f"would replace directory with file: {path}")
        elif path.exists() and path.read_bytes() != content:
            conflicts.append(f"would overwrite modified file: {path}")
    return conflicts


def solution_extra_files(day: Path, outputs: dict[Path, bytes]) -> list[Path]:
    solution_dir = day / "solution"
    existing = collect_files(solution_dir)
    return sorted(set(existing) - set(outputs))


def print_conflicts(conflicts: list[str]) -> None:
    for conflict in conflicts:
        print(conflict, file=sys.stderr)
    if conflicts:
        print("Use --force to overwrite generated targets.", file=sys.stderr)


def generate_day(
    day: Path,
    *,
    student: bool,
    solution: bool,
    force: bool,
) -> bool:
    master_dir = master_dir_for(day)
    student_outputs, student_executables = expected_student_outputs(day)
    solution_outputs, solution_executables = expected_solution_outputs(day)
    conflicts: list[str] = []

    if student:
        conflicts.extend(output_conflicts(student_outputs))

    if solution:
        conflicts.extend(output_conflicts(solution_outputs))
        conflicts.extend(
            f"would remove stale solution file: {path}"
            for path in solution_extra_files(day, solution_outputs)
        )

    if conflicts and not force:
        print_conflicts(conflicts)
        return False

    if student:
        write_outputs(student_outputs, student_executables)

    if solution:
        solution_dir = day / "solution"
        if solution_dir.exists():
            shutil.rmtree(solution_dir)
        write_outputs(solution_outputs, solution_executables)

    mode = "student and solution" if student and solution else "student" if student else "solution"
    print(f"generated {mode}: {day} from {master_dir}")
    return True


def compare_expected_outputs(label: str, outputs: dict[Path, bytes]) -> bool:
    ok = True
    for path, content in sorted(outputs.items()):
        if not path.exists():
            print(f"missing {label} file: {path}", file=sys.stderr)
            ok = False
        elif path.is_dir():
            print(f"{label} target is a directory: {path}", file=sys.stderr)
            ok = False
        elif path.read_bytes() != content:
            print(f"stale {label} file: {path}", file=sys.stderr)
            ok = False
    return ok


def check_day(day: Path, *, student: bool, solution: bool) -> bool:
    ok = True

    if student:
        student_outputs, _ = expected_student_outputs(day)
        ok = compare_expected_outputs("student", student_outputs) and ok

    if solution:
        solution_outputs, _ = expected_solution_outputs(day)
        ok = compare_expected_outputs("solution", solution_outputs) and ok
        for path in solution_extra_files(day, solution_outputs):
            print(f"stale solution file: {path}", file=sys.stderr)
            ok = False

    if ok:
        print(f"ok: {day}")
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate student files and visible solution bundles from .master."
    )
    parser.add_argument(
        "days",
        nargs="*",
        type=Path,
        help="Day directories to process. Defaults to all day<n> directories with .master.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--student",
        action="store_true",
        help="Generate only the student tree.",
    )
    mode.add_argument(
        "--solution",
        action="store_true",
        help="Generate only the visible solution/ directory.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check generated outputs without writing files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite generated targets and, with --solution, remove stale solution/ files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    student = args.student or not args.solution
    solution = args.solution or not args.student
    days = [resolve_day(day) for day in args.days] if args.days else discover_days()

    if args.check:
        ok = True
        for day in days:
            ok = check_day(day, student=student, solution=solution) and ok
        return 0 if ok else 1

    ok = True
    for day in days:
        ok = generate_day(day, student=student, solution=solution, force=args.force) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
