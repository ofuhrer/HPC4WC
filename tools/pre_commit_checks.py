#!/usr/bin/env python3
"""Course-specific pre-commit guardrails."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


DAY_SOLUTION_RE = re.compile(r"^(day\d+)/solution/")


def run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def staged_paths(repo_root: Path) -> list[str]:
    completed = run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=repo_root,
    )
    if completed.returncode != 0:
        print(completed.stderr or completed.stdout, file=sys.stderr)
        raise SystemExit(completed.returncode)
    return [line for line in completed.stdout.splitlines() if line]


def indexed_paths(repo_root: Path, pathspec: str) -> list[str]:
    completed = run(["git", "ls-files", pathspec], cwd=repo_root)
    if completed.returncode != 0:
        print(completed.stderr or completed.stdout, file=sys.stderr)
        raise SystemExit(completed.returncode)
    return [line for line in completed.stdout.splitlines() if line]


def published_solution_days(repo_root: Path) -> list[str]:
    paths = [*indexed_paths(repo_root, "day*/solution/*"), *staged_paths(repo_root)]
    days = {match.group(1) for path in paths if (match := DAY_SOLUTION_RE.match(path)) is not None}
    return sorted(days)


def check_published_solutions(repo_root: Path) -> bool:
    days = published_solution_days(repo_root)
    if not days:
        return True

    completed = run(
        [
            sys.executable,
            str(repo_root / "tools" / "generate_from_master.py"),
            "--solution",
            "--check",
            *days,
        ],
        cwd=repo_root,
    )
    if completed.returncode == 0:
        return True

    print("Published solution bundles are stale:", file=sys.stderr)
    print(completed.stdout, end="", file=sys.stderr)
    print(completed.stderr, end="", file=sys.stderr)
    return False


def check_generated_students(repo_root: Path) -> bool:
    completed = run(
        [
            sys.executable,
            str(repo_root / "tools" / "generate_from_master.py"),
            "--student",
            "--check",
            "day1",
            "day2",
            "day3",
            "day4",
            "day5",
        ],
        cwd=repo_root,
    )
    if completed.returncode == 0:
        return True

    print(completed.stdout, end="", file=sys.stderr)
    print(completed.stderr, end="", file=sys.stderr)
    return False


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    checks = [
        check_published_solutions(repo_root),
        check_generated_students(repo_root),
    ]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
