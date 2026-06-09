#!/usr/bin/env python3
"""Check generated course material without requiring the course runtime stack."""

from __future__ import annotations

import ast
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

import nbformat
import numpy as np
from IPython.core.inputtransformer2 import TransformerManager


MARKER_BYTES = (b"hpc4wc:", b'"hpc4wc"')
SKIP_DIRS = {".ipynb_checkpoints", "__pycache__", ".gt4py_cache"}
IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
HTML_IMAGE_RE = re.compile(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"']", re.IGNORECASE)
SUPPORTED_DAYS = {"day1", "day2", "day5"}


class Checker:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.failures: dict[str, list[str]] = defaultdict(list)
        self.transformer = TransformerManager()

    def log(self, message: str) -> None:
        print(message)

    def fail(self, group: str, message: str) -> None:
        self.failures[group].append(message)

    def run(self, days: Iterable[Path]) -> int:
        for day in days:
            self.check_day(day)

        if not self.failures:
            print("material checks passed")
            return 0

        for group, messages in self.failures.items():
            print(f"\n{group}", file=sys.stderr)
            for message in messages:
                print(f"  - {message}", file=sys.stderr)
        return 1

    def check_day(self, day: Path) -> None:
        self.log(f"checking {self.rel(day)}")
        if day.name not in SUPPORTED_DAYS:
            self.fail("configuration", f"unsupported day for checker: {day}")
            return
        if not day.is_dir():
            self.fail("configuration", f"missing day directory: {day}")
            return

        has_master = (day / ".master").is_dir()

        if has_master:
            self.check_generated_outputs(day)
        notebooks = list(self.notebooks(day))
        self.log(f"  notebooks: validating {len(notebooks)} notebook(s)")
        self.check_notebooks(notebooks)
        if has_master:
            self.log("  generated notebooks: checking metadata cleanup")
            self.check_generated_notebooks_are_clean(day)
            self.log("  generated files: checking source markers are absent")
            self.check_generated_files_have_no_markers(day)
        self.check_python_files(day)
        self.check_fortran_files(day)
        self.check_cpp_files(day)
        self.check_shell_files(day)
        self.log("  notebook assets: checking local image links")
        self.check_notebook_assets(notebooks)
        self.log("  smoke tests: running lightweight script checks")
        self.check_smoke_scripts(day)

    def check_generated_outputs(self, day: Path) -> None:
        self.log("  generation: checking student outputs are current")
        command = [
            sys.executable,
            str(self.repo_root / "tools" / "generate_from_master.py"),
            "--student",
            "--check",
            str(day),
        ]
        completed = subprocess.run(
            command,
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            self.fail("generation", detail or "generated outputs are stale")

    def notebooks(self, day: Path) -> Iterable[Path]:
        for path in sorted(day.rglob("*.ipynb")):
            if self.should_skip(path):
                continue
            yield path

    def check_notebooks(self, notebooks: Iterable[Path]) -> None:
        for path in notebooks:
            try:
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
            except Exception as exc:
                self.fail("notebooks", f"{self.rel(path)}: invalid notebook: {exc}")
                continue

            language = notebook.get("metadata", {}).get("language_info", {}).get("name")
            if language and language != "python":
                continue

            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                source = cell.get("source", "")
                try:
                    transformed = self.transformer.transform_cell(source)
                    ast.parse(transformed or "\n")
                except SyntaxError as exc:
                    text = exc.text.strip() if exc.text else exc.msg
                    self.fail(
                        "notebook syntax",
                        f"{self.rel(path)} cell {index}: line {exc.lineno}: {text}",
                    )
                except Exception as exc:
                    self.fail(
                        "notebook syntax",
                        f"{self.rel(path)} cell {index}: could not transform cell: {exc}",
                    )

    def check_generated_notebooks_are_clean(self, day: Path) -> None:
        generated_roots = [day, day / "solutions"]
        for root in generated_roots:
            for path in sorted(root.glob("*.ipynb")):
                try:
                    notebook = nbformat.read(path, as_version=4)
                except Exception:
                    continue
                for index, cell in enumerate(notebook.get("cells", [])):
                    metadata = cell.get("metadata", {})
                    if isinstance(metadata, dict) and "hpc4wc" in metadata:
                        self.fail(
                            "generated notebooks",
                            f"{self.rel(path)} cell {index}: contains hpc4wc metadata",
                        )
                metadata = notebook.get("metadata", {})
                if isinstance(metadata, dict) and "hpc4wc" in metadata:
                    self.fail(
                        "generated notebooks",
                        f"{self.rel(path)}: contains top-level hpc4wc metadata",
                    )

    def check_generated_files_have_no_markers(self, day: Path) -> None:
        for path in sorted(day.rglob("*")):
            if not path.is_file() or self.should_skip(path):
                continue
            relative = path.relative_to(day)
            if relative.parts and relative.parts[0] == ".master":
                continue
            try:
                content = path.read_bytes()
            except OSError as exc:
                self.fail("generated markers", f"{self.rel(path)}: could not read: {exc}")
                continue
            for marker in MARKER_BYTES:
                if marker in content:
                    self.fail(
                        "generated markers",
                        f"{self.rel(path)}: contains generated-source marker {marker.decode()}",
                    )

    def check_python_files(self, day: Path) -> None:
        python_files = [path for path in sorted(day.rglob("*.py")) if not self.should_skip(path)]
        self.log(f"  python: compiling {len(python_files)} file(s)")
        with tempfile.TemporaryDirectory(prefix="hpc4wc-pycompile-") as tmp:
            bytecode_dir = Path(tmp)
            for index, path in enumerate(python_files):
                try:
                    py_compile.compile(
                        str(path),
                        cfile=str(bytecode_dir / f"{index}.pyc"),
                        doraise=True,
                        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
                    )
                except py_compile.PyCompileError as exc:
                    self.fail("python syntax", f"{self.rel(path)}: {exc.msg}")

    def check_shell_files(self, day: Path) -> None:
        shell_files = [path for path in sorted(day.rglob("*.sh")) if not self.should_skip(path)]
        self.log(f"  shell: checking {len(shell_files)} script(s)")
        for path in shell_files:
            completed = subprocess.run(
                ["bash", "-n", str(path)],
                cwd=self.repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()
                self.fail("shell syntax", f"{self.rel(path)}: {detail}")

    def check_fortran_files(self, day: Path) -> None:
        fortran_files = [path for path in sorted(day.rglob("*.F90")) if not self.should_skip(path)]
        compiler = shutil.which("mpif90") or shutil.which("mpifort")
        if not compiler:
            self.log(f"  fortran: skipping {len(fortran_files)} file(s), no mpif90/mpifort found")
            return

        self.log(f"  fortran: compiling {len(fortran_files)} file(s) with {Path(compiler).name}")
        with tempfile.TemporaryDirectory(prefix="hpc4wc-fortran-") as tmp:
            build_dir = Path(tmp)
            module_sources = [path for path in fortran_files if path.name == "m_utils.F90"]
            other_sources = [path for path in fortran_files if path.name != "m_utils.F90"]
            for index, path in enumerate([*module_sources, *other_sources]):
                object_path = build_dir / f"{index}-{path.stem}.o"
                completed = subprocess.run(
                    [
                        compiler,
                        "-cpp",
                        "-ffree-line-length-none",
                        "-fopenmp",
                        "-J",
                        str(build_dir),
                        "-I",
                        str(build_dir),
                        "-c",
                        str(path),
                        "-o",
                        str(object_path),
                    ],
                    cwd=self.repo_root,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                if completed.returncode != 0:
                    detail = (completed.stderr or completed.stdout).strip()
                    self.fail("fortran syntax", f"{self.rel(path)}: {detail}")

    def check_cpp_files(self, day: Path) -> None:
        cpp_files = [path for path in sorted(day.rglob("*.cpp")) if not self.should_skip(path)]
        compiler = self.cpp_compiler()
        if not compiler:
            self.log(f"  c++: skipping {len(cpp_files)} file(s), no OpenMP-capable compiler found")
            return

        self.log(f"  c++: compiling {len(cpp_files)} file(s) with {Path(compiler).name}")
        with tempfile.TemporaryDirectory(prefix="hpc4wc-cpp-") as tmp:
            build_dir = Path(tmp)
            for index, path in enumerate(cpp_files):
                object_path = build_dir / f"{index}-{path.stem}.o"
                completed = subprocess.run(
                    [
                        compiler,
                        "-std=c++17",
                        "-fopenmp",
                        "-I",
                        str(path.parent),
                        "-c",
                        str(path),
                        "-o",
                        str(object_path),
                    ],
                    cwd=self.repo_root,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                if completed.returncode != 0:
                    detail = (completed.stderr or completed.stdout).strip()
                    self.fail("c++ syntax", f"{self.rel(path)}: {detail}")

    def check_notebook_assets(self, notebooks: Iterable[Path]) -> None:
        for path in notebooks:
            try:
                notebook = nbformat.read(path, as_version=4)
            except Exception:
                continue
            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") != "markdown":
                    continue
                for link in self.local_image_links(cell.get("source", "")):
                    asset = (path.parent / link).resolve()
                    if not asset.exists():
                        self.fail(
                            "assets",
                            f"{self.rel(path)} cell {index}: missing image {link}",
                        )

    def check_smoke_scripts(self, day: Path) -> None:
        if day.name == "day1":
            self.check_day1_smoke_scripts(day)
        elif day.name == "day2":
            self.check_day2_smoke_scripts(day)
        elif day.name == "day5":
            self.check_day5_smoke_scripts(day)

    def check_day1_smoke_scripts(self, day: Path) -> None:
        with tempfile.TemporaryDirectory(prefix="hpc4wc-day1-") as tmp:
            tmpdir = Path(tmp)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(day / "stencil2d.py"),
                    "--nx=8",
                    "--ny=8",
                    "--nz=4",
                    "--num_iter=1",
                ],
                cwd=tmpdir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()
                self.fail("smoke", f"{self.rel(day / 'stencil2d.py')}: {detail}")
            for name in ("in_field.npy", "out_field.npy"):
                if not (tmpdir / name).is_file():
                    self.fail("smoke", f"{self.rel(day / 'stencil2d.py')}: missing {name}")

    def check_day5_smoke_scripts(self, day: Path) -> None:
        with tempfile.TemporaryDirectory(prefix="hpc4wc-day5-") as tmp:
            tmpdir = Path(tmp)
            baseline = subprocess.run(
                [
                    sys.executable,
                    str(day / "stencil2d.py"),
                    "--nx=8",
                    "--ny=8",
                    "--nz=4",
                    "--num_iter=1",
                ],
                cwd=tmpdir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if baseline.returncode != 0:
                detail = (baseline.stderr or baseline.stdout).strip()
                self.fail("smoke", f"{self.rel(day / 'stencil2d.py')}: {detail}")
            for name in ("in_field.npy", "out_field.npy"):
                if not (tmpdir / name).is_file():
                    self.fail("smoke", f"{self.rel(day / 'stencil2d.py')}: missing {name}")

        self.check_compare_fields(day / "compare_fields.py")
        solution_compare = day / "solutions" / "compare_fields.py"
        if solution_compare.exists():
            self.check_compare_fields(solution_compare)

    def check_compare_fields(self, script: Path) -> None:
        with tempfile.TemporaryDirectory(prefix="hpc4wc-compare-") as tmp:
            tmpdir = Path(tmp)
            a = tmpdir / "a.npy"
            b = tmpdir / "b.npy"
            c = tmpdir / "c.npy"
            np.save(a, np.array([1.0, 2.0]))
            np.save(b, np.array([1.0, 2.0]))
            np.save(c, np.array([1.0, 3.0]))

            equal = subprocess.run(
                [sys.executable, str(script), f"--src={a}", f"--trg={b}"],
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if equal.returncode != 0:
                detail = (equal.stderr or equal.stdout).strip()
                self.fail("smoke", f"{self.rel(script)} equal arrays failed: {detail}")

            different = subprocess.run(
                [sys.executable, str(script), f"--src={a}", f"--trg={c}"],
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if different.returncode == 0:
                self.fail("smoke", f"{self.rel(script)} mismatch arrays exited with 0")

    def check_day2_smoke_scripts(self, day: Path) -> None:
        compiler = shutil.which("mpif90") or shutil.which("mpifort")
        if not compiler:
            self.log("    day2 stencil smoke: skipping, no mpif90/mpifort found")
            return

        solution_source_root = day / ".master"
        if not solution_source_root.is_dir():
            solution_source_root = day / ".solutions"

        variants = ["base", "kparallel", "jparallel", "fusion", "kparallel-halo"]
        with tempfile.TemporaryDirectory(prefix="hpc4wc-day2-") as tmp:
            tmpdir = Path(tmp)
            module_object = tmpdir / "m_utils.o"
            module_compile = subprocess.run(
                [
                    compiler,
                    "-cpp",
                    "-ffree-line-length-none",
                    "-fopenmp",
                    "-J",
                    str(tmpdir),
                    "-I",
                    str(tmpdir),
                    "-c",
                    str(day / "m_utils.F90"),
                    "-o",
                    str(module_object),
                ],
                cwd=tmpdir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if module_compile.returncode != 0:
                detail = (module_compile.stderr or module_compile.stdout).strip()
                self.fail("smoke", f"{self.rel(day / 'm_utils.F90')} compile failed: {detail}")
                return

            outputs: dict[str, tuple[int, np.ndarray]] = {}
            for variant in variants:
                source = solution_source_root / f"stencil2d-{variant}.F90"
                executable = tmpdir / f"stencil2d-{variant}.x"
                completed = subprocess.run(
                    [
                        compiler,
                        "-cpp",
                        "-ffree-line-length-none",
                        "-fopenmp",
                        "-J",
                        str(tmpdir),
                        "-I",
                        str(tmpdir),
                        str(source),
                        str(module_object),
                        "-o",
                        str(executable),
                    ],
                    cwd=tmpdir,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                if completed.returncode != 0:
                    detail = (completed.stderr or completed.stdout).strip()
                    self.fail("smoke", f"{self.rel(source)} compile failed: {detail}")
                    continue

                run = subprocess.run(
                    [
                        str(executable),
                        "--nx",
                        "8",
                        "--ny",
                        "8",
                        "--nz",
                        "4",
                        "--num_iter",
                        "2",
                    ],
                    cwd=tmpdir,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                if run.returncode != 0:
                    detail = (run.stderr or run.stdout).strip()
                    self.fail("smoke", f"{self.rel(source)} run failed: {detail}")
                    continue

                output_path = tmpdir / "out_field.dat"
                if not output_path.is_file():
                    self.fail("smoke", f"{self.rel(source)} did not write out_field.dat")
                    continue
                try:
                    outputs[variant] = self.read_fortran_field(output_path)
                except Exception as exc:
                    self.fail("smoke", f"{self.rel(source)} output unreadable: {exc}")
                output_path.unlink(missing_ok=True)
                (tmpdir / "in_field.dat").unlink(missing_ok=True)

            baseline = outputs.get("base")
            if baseline is None:
                return
            baseline_halo, baseline_field = baseline
            interior = (
                slice(None),
                slice(baseline_halo, -baseline_halo),
                slice(baseline_halo, -baseline_halo),
            )
            for variant, output in outputs.items():
                if variant == "base":
                    continue
                _halo, field = output
                if not np.allclose(field[interior], baseline_field[interior], rtol=1e-6, atol=1e-7):
                    diff = float(np.max(np.abs(field[interior] - baseline_field[interior])))
                    self.fail("smoke", f"day2 stencil2d-{variant}.F90 differs from base: {diff}")

    def local_image_links(self, source: str) -> Iterable[str]:
        for raw_link in IMAGE_LINK_RE.findall(source) + HTML_IMAGE_RE.findall(source):
            link = raw_link.strip().strip("<>")
            if not link:
                continue
            link = link.split()[0]
            parsed = urlparse(link)
            if parsed.scheme or parsed.netloc or link.startswith("#"):
                continue
            if parsed.fragment and not parsed.path:
                continue
            yield unquote(parsed.path)

    def should_skip(self, path: Path) -> bool:
        return any(part in SKIP_DIRS for part in path.parts)

    def cpp_compiler(self) -> str | None:
        for candidate in ("g++-15", "g++-14", "g++-13", "g++-12", "g++"):
            compiler = shutil.which(candidate)
            if not compiler:
                continue
            with tempfile.TemporaryDirectory(prefix="hpc4wc-openmp-cpp-") as tmp:
                tmpdir = Path(tmp)
                source = tmpdir / "openmp.cpp"
                source.write_text("#include <omp.h>\nint main(){return omp_get_max_threads()<1;}\n")
                completed = subprocess.run(
                    [
                        compiler,
                        "-std=c++17",
                        "-fopenmp",
                        str(source),
                        "-o",
                        str(tmpdir / "openmp"),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                if completed.returncode == 0:
                    return compiler
        return None

    def read_fortran_field(self, path: Path) -> tuple[int, np.ndarray]:
        header = np.fromfile(path, dtype=np.int32, count=6)
        if header.size != 6:
            raise ValueError("missing binary field header")
        rank, nbits, num_halo, nx, ny, nz = header
        dtype = np.float32 if nbits == 32 else np.float64
        data = np.fromfile(path, dtype=dtype)
        offset = (3 + int(rank)) * 32 // int(nbits)
        field = data[offset:].reshape((int(nz), int(ny), int(nx)))
        return int(num_halo), field

    def rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.repo_root))
        except ValueError:
            return str(path)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("Usage: check_material.py <day> [<day> ...]", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    days = []
    for arg in args:
        path = Path(arg)
        if not path.is_absolute():
            path = repo_root / path
        days.append(path.resolve())

    return Checker(repo_root).run(days)


if __name__ == "__main__":
    raise SystemExit(main())
