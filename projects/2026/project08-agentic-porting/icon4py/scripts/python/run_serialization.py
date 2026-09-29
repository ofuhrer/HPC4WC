#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --group scripts python3
#
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Run serialization jobs, collect ser_data and NAMELISTS, and archive outputs."""

from __future__ import annotations

import dataclasses
import itertools
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import TYPE_CHECKING

import f90nml
import typer

from icon4py.model.common.utils import fortran_config


if TYPE_CHECKING:
    from icon4py.model.testing import definitions
else:
    definitions = None
    dt_utils = None


cli = typer.Typer(no_args_is_help=True, help=__doc__)


@dataclasses.dataclass(frozen=True)
class SerializationSettings:
    comm_sizes: list[int]
    experiment_descriptions: list[definitions.ExperimentDescription]
    sbatch_partition: str
    sbatch_time: str
    sbatch_account: str
    sbatch_uenv: str
    sbatch_uenv_view: str
    job_poll_seconds: int
    iconf90_repo_dir: pathlib.Path
    icon4py_repo_dir: pathlib.Path
    build_dir: pathlib.Path
    runscript_dir: pathlib.Path
    experiments_dir: pathlib.Path
    output_root: pathlib.Path
    max_threads: int

    @classmethod
    def defaults(cls) -> SerializationSettings:
        # ======================================
        # START DEFAULT USER CONFIGURATION
        # ======================================
        # We hardcode the settings here for simplicity, but they could be
        # extended to be read from a config file or command-line arguments
        # if needed in the future.

        COMM_SIZES: list[int] = [1, 2, 4]

        EXPERIMENTS = [
            definitions.Experiments.MCH_CH_R04B09,
            definitions.Experiments.JW,
            definitions.Experiments.EXCLAIM_APE,
            definitions.Experiments.EXCLAIM_APE_AES,
            definitions.Experiments.GAUSS3D,
            definitions.Experiments.WEISMAN_KLEMP_TORUS,
        ]

        # Slurm settings
        SBATCH_PARTITION = "normal"
        SBATCH_TIME = "00:15:00"
        SBATCH_ACCOUNT = "cwd01"
        SBATCH_UENV = "icon/25.2:v3"
        SBATCH_UENV_VIEW = "default"
        JOB_POLL_SECONDS = 10

        # Directories (derived from this script's location in icon4py/)
        _THIS_FILE = pathlib.Path(__file__).resolve()
        ICON4PY_REPO_DIR = _THIS_FILE.parents[2]
        assert ICON4PY_REPO_DIR.name == "icon4py", (
            f"Expected icon4py repo dir, got {ICON4PY_REPO_DIR}"
        )
        ROOT_PROJECT_DIR = ICON4PY_REPO_DIR.parent
        ICONF90_REPO_DIR = ROOT_PROJECT_DIR / "icon"
        BUILD_DIR = ROOT_PROJECT_DIR / "build_serialize"
        RUNSCRIPTS_DIR = BUILD_DIR / "run"
        EXPERIMENTS_DIR = BUILD_DIR / "experiments"

        # Output location for copied ser_data and tarballs
        OUTPUT_ROOT = EXPERIMENTS_DIR / definitions.SERIALIZED_DATA_DIR

        # Maximum concurrent threads for running experiments
        MAX_THREADS: int = 5

        return cls(
            comm_sizes=COMM_SIZES,
            experiment_descriptions=EXPERIMENTS,
            sbatch_partition=SBATCH_PARTITION,
            sbatch_time=SBATCH_TIME,
            sbatch_account=SBATCH_ACCOUNT,
            sbatch_uenv=SBATCH_UENV,
            sbatch_uenv_view=SBATCH_UENV_VIEW,
            job_poll_seconds=JOB_POLL_SECONDS,
            iconf90_repo_dir=ICONF90_REPO_DIR,
            icon4py_repo_dir=ICON4PY_REPO_DIR,
            build_dir=BUILD_DIR,
            runscript_dir=RUNSCRIPTS_DIR,
            experiments_dir=EXPERIMENTS_DIR,
            output_root=OUTPUT_ROOT,
            max_threads=MAX_THREADS,
        )

        # ======================================
        # END DEFAULT USER CONFIGURATION
        # ======================================


def get_f90exp_name(experiment_description: definitions.ExperimentDescription) -> str:
    return f"{experiment_description.name}_sb"


def get_f90exp_dir(
    experiment_description: definitions.ExperimentDescription, *, settings: SerializationSettings
) -> pathlib.Path:
    return settings.experiments_dir / get_f90exp_name(experiment_description)


def get_nmlfile_name(experiment_description: definitions.ExperimentDescription) -> str:
    return f"exp.{get_f90exp_name(experiment_description)}"


def get_dumped_nmlfile_name(experiment_description: definitions.ExperimentDescription) -> str:
    return f"NAMELIST_{get_f90exp_name(experiment_description)}"


def get_slurmscript_name(experiment_description: definitions.ExperimentDescription) -> str:
    return f"{get_nmlfile_name(experiment_description)}.run"


def get_serdata_dst_dir(
    experiment_description: definitions.ExperimentDescription,
    comm_size: int,
    *,
    settings: SerializationSettings,
) -> pathlib.Path:
    """Get the destination directory for serialized data."""
    return settings.output_root / dt_utils.get_ranked_experiment_name_with_version(
        experiment_description, comm_size
    )


def get_tar_path(
    experiment_description: definitions.ExperimentDescription,
    comm_size: int,
    *,
    settings: SerializationSettings,
) -> pathlib.Path:
    """Get the path to the tar archive for the experiment."""
    return settings.output_root / dt_utils.get_experiment_archive_filename(
        experiment_description, comm_size
    )


def cleanup_exp_output(
    experiment_description: definitions.ExperimentDescription,
    comm_size: int,
    *,
    settings: SerializationSettings,
) -> None:
    """Clean up experiment output directories and archives.

    Deletes:
    - Experiment directory (exp_dir)
    - Serialized data destination directory (dest_dir)
    - Tar archive (tar_path)
    """
    # Delete experiment directory
    exp_dir = get_f90exp_dir(experiment_description, settings=settings)
    if exp_dir.exists():
        shutil.rmtree(exp_dir)

    # Delete serialized data destination directory
    dest_dir = get_serdata_dst_dir(experiment_description, comm_size, settings=settings)
    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    # Delete tar archive
    tar_path = get_tar_path(experiment_description, comm_size, settings=settings)
    if tar_path.exists():
        tar_path.unlink()


def run_command(
    cmd: list[str], check: bool = True, cwd: pathlib.Path | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True, capture_output=True, cwd=cwd)


def log_status(message: str) -> None:
    """Log a status message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def parse_extra_mpi_ranks(script_path: pathlib.Path, comm_size: int) -> int:
    """Parse extra MPI ranks from the Fortran script by summing num_* variables
    found in the &parallel_nml section.

    Looks for lines starting with:
        num_io_procs      =
        num_prefetch_proc =
        num_restart_procs =

    Supports both direct values (e.g., "num_io_procs = 1") and variable references
    (e.g., "num_io_procs = ${num_io_procs}") where the variable is defined elsewhere
    in the file.

    Args:
        script_path: Path to the script file to parse
        comm_size: Communicator size used for the run

    Returns:
        Sum of num_io_procs, num_prefetch_proc, and num_restart_procs values
    """
    content = script_path.read_text()
    extra_ranks = 0

    # First, parse all variable definitions from the entire file
    # Pattern: variable_name=value (outside of namelist sections)
    var_definitions: dict[str, int] = {}
    for match in re.finditer(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(\d+)", content, flags=re.MULTILINE):
        var_name = match.group(1)
        var_value = int(match.group(2))
        var_definitions[var_name] = var_value

    # Find the &parallel_nml section
    start_match = re.search(r"^\s*&parallel_nml\b.*$", content, flags=re.MULTILINE)
    if not start_match:
        return extra_ranks

    end_match = re.search(r"^\s*/\s*$", content[start_match.end() :], flags=re.MULTILINE)
    if not end_match:
        return extra_ranks

    section_start = start_match.start()
    section_end = start_match.end() + end_match.start()
    section = content[section_start:section_end]

    # Pattern to match num_* variables with either direct values or variable references
    var_names = ["num_io_procs", "num_prefetch_proc", "num_restart_procs"]

    for var_name in var_names:
        # num_io_procs only applies to MPI runs (no extra IO ranks for serial).
        if var_name == "num_io_procs" and comm_size <= 1:
            continue
        # Try to match direct integer value
        pattern_direct = rf"{var_name}\s*=\s*(\d+)"
        match = re.search(pattern_direct, section)
        if match:
            extra_ranks += int(match.group(1))
            continue

        # Try to match variable reference ${var_name}
        pattern_ref = rf"{var_name}\s*=\s*\$\{{([a-zA-Z_][a-zA-Z0-9_]*)\}}"
        match = re.search(pattern_ref, section)
        if match:
            ref_var_name = match.group(1)
            if ref_var_name in var_definitions:
                extra_ranks += var_definitions[ref_var_name]
            # If variable not found, silently ignore (value is 0)

    return extra_ranks


def update_slurm_variables(script_path: pathlib.Path, *, settings: SerializationSettings) -> None:
    """Update SBATCH directives in the Slurm script (partition, account, time, uenv, view)."""
    content = script_path.read_text()

    # Find the position after #SBATCH --job-name= line
    job_name_match = re.search(r"^#SBATCH\s+--job-name=.*$", content, flags=re.MULTILINE)
    if not job_name_match:
        raise RuntimeError("Could not find #SBATCH --job-name= line in script")

    # Prepare the new SBATCH lines to insert
    new_lines = (
        f"#SBATCH --partition={settings.sbatch_partition}\n"
        f"#SBATCH --account={settings.sbatch_account}\n"
        f"#SBATCH --time={settings.sbatch_time}\n"
        f"#SBATCH --uenv='{settings.sbatch_uenv}'\n"
        f"#SBATCH --view='{settings.sbatch_uenv_view}'"
    )

    # Remove existing partition, account, time, uenv, and view lines if they exist
    content = re.sub(r"^#SBATCH\s+--partition=.*$\n?", "", content, flags=re.MULTILINE)
    content = re.sub(r"^#SBATCH\s+--account=.*$\n?", "", content, flags=re.MULTILINE)
    content = re.sub(r"^#SBATCH\s+--time=.*$\n?", "", content, flags=re.MULTILINE)
    content = re.sub(r"^#SBATCH\s+--uenv=.*$\n?", "", content, flags=re.MULTILINE)
    content = re.sub(r"^#SBATCH\s+--view=.*$\n?", "", content, flags=re.MULTILINE)

    # Re-find job-name position in the cleaned text
    job_name_match = re.search(r"^(#SBATCH\s+--job-name=.*$)", content, flags=re.MULTILINE)
    if not job_name_match:
        raise RuntimeError("Could not find #SBATCH --job-name= line in script")

    # Insert new lines after the job-name line
    insertion_point = job_name_match.end()
    content = content[:insertion_point] + "\n" + new_lines + content[insertion_point:]

    script_path.write_text(content)


def update_slurm_ranks(script_path: pathlib.Path, mpi_ranks: int, extra_mpi_ranks: int = 0) -> None:
    """Update ranks in the Slurm script (ntasks-per-node and mpi_procs_pernode).

    Args:
        script_path: Path to the Slurm script
        mpi_ranks: Base number of MPI ranks
        extra_mpi_ranks: Additional ranks reserved for special operations (e.g., pre-fetch)
    """
    total_ranks = mpi_ranks + extra_mpi_ranks

    content = script_path.read_text()

    content = re.sub(
        r"^#SBATCH\s+--ntasks-per-node\s*=\s*\d+\s*$",
        f"#SBATCH --ntasks-per-node={total_ranks}",
        content,
        flags=re.MULTILINE,
    )

    content = re.sub(
        r"^:\s+\$\{no_of_nodes:=\d+\}\s+\$\{mpi_procs_pernode:=\d+\}\s*$",
        f": ${{no_of_nodes:=1}} ${{mpi_procs_pernode:={total_ranks}}}",
        content,
        flags=re.MULTILINE,
    )

    script_path.write_text(content)


def submit_job(script_path: pathlib.Path, *, settings: SerializationSettings) -> str:
    cmd = ["sbatch", str(script_path)]
    result = run_command(cmd, cwd=settings.runscript_dir)
    match = re.search(r"Submitted batch job\s+(\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Unable to parse job id from sbatch output: {result.stdout}")
    return match.group(1)


def normalize_state(raw_state: str) -> str:
    cleaned = raw_state.strip().upper()
    cleaned = cleaned.split("+")[0]
    cleaned = cleaned.split(":")[0]
    return cleaned


def get_job_state(job_id: str) -> str | None:
    """Query the state of a Slurm job.

    Returns the normalized job state string, or None if the job cannot be found
    or Slurm commands are unavailable.
    """
    # First try sacct for completed jobs
    try:
        result = run_command(["sacct", "-j", job_id, "--format=State", "--noheader"], check=False)
        if result.stdout.strip():
            return normalize_state(result.stdout.strip().splitlines()[0])
    except FileNotFoundError:
        # sacct command not found - continue to fallback
        pass

    # Fallback to squeue for running jobs
    try:
        result = run_command(["squeue", "-j", job_id, "-h", "-o", "%T"], check=False)
        if result.stdout.strip():
            return normalize_state(result.stdout.strip().splitlines()[0])
    except FileNotFoundError:
        # squeue command not found - Slurm may not be installed
        pass

    # Job not found in either command, or Slurm commands unavailable
    return None


def wait_for_success(job_id: str, *, settings: SerializationSettings) -> None:
    terminal_states = {
        "COMPLETED": True,
        "FAILED": False,
        "CANCELLED": False,
        "TIMEOUT": False,
        "OUT_OF_MEMORY": False,
        "NODE_FAIL": False,
    }

    while True:
        state = get_job_state(job_id)
        if state is None:
            time.sleep(settings.job_poll_seconds)
            continue

        if state in terminal_states:
            if terminal_states[state]:
                return
            raise RuntimeError(f"Job {job_id} finished unsuccessfully with state: {state}")

        time.sleep(settings.job_poll_seconds)


def copy_ser_data(
    experiment_description: definitions.ExperimentDescription,
    comm_size: int,
    job_id: str | None = None,
    *,
    settings: SerializationSettings,
) -> pathlib.Path:
    exp_dir = get_f90exp_dir(experiment_description, settings=settings)
    src_dir = exp_dir / "ser_data"
    if not src_dir.exists():
        raise FileNotFoundError(f"Missing ser_data folder: {src_dir}")

    # Flattened structure: OUTPUT_ROOT/mpitaskX_expname_vYY/
    dest_dir = get_serdata_dst_dir(experiment_description, comm_size, settings=settings)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)
    # Copy ser_data folder
    shutil.copytree(src_dir, dest_dir / definitions.SERIALIZED_DATA_SUBDIR)

    # Translate to json and copy NAMELIST_ICON_output_atm
    nml = f90nml.read(exp_dir / fortran_config.NAMELIST_ATM_FNAME)
    with (dest_dir / (fortran_config.ATM_DICT_FNAME)).open("w") as f:
        json.dump(nml.todict(), f, indent=4)
    # same for icon_master.namelist
    nml = f90nml.read(exp_dir / fortran_config.NAMELIST_MASTER_FNAME)
    with (dest_dir / (fortran_config.MASTER_DICT_FNAME)).open("w") as f:
        json.dump(nml.todict(), f, indent=4)
    # same for NAMELIST_expname
    nml = f90nml.read(exp_dir / get_dumped_nmlfile_name(experiment_description))
    with (dest_dir / (fortran_config.INPUT_DICT_FNAME)).open("w") as f:
        json.dump(nml.todict(), f, indent=4)

    # Copy NAMELIST files
    namelist_files = sorted(itertools.chain(exp_dir.glob("NAMELIST_*"), exp_dir.glob("*.namelist")))
    for src_file in namelist_files:
        if src_file.is_file():
            shutil.copy2(src_file, dest_dir / src_file.name)

    # Copy LOG file if available
    if job_id is not None:
        log_file = (
            settings.runscript_dir
            / f"LOG.{get_slurmscript_name(experiment_description)}.{job_id}.o"
        )
        if log_file.is_file():
            shutil.copy2(log_file, dest_dir / log_file.name)

    return dest_dir


def tar_folder(
    folder: pathlib.Path,
    experiment_description: definitions.ExperimentDescription,
    comm_size: int,
    *,
    settings: SerializationSettings,
) -> pathlib.Path:
    tar_path = get_tar_path(experiment_description, comm_size, settings=settings)

    with tarfile.open(tar_path, "w:gz") as tar:
        # Add only the contents of the folder (NAMELIST files and ser_data), not the folder itself
        for item in folder.iterdir():
            tar.add(item, arcname=item.name)

    return tar_path


def generate_update_script(
    experiment_description: definitions.ExperimentDescription, *, settings: SerializationSettings
) -> None:
    # copy namelist file from repo to build_dir
    shutil.copy2(
        settings.iconf90_repo_dir / "run" / get_nmlfile_name(experiment_description),
        settings.runscript_dir / get_nmlfile_name(experiment_description),
    )

    # run make_runscript
    cmd = ["./make_runscripts", get_f90exp_name(experiment_description)]
    _ = run_command(cmd, cwd=settings.build_dir)


def run_experiment(
    experiment_description: definitions.ExperimentDescription,
    comm_size: int,
    *,
    settings: SerializationSettings,
) -> None:
    """Execute a single experiment with the given communicator size."""
    try:
        # Clean up previous experiment output
        cleanup_exp_output(experiment_description, comm_size, settings=settings)

        generate_update_script(experiment_description, settings=settings)

        script_path = settings.runscript_dir / get_slurmscript_name(experiment_description)
        if not script_path.exists():
            raise FileNotFoundError(f"Missing slurm script: {script_path}")

        # Parse extra MPI ranks from the script
        extra_mpi_ranks = parse_extra_mpi_ranks(script_path, comm_size)

        log_status(
            f"Setting up {experiment_description.name} with {comm_size} ranks"
            + (f" + {extra_mpi_ranks} extra" if extra_mpi_ranks > 0 else "")
        )
        update_slurm_variables(script_path, settings=settings)
        update_slurm_ranks(script_path, comm_size, extra_mpi_ranks)

        log_status(f"Submitting {experiment_description.name} with {comm_size} ranks")
        job_id = submit_job(script_path, settings=settings)

        log_status(
            f"Waiting for {experiment_description.name} (ranks={comm_size}, job_id={job_id})"
        )
        wait_for_success(job_id, settings=settings)

        log_status(f"Copying ser_data for {experiment_description.name} with {comm_size} ranks")
        dest_dir = copy_ser_data(experiment_description, comm_size, job_id, settings=settings)

        log_status(f"Creating tar archive for {experiment_description.name} with {comm_size} ranks")
        tar_folder(dest_dir, experiment_description, comm_size, settings=settings)

        log_status(f"Completed {experiment_description.name} with {comm_size} ranks")
    except Exception as e:
        log_status(f"ERROR in {experiment_description.name} with {comm_size} ranks: {e}")
        raise


@cli.command()
def run_serialization() -> None:
    """Run the serialization experiment series."""

    # Import here to reduce startup time for the CLI
    global dt_utils, definitions  # noqa: PLW0603 [global-statement]
    import icon4py.model.testing.datatest_utils as dt_utils  # noqa: PLC0415 [import-outside-top-level]
    from icon4py.model.testing import definitions  # noqa: PLC0415 [import-outside-top-level]

    settings = SerializationSettings.defaults()
    settings.output_root.mkdir(parents=True, exist_ok=True)

    total_tasks = len(settings.experiment_descriptions) * len(settings.comm_sizes)
    log_status(
        f"Starting experiment series with {total_tasks} tasks ({len(settings.experiment_descriptions)} experiments x {len(settings.comm_sizes)} communicator sizes)"
    )

    for rank_idx, comm_size in enumerate(settings.comm_sizes, 1):
        num_experiments = len(settings.experiment_descriptions)
        log_status(
            f"Starting communicator size {rank_idx}/{len(settings.comm_sizes)}: {comm_size} ranks ({num_experiments} experiments parallel)"
        )

        with ThreadPoolExecutor(max_workers=settings.max_threads) as executor:
            futures = []

            for experiment_description in settings.experiment_descriptions:
                future = executor.submit(
                    run_experiment, experiment_description, comm_size, settings=settings
                )
                futures.append(future)

            log_status(
                f"All {len(futures)} experiments queued for {comm_size} ranks, waiting for completion..."
            )

            # Wait for all futures to complete and collect exceptions
            for future in futures:
                future.result()  # Re-raises any exceptions from the thread

        log_status(
            f"Completed communicator size {rank_idx}/{len(settings.comm_sizes)}: {comm_size} ranks"
        )

    log_status(f"All {total_tasks} tasks completed successfully!")


if __name__ == "__main__":
    sys.exit(cli())
