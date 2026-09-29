#!/usr/bin/env python3
import sys
import re
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from tabulate import tabulate
from pathlib import Path
from PandaSQLite import PandaSQLiteDB

def parse_hardware_info(filepath):
    hardware_summary = {}
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

            cpu_info_start = -1
            mem_info_start = -1
            for i, line in enumerate(lines):
                if "CPU Information:" in line:
                    cpu_info_start = i
                if "Memory Information:" in line:
                    mem_info_start = i
                if cpu_info_start != -1 and mem_info_start != -1:
                    break

            if cpu_info_start != -1:
                for i in range(cpu_info_start + 1, len(lines)):
                    line = lines[i].strip()
                    if (not line or "Memory Information:" in line or
                        "GPU Information:" in line or "Detected total" in line):
                        break
                    if "Model name:" in line and 'CPU Model Name' not in hardware_summary:
                        hardware_summary['CPU Model Name'] = line.split(':', 1)[1].strip()
                    elif "Architecture:" in line:
                        hardware_summary['Architecture'] = line.split(':', 1)[1].strip()
                    elif "CPU(s):" in line:
                        hardware_summary['Total CPUs'] = line.split(':', 1)[1].strip()
                    elif "Socket(s):" in line:
                        hardware_summary['Sockets'] = line.split(':', 1)[1].strip()
                    elif "Core(s) per socket:" in line:
                        hardware_summary['Cores per Socket'] = line.split(':', 1)[1].strip()
                    elif "L3 cache:" in line:
                        hardware_summary['L3 Cache'] = line.split(':', 1)[1].strip()

            if mem_info_start != -1:
                for i in range(mem_info_start + 1, len(lines)):
                    line = lines[i].strip()
                    if not line or "GPU Information:" in line or "Detected total" in line:
                        break
                    if "Mem:" in line:
                        parts = line.split()
                        if len(parts) > 1:
                            hardware_summary['Total Memory'] = parts[1]
                        break
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while parsing hardware info: {e}")
        sys.exit(1)
    return hardware_summary

def hardware_df(hardware_summary: dict) -> pd.DataFrame:
    if not hardware_summary:
        return pd.DataFrame(columns=[
            "CPU Model Name","Architecture","Total CPUs","Sockets",
            "Cores per Socket","L3 Cache","Total Memory"
        ])
    return pd.DataFrame([hardware_summary])

def parse_benchmark_log(filepath):
    data = defaultdict(lambda: defaultdict(list))
    run_arguments = {}
    current_program = None
    temp_run_config = None

    try:
        with open(filepath, 'r') as f:
            for line in f:
                args_match = re.search(r'initial nx=ny=(\d+), max_iter=(\d+), repetitions=(\d+)', line)
                if args_match and not run_arguments:
                    run_arguments['initial_nx'] = int(args_match.group(1))
                    run_arguments['initial_ny'] = int(args_match.group(1))
                    run_arguments['max_iter'] = int(args_match.group(2))
                    run_arguments['repetitions'] = int(args_match.group(3))
                    continue

                program_match = re.search(
                    r'--- Processing (C\+\+|Python) program: \./CPU/(?:cpp/build|numba)/([^ "]+)',
                    line
                )
                if program_match:
                    current_program = program_match.group(2)
                    temp_run_config = None
                    continue

                run_config_match = re.search(r'OMP_NUM_THREADS=(\d+), MPI Processes=(\d+), NX=(\d+)', line)
                if run_config_match:
                    threads = int(run_config_match.group(1))
                    mpi_procs = int(run_config_match.group(2))
                    nx = int(run_config_match.group(3))
                    temp_run_config = (threads, mpi_procs, nx)
                    continue

                time_match = re.search(r'Elapsed time:\s*([\d.]+)\s*seconds', line)
                if time_match and current_program and temp_run_config is not None:
                    elapsed_time = float(time_match.group(1))
                    data[current_program][temp_run_config].append(elapsed_time)
                    temp_run_config = None
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while parsing the file: {e}")
        sys.exit(1)

    return data, run_arguments

def analyze_data_to_df(parsed_data) -> pd.DataFrame:
    rows = []
    for program, config_data in parsed_data.items():
        sorted_configs = sorted(config_data.keys())

        base_median_time = None
        if sorted_configs:
            base_config_key = sorted_configs[0]
            base_times = np.array(config_data[base_config_key])
            base_median_time = np.median(base_times) if len(base_times) else None
            if len(base_times) < 2:
                print(f"Warning: Not enough samples for meaningful bootstrap CI for base configuration "
                      f"(Program: {program}, Config: {base_config_key}). At least 2 samples required.")

        for (threads, mpi_procs, nx) in sorted_configs:
            times = np.array(config_data[(threads, mpi_procs, nx)])
            median_time = float(np.median(times)) if len(times) else np.nan
            ci_low = np.nan
            ci_high = np.nan

            if len(times) >= 2:
                try:
                    res = bootstrap(
                        (times,), statistic=np.median,
                        confidence_level=0.95, n_resamples=9999,
                        method='percentile', random_state=42
                    )
                    ci_low = float(res.confidence_interval.low)
                    ci_high = float(res.confidence_interval.high)
                except Exception as e:
                    print(f"Warning: Could not perform bootstrap for {program} with "
                          f"threads={threads}, MPI={mpi_procs}, NX={nx}. Error: {e}")
            else:
                print(f"Warning: Not enough samples for meaningful bootstrap CI "
                      f"(Program: {program}, Threads: {threads}, MPI: {mpi_procs}, NX: {nx}). "
                      f"At least 2 samples required.")

            speedup = (base_median_time / median_time) if (base_median_time and median_time and median_time > 0) else 1.0

            rows.append({
                'Program': program,
                'Threads': threads,
                'MPI Procs': mpi_procs,
                'NX': nx,
                'Median Time (s)': median_time,
                '95% CI Lower (s)': ci_low,
                '95% CI Upper (s)': ci_high,
                'Speedup': speedup
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=['Program', 'Threads', 'MPI Procs', 'NX']).reset_index(drop=True)
    return df

def run_args_df(run_arguments: dict) -> pd.DataFrame:
    if not run_arguments:
        return pd.DataFrame(columns=['Initial nx', 'Initial ny', 'Max Iterations', 'Repetitions'])
    return pd.DataFrame([{
        'Initial nx': run_arguments.get('initial_nx', 'N/A'),
        'Initial ny': run_arguments.get('initial_ny', 'N/A'),
        'Max Iterations': run_arguments.get('max_iter', 'N/A'),
        'Repetitions': run_arguments.get('repetitions', 'N/A'),
    }])

def print_with_tabulate(df: pd.DataFrame, title: str = None, floatfmt=".4f", tablefmt="github"):
    if title:
        print(title)
    if df.empty:
        print("(no data)\n")
        return
    print(tabulate(df, headers="keys", tablefmt=tablefmt, showindex=False, floatfmt=floatfmt))
    print()


def print_latex_summary_table(df: pd.DataFrame, floatfmt=".2f"):
    """
    Prints the benchmark summary as a single LaTeX table with a nested structure.
    Algorithm name is a main column, with threads as horizontal
    headers and metrics (Median, 95% CI Lower, 95% CI Upper, Speedup) as rows.
    """
    if df.empty:
        print("% (no data for LaTeX summary)\n")
        return

    # Prepare data for the nested structure
    unique_programs = df['Program'].unique()
    all_unique_threads = sorted(df['Threads'].unique().tolist())
    num_thread_cols = len(all_unique_threads)

    # LaTeX document preamble
    latex_output = []
    latex_output.append("\n\\begin{table}[h!]")
    latex_output.append("\\centering")
    latex_output.append("\\caption{Comprehensive Benchmark Performance Summary}")
    latex_output.append("\\label{tab:benchmark_summary}")

    # Define tabular environment columns: Program (l), Metric (l), and Thread columns (r per thread)
    # Total columns = 2 (Program, Metric) + num_thread_cols
    col_def = "l l " + " ".join(["r"] * num_thread_cols)
    latex_output.append(f"\\begin{{tabular}}{{{col_def}}}")
    latex_output.append("\\toprule")

    # First header row: Program, empty cell for metrics, Multicolumn for Threads
    # Note: \multirow takes number of rows, then width (can be * for natural width), then content.
    latex_output.append(
        f"\\multirow{{2}}{{*}}{{Algorithm}} & "
        f"\\multirow{{2}}{{*}}{{}} & " # Empty cell for the metric label column header
        f"\\multicolumn{{{num_thread_cols}}}{{c}}{{{'Threads'}}} \\\\"
    )
    # Second header row: Individual Thread values under the 'Threads' multicolumn
    latex_output.append("\\cmidrule{3-" + str(2 + num_thread_cols) + "}") # Spans from 3rd col to last
    thread_headers = [str(t) for t in all_unique_threads]
    latex_output.append(f" & & {' & '.join(thread_headers)} \\\\")
    latex_output.append("\\midrule")

    # Metrics rows for display
    metrics_display_labels = [
        'Median (s)',
        '95\\% CI Lower (s)',
        '95\\% CI Upper (s)',
        'Speedup'
    ]
    num_metric_rows_per_program_block = len(metrics_display_labels) # This is now 4

    # Data rows for each program
    for i, program in enumerate(unique_programs):
        program_data = df[df['Program'] == program].copy()

        # Group by threads, calculate median for time metrics and CI, and max for speedup
        grouped_data = program_data.groupby('Threads').agg(
            median_time=('Median Time (s)', 'median'),
            ci_lower=('95% CI Lower (s)', 'median'),
            ci_upper=('95% CI Upper (s)', 'median'),
            speedup=('Speedup', 'max')
        ).reset_index()

        # Store all metric values for each thread for easy lookup
        all_thread_metrics = {}
        for _, row in grouped_data.iterrows():
            thread = row['Threads']
            all_thread_metrics[thread] = {
                'mu': row['median_time'],
                'ci_lower': row['ci_lower'],
                'ci_upper': row['ci_upper'],
                'speedup': row['speedup']
            }

        # Format and append rows for the current program block
        # Use program.replace('_', '\\_') to escape underscores for LaTeX
        escaped_program = program.replace('_', '\\_')
        program_name_cell = f"\\multirow{{{num_metric_rows_per_program_block}}}{{*}}{{{escaped_program}}}"

        # Loop through each metric type (mu, ci_lower, ci_upper, speedup)
        for row_idx, metric_label in enumerate(metrics_display_labels):
            current_row_values = []
            for thread_val in all_unique_threads:
                metric_data = all_thread_metrics.get(thread_val)
                value = '---'
                if metric_data:
                    if metric_label == 'Median (s)':
                        value = f"{metric_data['mu']:{floatfmt}}"
                    elif metric_label == '95\\% CI Lower (s)':
                        value = f"{metric_data['ci_lower']:{floatfmt}}"
                    elif metric_label == '95\\% CI Upper (s)':
                        value = f"{metric_data['ci_upper']:{floatfmt}}"
                    elif metric_label == 'Speedup':
                        value = f"{metric_data['speedup']:{floatfmt}}"
                current_row_values.append(value)

            # Construct the full LaTeX row
            if row_idx == 0: # First metric row includes Program cell, and its metric label
                latex_output.append(
                    f"{program_name_cell} & {metric_label} & " + " & ".join(current_row_values) + " \\\\"
                )
            else: # Subsequent metric rows only need their metric label and thread-specific values
                latex_output.append(
                    f" & {metric_label} & " + " & ".join(current_row_values) + " \\\\"
                )

        if i < len(unique_programs) - 1: # Add midrule between programs
            latex_output.append("\\midrule")

    latex_output.append("\\bottomrule")
    latex_output.append("\\end{tabular}")
    latex_output.append("\\end{table}")

    print("\n".join(latex_output))
    print()

def save_to_sql(db_path: Path, table_prefix: str, df_hw: pd.DataFrame, df_args: pd.DataFrame, df_summary: pd.DataFrame):
    """
    Save DataFrames into a single SQLite file using PandaSQLite.
    Table names are derived from the input filename prefix.
    """
    db = PandaSQLiteDB(str(db_path))
    tables = {
        f"{table_prefix}_hardware": df_hw,
        f"{table_prefix}_run_args": df_args,
        f"{table_prefix}_summary": df_summary,
    }
    for name, df in tables.items():
        if not df.empty:
            # Note: PandaSQLiteDB.create_table will create a new table.
            # If you expect repeated runs, consider using unique prefixes or
            # dropping/replacing tables beforehand if the library supports it.
            db.create_table(name, df)
            print(f"Wrote table '{name}' to {db_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results → print (tabulate) → store to SQLite with PandaSQLite.")
    parser.add_argument("log_file", help="Path to the benchmark log")
    parser.add_argument("--db-path", default="benchmarks.sqlite",
                        help="SQLite output file (default: benchmarks.sqlite)")
    parser.add_argument("--floatfmt", default=".4f", help="Float format for printed tables (default: .4f)")
    args = parser.parse_args()

    log_file_path = Path(args.log_file)
    if not log_file_path.exists():
        print(f"Error: File not found at '{log_file_path}'")
        sys.exit(1)

    # Parse
    hw_info = parse_hardware_info(log_file_path)
    data_dict, run_args = parse_benchmark_log(log_file_path)

    # DataFrames
    df_hw = hardware_df(hw_info)
    df_args = run_args_df(run_args)
    df_summary = analyze_data_to_df(data_dict)

    print_with_tabulate(df_hw, title="Hardware Summary", floatfmt=args.floatfmt)
    print_with_tabulate(df_args, title="Run Arguments (Initial Benchmark Configuration)", floatfmt=args.floatfmt)
    print_with_tabulate(df_summary, title="Benchmark Summary (per Program/Config)", floatfmt=args.floatfmt)
    print("* Speedup* is based on the median time at the lowest available OMP_NUM_THREADS "
          "for each program (matching that program’s baseline config). No cross-program speedup.\n")

    # Print LaTeX table, used for the final report
    print_as_latex_table = True
    if print_as_latex_table:
        print("\n" + "="*80)
        print("Raw LaTeX Table Source Code for Benchmark Summary:")
        print("="*80)
        print_latex_summary_table(df_summary, floatfmt=".2f")
        print("="*80 + "\n")

    # SQLite output
    db_path = Path(args.db_path)
    prefix = log_file_path.with_suffix('').name  # infer from input filename
    save_to_sql(db_path, prefix, df_hw, df_args, df_summary)

if __name__ == "__main__":
    main()
