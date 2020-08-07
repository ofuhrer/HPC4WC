# ******************************************************
# Functions for Performance Report generation
# ******************************************************


import pandas as pd
import os.path


def add_data(
    df_name,
    stencil_name,
    backend,
    numba_parallel,
    numba_cudadevice,
    gt4py_backend,
    nx,
    ny,
    nz,
    num_iter,
    elapsedtime,
    run_avg,
    run_stdev,
    run_first10,
    run_last10,
):
    """
    Appends a row with several variables to the DataFrame performance report.

    Parameters
    ----------
    report_name  : Name of File on disk
    stencil_type : Stencil Type from stencil list
    backend  : backend from the backend list
    nx : field size in x-Direction.
    ny : field size in y-Direction.
    nz : field size in z-Direction.
    elapsedtime : measured work time
    valid_var : Boolean if Validation was successful

    Returns
    -------
    None.

    """
    if os.path.exists("eval/{}_result.pkl".format(df_name)) == False:
        df = pd.DataFrame(
            data=None,
            columns=[
                "stencil_name",
                "backend",
                "numba_parallel",
                "numba_cudadevice",
                "gt4py_backend",
                "nx",
                "ny",
                "nz",
                "num_iter",
                "time_total",
                "run_avg",
                "run_stdev",
                "run_first10",
                "run_last10",
            ],
        )
        print("New dataframe {} generated.".format(df_name))

    else:
        df = pd.read_pickle("eval/{}_result.pkl".format(df_name))

    # Add data
    df = df.append(
        {
            "stencil_name": stencil_name,
            "backend": backend,
            "numba_parallel": numba_parallel,
            "numba_cudadevice": numba_cudadevice,
            "gt4py_backend": gt4py_backend,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "time_total": elapsedtime,
            "run_avg": run_avg,
            "run_stdev": run_stdev,
            "run_first10": run_first10,
            "run_last10": run_last10,
            "num_iter": num_iter,
        },
        ignore_index=True,
        sort=False,
    )
    df.to_pickle("eval/{}_result.pkl".format(df_name))


def save_runtime_as_df(time_list):
    """
    

    Parameters
    ----------
    time_list : Save the indidual runtime into a dataframe.

    Returns
    -------
    None.

    """

    df = pd.DataFrame(time_list)
    df.to_pickle("eval/runtimedevelopment.pkl")

def add_data_bandwidth(
    df_name,
        backend,
        nx,
        ny,
        nz,
        num_iter,
        time_total,
        time_avg,
        time_stdev,
        number_of_gbytes,
        memory_bandwidth_in_gbs,
        memory_bandwidth_stdev,
        peak_bandwidth_in_gbs,
        fraction_of_peak_bandwidth,
        fraction_of_peak_bandwidth_stdev
    ):
    """
    Appends a row with several variables to the DataFrame performance report.

    Parameters
    ----------
    report_name  : Name of File on disk
    stencil_type : Stencil Type from stencil list
    backend  : backend from the backend list
    nx : field size in x-Direction.
    ny : field size in y-Direction.
    nz : field size in z-Direction.
    elapsedtime : measured work time
    valid_var : Boolean if Validation was successful

    Returns
    -------
    None.

    """
    if os.path.exists("eval/{}_result.pkl".format(df_name)) == False:
        df = pd.DataFrame(
            data=None,
            columns=[
                "backend",
                "nx",
                "ny",
                "nz",
                "num_iter",
                "time_total",
                "time_avg",
                "time_stdev",
                "number_of_gbytes",
                "memory_bandwidth_in_gbs",
                "memory_bandwidth_stdev",
                "peak_bandwidth_in_gbs",
                "fraction_of_peak_bandwidth",
                "fraction_of_peak_bandwidth_stdev"
            ],
        )
        print("New dataframe {} generated.".format(df_name))

    else:
        df = pd.read_pickle("eval/{}_result.pkl".format(df_name))

    # Add data
    df = df.append(
        {
            "backend":backend,
                "nx":nx,
                "ny":ny,
                "nz":nz,
                "num_iter":num_iter,
                "time_total":time_total,
                "time_avg": time_avg,
                "time_stdev":time_stdev,
                "number_of_gbytes":number_of_gbytes,
                "memory_bandwidth_in_gbs":memory_bandwidth_in_gbs,
                "memory_bandwidth_stdev":memory_bandwidth_stdev,
                "peak_bandwidth_in_gbs":peak_bandwidth_in_gbs,
                "fraction_of_peak_bandwidth":fraction_of_peak_bandwidth,
                "fraction_of_peak_bandwidth_stdev":fraction_of_peak_bandwidth_stdev
        },
        ignore_index=True,
        sort=False,
    )
    df.to_pickle("eval/{}_result.pkl".format(df_name))