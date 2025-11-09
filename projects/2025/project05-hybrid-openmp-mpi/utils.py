import re
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm 

def read_field_from_file(filename, num_halo=None):
    (rank, nbits, num_halo, nx, ny, nz) = np.fromfile(filename, dtype=np.int32, count=6)
    offset = (3 + rank) * 32 // nbits
    data = np.fromfile(
        filename,
        dtype=np.float32 if nbits == 32 else np.float64,
        count=nz * ny * nx + offset,
    )
    if rank == 3:
        return np.reshape(data[offset:], (nz, ny, nx))
    else:
        return np.reshape(data[offset:], (ny, nx))


def validate_results():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    in_field = read_field_from_file("in_field.dat")
    im1 = axs[0].imshow(
        in_field[in_field.shape[0] // 2, :, :], origin="lower", vmin=-0.1, vmax=1.1
    )
    fig.colorbar(im1, ax=axs[0])
    axs[0].set_title("Initial condition")

    out_field = read_field_from_file("out_field.dat")
    im2 = axs[1].imshow(
        out_field[out_field.shape[0] // 2, :, :], origin="lower", vmin=-0.1, vmax=1.1
    )
    fig.colorbar(im2, ax=axs[1])
    axs[1].set_title("Final result")

    plt.show()

def bar_plot_omp_runtime(times_masked, mpi_rank):
    # OMP Threads, MPI Rank const
    # times_masked: 2d runtime of OMP-MPI combinations (#OMP, #MPI)

    labels = np.arange(len(times_masked[:, mpi_rank]))
    x = labels  # the label locations
    width = 0.35  # the width of the bars
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(labels, times_masked[:, mpi_rank])
    ax.set_xlabel('Number of OMP Threads')
    ax.set_ylabel('Runtime (s)')
    ax.set_title(f"OMP Threads vs Runtimes, {mpi_rank} MPI Rank")

    return fig

def bar_plot_mpi_runtime(times_masked, omp_threads):
    # MPI Ranks, OMP Threads const
    # times_masked: 2d runtime of OMP-MPI combinations (#OMP, #MPI)
    
    labels = np.arange(len(times_masked[omp_threads, :]))
    x = labels  # the label locations
    width = 0.35  # the width of the bars
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    # plt.plot(runtimes[:][1])
    ax.bar(labels, times_masked[omp_threads, :])
    ax.set_xlabel('Number of MPI Ranks')
    ax.set_ylabel('Runtime (s)')
    ax.set_title(f"MPI Ranks vs Runtimes, {omp_threads} OMP Thread")
    return fig


def plot_speedup_omp_threads(times_masked, nx_len, ny_len, mpi_ranks='all'):
    T_baseline=times_masked[1][1]
    
    nthreads, nranks = times_masked.shape
    threads = np.arange(nthreads)
    if mpi_ranks == 'all':
        ranks = np.arange(nranks)  # column indices = MPI ranks
    else:
        ranks = mpi_ranks
    
    cmap = cm.bamako_r
    colors = [cmap(i / (len(ranks) - 1)) for i in range(len(ranks))]
    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'h', '<', '>', '8', 'p']
    
    fig = plt.figure(figsize=(10, 6))
    
    for idx, (j, color) in enumerate(zip(ranks, colors)):
        runtimes_j = T_baseline/(times_masked[:, j])
        # runtimes_j=(runtimes_array[:, j])
        if np.any(runtimes_j > 0):
            marker = markers[j % len(markers)]  # Cycle through markers
            plt.plot(threads, runtimes_j, label=f"{j} MPI ranks", color=color, marker=marker)
    
    plt.xlabel("OMP Threads")
    plt.ylabel("Speedup")
    # plt.xscale("log",  base=2)
    # plt.yscale("log",  base=2)
    plt.title(f"Speedup vs OMP Threads for Different MPI Ranks [{nx_len}, {ny_len}]")
    plt.legend(title="MPI Ranks", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.xlim(1,74)
    plt.tight_layout()
    return fig

def plot_heatmap(times_masked, nx_len, ny_len, mpi_ranks="all", omp_threads='all'):
    T_baseline=times_masked[1][1]
    labels = []
    times = []
    speedups= []

    nthreads, nranks = times_masked.shape
    if mpi_ranks == "all":
        nranks = range(nranks)
    else:
        nranks = mpi_ranks

    if omp_threads == "all":
        nthreads = range(nthreads)
    else:
        nthreads = omp_threads

    for thread in nthreads:
        for rank in nranks:
            runtime = times_masked[thread, rank]
            if runtime > 0.0:  # only plot entries that were filled
                labels.append([thread, rank])
                times.append(runtime)
                speedups.append(T_baseline/runtime)

    labels_array = np.array(labels)
    # threads = sorted(set(labels_array[:, 0]))
    # ranks = sorted(set(labels_array[:, 1]))

    threads = nthreads
    ranks = nranks
    
    heatmap = np.full((len(threads), len(ranks)), np.nan)
    
    for (t, r), val in zip(labels, speedups):
        i = threads.index(t)
        j = ranks.index(r)
        heatmap[i, j] = val
    heatmap=heatmap.T
    
    fig, ax = plt.subplots(figsize=(10, 6))
    c = ax.imshow(heatmap, cmap=cm.bamako, origin='lower', 
                  vmin=1,
                  aspect='auto')
    
    ax.set_ylabel('MPI Ranks')
    ax.set_xlabel('OMP Threads')
    ax.set_title(f'Speedup for domain size [{nx_len}, {ny_len}]')
    fig.colorbar(c, label='Speedup', shrink=0.8) 
    ax.set_yticks(np.arange(len(ranks)))
    ax.set_xticks(np.arange(len(threads)))
    ax.set_xticklabels(nthreads)
    ax.set_yticklabels(ranks)
    plt.tight_layout()
    return fig


############### Load Data  ####################################

def load_runtimes(path):
    pat = re.compile(r"runtimes\[(\d+)\]\[(\d+)\]\[(\d+)\]\s*=\s*([0-9.Ee+\-]+)")
    entries = []
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            i, j, k, val = m.groups()
            entries.append((int(i), int(j), int(k), float(val)))

    if not entries:
        raise ValueError("No runtimes[...] entries found.")

    imax = max(i for i,_,_,_ in entries)
    jmax = max(j for _,j,_,_ in entries)
    kmax = max(k for _,_,k,_ in entries)

    arr = np.zeros((imax + 1, jmax + 1, kmax + 1), dtype=float)
    
    for i, j, k, val in entries:
        arr[i, j, k] = val
    return arr

def get_runtimes(size, folder, method='mean'): #method = 'one'
    if size == 'small':
        ratio_1_1 = load_runtimes(f"{folder}/out_s_{128}_{128}.txt")
        ratio_2_1 = load_runtimes(f"{folder}/out_s_{176}_{88}.txt")
        ratio_1_2 = load_runtimes(f"{folder}/out_s_{88}_{176}.txt")
        ratio_4_1 = load_runtimes(f"{folder}/out_s_{256}_{64}.txt")
        ratio_1_4 = load_runtimes(f"{folder}/out_s_{64}_{256}.txt")
    elif size == 'medium':
        ratio_1_1 = load_runtimes(f"{folder}/out_m_{512}_{512}.txt")
        ratio_2_1 = load_runtimes(f"{folder}/out_m_{720}_{360}.txt")
        ratio_1_2 = load_runtimes(f"{folder}/out_m_{360}_{720}.txt")
        ratio_4_1 = load_runtimes(f"{folder}/out_m_{1024}_{256}.txt")
        ratio_1_4 = load_runtimes(f"{folder}/out_m_{256}_{1024}.txt")
    elif size == 'large':
        ratio_1_1 = load_runtimes(f"{folder}/out_l_{2048}_{2048}.txt")
        ratio_2_1 = load_runtimes(f"{folder}/out_l_{2896}_{1448}.txt")
        ratio_1_2 = load_runtimes(f"{folder}/out_l_{1448}_{2896}.txt")
        ratio_4_1 = load_runtimes(f"{folder}/out_l_{4096}_{1024}.txt")
        ratio_1_4 = load_runtimes(f"{folder}/out_l_{1024}_{4096}.txt")

    ratio_1_1_masked = np.where(ratio_1_1>0, ratio_1_1, np.nan)
    ratio_2_1_masked = np.where(ratio_2_1>0, ratio_2_1, np.nan)
    ratio_1_2_masked = np.where(ratio_1_2>0, ratio_1_2, np.nan)
    ratio_4_1_masked = np.where(ratio_4_1>0, ratio_4_1, np.nan)    
    ratio_1_4_masked = np.where(ratio_1_4>0, ratio_1_4, np.nan)

    if method == 'mean':
        ratio_1_1 = np.nanmean(ratio_1_1_masked, 0)
        ratio_2_1 = np.nanmean(ratio_2_1_masked, 0)
        ratio_1_2 = np.nanmean(ratio_1_2_masked, 0)
        ratio_4_1 = np.nanmean(ratio_4_1_masked, 0)
        ratio_1_4 = np.nanmean(ratio_1_4_masked, 0)
    else:
        ratio_1_1 = ratio_1_1_masked[1,:,:]
        ratio_2_1 = ratio_2_1_masked[1,:,:]
        ratio_1_2 = ratio_1_2_masked[1,:,:]
        ratio_4_1 = ratio_4_1_masked[1,:,:]
        ratio_1_4 = ratio_1_4_masked[1,:,:]

    return ratio_1_1, ratio_2_1, ratio_1_2, ratio_4_1, ratio_1_4
    
def get_results(folder, sizes=['small', 'medium', 'large'], 
                ratios = ["1:1","2:1","1:2","4:1","1:4"],method='mean'):

    results = {}

    for size in sizes:
        results[size] = {}
        for ratio in ratios:
            results[size][ratio] = {}
        r_11, r_21, r_12, r_41, r_14 = get_runtimes(size, folder, method)
        results[size]['1:1']['runtime'] = r_11
        results[size]['2:1']['runtime'] = r_21
        results[size]['1:2']['runtime'] = r_12
        results[size]['4:1']['runtime'] = r_41
        results[size]['1:4']['runtime'] = r_14

    results['small']['1:1']['dims'] = (128, 128)
    results['small']['2:1']['dims'] = (176, 88)
    results['small']['1:2']['dims'] = (88, 176)
    results['small']['4:1']['dims'] = (256, 64)
    results['small']['1:4']['dims'] = (64, 256)

    results['medium']['1:1']['dims'] = (512, 512)
    results['medium']['2:1']['dims'] = (720, 360)
    results['medium']['1:2']['dims'] = (360, 720)
    results['medium']['4:1']['dims'] = (1024, 256)
    results['medium']['1:4']['dims'] = (256, 1024)

    results['large']['1:1']['dims'] = (2048, 2048)
    results['large']['2:1']['dims'] = (2896, 1448)
    results['large']['1:2']['dims'] = (1448, 2896)
    results['large']['4:1']['dims'] = (4096, 1024)
    results['large']['1:4']['dims'] = (1024, 4096)        

    for size in sizes:
        for ratio in ratios:
            runtimes = results[size][ratio]['runtime']
            T_baseline = runtimes[1][1]
            results[size][ratio]['speedup'] = T_baseline/runtimes

    return results

###################################################


def get_best_speedups(results, sizes, ratios, threads, ranks):
    best_speedups = {}
    for size in sizes:
        best_speedups[size] = {}
        for ratio in ratios:
            best_speedups[size][ratio] = {}
            speedup = results[size][ratio]['speedup']

            # Get best speedups from the threads and ranks selected
            best = 0
            best_pair = 0
            for thread in threads:
                for rank in ranks:
                    v = speedup[thread, rank]
                    if np.isnan(v):
                        continue
                    if v > best:
                        best = v
                        best_pair = (thread, rank)
            best_speedups[size][ratio]['value'] = best
            best_speedups[size][ratio]['threads_ranks'] = best_pair
            
    return best_speedups


def grouped_shape_bars(data, sizes, ratios=("1:4","1:2","1:1","2:1","4:1"),
    metric_name="speedups", ylim_pad=0.10, bar_width=0.15,
    rotate_xticks=0, title=None, savepath=None):

    # build matrix values[size_idx, shape_idx]
    V = []
    for sz in sizes:
        row = []
        for sh in ratios:
            row.append(data[sz][sh]['value'])
        V.append(row)
    V = np.array(V, dtype=float)  # shape [S, H]
    S, H = V.shape

    x = np.arange(S)  # group centers
    total_width = H * bar_width
    start = -0.5 * total_width + 0.5 * bar_width

    fig, ax = plt.subplots(figsize=(max(6, 1.6*S), 3.8))
    bars = []
    for j, sh in enumerate(ratios):
        offs = start + j * bar_width
        bj = ax.bar(x + offs, V[:, j], width=bar_width, label=sh)
        bars.append(bj)

    ax.set_xticks(x)
    ax.set_xticklabels(sizes, rotation=rotate_xticks)
    ax.set_xlabel("Domain size (constant area)")
    ax.set_ylabel(metric_name)
    if title:
        ax.set_title(title)
    ax.legend(title="Aspect ratio (nx:ny)", ncols=min(H, 5))

    # y-limits with padding
    finite_vals = V[np.isfinite(V)]
    if finite_vals.size:
        ymin = max(0, finite_vals.min()*0.9)
        ymax = finite_vals.max() * (1 + ylim_pad)
        ax.set_ylim(ymin, ymax)

    ax.grid(axis="y", linewidth=0.5)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    return fig, ax

######### Strong Scaling ##################

def build_curves(results, sizes, ratios, threads_vals, ranks_vals, metric='speedup'):
    """
    For each (size,shape), produce best-per-total-cores curve.
    metric: "speedup" | "runtime"
    """
    curves = {}
    for size in sizes:
        curves[size] = {}
        for ratio in ratios:
            runtime = results[size][ratio]["runtime"]

            # Get best_vs_total_cores
            best = {}
            best_pair = {}
            for thread in threads_vals:
                for rank in ranks_vals:
                    v = runtime[thread, rank]
                    if np.isnan(v):
                        continue
                    C = int(thread) * int(rank)
                    if (C not in best) or (v < best[C]):
                        best[C] = v
                        best_pair[C] = (thread, rank)
            cores = np.array(sorted(best.keys()))
            y = np.array([best[c] for c in cores], dtype=float)
            pairs = [best_pair[c] for c in cores]

            if len(cores):
                if metric=='speedup':
                    y = y[0] / y
                curves[size][ratio] = {"cores": cores, "y": y, "pairs": pairs}
            
    return curves

def plot_by_size(curves, sizes, shapes, ylabel="Speedup", title=None, save=None):
    """
    One figure per size; lines = shapes.
    """
    figs = []
    for sz in sizes:
        fig, ax = plt.subplots()
        for sh in shapes:
            ax.plot(curves[sz][sh]["cores"], curves[sz][sh]["y"], marker=".", label=sh)
        ax.plot(curves[sz][sh]["cores"], curves[sz][sh]["cores"], linestyle='dashed', color='black', label='Ideal')
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("Total cores (ranks × threads)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Strong scaling — {sz}")
        ax.grid(True, which="both", linewidth=0.5)
        ax.legend(title="Dimensions ratio")
        fig.tight_layout()
        if save:
            fig.savefig(f"{save}_{sz}.png", dpi=200, bbox_inches="tight")
        figs.append(fig)
    return figs

def plot_ss_OMP(results, sizes, shapes, threads, rank=1, ylabel="Speedup", title=None, save=None):
    """
    One figure per size; lines = shapes.
    """
    figs = []
    for sz in sizes:
        fig, ax = plt.subplots()
        for sh in shapes:
            ax.plot(threads, results[sz][sh]["speedup"][threads, rank], marker=".", label=sh)
        ax.plot(threads, threads, linestyle='dashed', color='black', label='Ideal')
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("OMP threads")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Strong scaling OpenMP — {sz}")
        ax.grid(True, which="both", linewidth=0.5)
        ax.legend(title="Dimensions ratio")
        fig.tight_layout()
        if save:
            fig.savefig(f"{save}_{sz}.png", dpi=200, bbox_inches="tight")
        figs.append(fig)
    return figs

def plot_ss_MPI(results, sizes, shapes, thread, ranks, ylabel="Speedup", title=None, save=None):
    """
    One figure per size; lines = shapes.
    """
    figs = []
    for sz in sizes:
        fig, ax = plt.subplots()
        for sh in shapes:
            ax.plot(ranks, results[sz][sh]["speedup"][thread, ranks], marker=".", label=sh)
        ax.plot(ranks, ranks, linestyle='dashed', color='black', label='Ideal')
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Strong scaling MPI — {sz}")
        ax.grid(True, which="both", linewidth=0.5)
        ax.legend(title="Dimensions ratio")
        fig.tight_layout()
        if save:
            fig.savefig(f"{save}_{sz}.png", dpi=200, bbox_inches="tight")
        figs.append(fig)
    return figs



