import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time
import cupy as cp


def laplacian(in_field, lap_field, num_halo, extend=0):
    """Compute the Laplacian using 2nd-order centered differences.

    Parameters
    ----------
    in_field : array-like
        Input field (nz x ny x nx with halo in x- and y-direction).
    lap_field : array-like
        Result (must be same size as ``in_field``).
    num_halo : int
        Number of halo points.
    extend : `int`, optional
        Extend computation into halo-zone by this number of points.
    """
    ib = num_halo - extend
    ie = -num_halo + extend
    jb = num_halo - extend
    je = -num_halo + extend

    lap_field[:, jb:je, ib:ie] = (
        -4.0 * in_field[:, jb:je, ib:ie]
        + in_field[:, jb:je, ib - 1 : ie - 1]
        + in_field[:, jb:je, ib + 1 : ie + 1 if ie != -1 else None]
        + in_field[:, jb - 1 : je - 1, ib:ie]
        + in_field[:, jb + 1 : je + 1 if je != -1 else None, ib:ie]
    )


def halo_update(field, num_halo):
    """Update the halo-zone using an up/down and left/right strategy.

    Parameters
    ----------
    field : array-like
        Input/output field (nz x ny x nx with halo in x- and y-direction).
    num_halo : int
        Number of halo points.

    Note
    ----
        Corners are updated in the left/right phase of the halo-update.
    """
    # bottom edge (without corners)
    field[:, :num_halo, num_halo:-num_halo] = field[
        :, -2 * num_halo : -num_halo, num_halo:-num_halo
    ]

    # top edge (without corners)
    field[:, -num_halo:, num_halo:-num_halo] = field[
        :, num_halo : 2 * num_halo, num_halo:-num_halo
    ]

    # left edge (including corners)
    field[:, :, :num_halo] = field[:, :, -2 * num_halo : -num_halo]

    # right edge (including corners)
    field[:, :, -num_halo:] = field[:, :, num_halo : 2 * num_halo]


def apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=1):
    """Integrate 4th-order diffusion equation by a certain number of iterations.

    Parameters
    ----------
    in_field : array-like
        Input field (nz x ny x nx with halo in x- and y-direction).
    lap_field : array-like
        Result (must be same size as ``in_field``).
    alpha : float
        Diffusion coefficient (dimensionless).
    num_iter : `int`, optional
        Number of iterations to execute.
    """
    tmp_field = cp.empty_like(in_field)
    stream = cp.cuda.Stream(non_blocking=True)

    # Capture a CUDA graph for the first diffusion iteration.
    with stream:
        stream.begin_capture()

        halo_update(in_field, num_halo)
        laplacian(in_field, tmp_field, num_halo=num_halo, extend=1)
        laplacian(tmp_field, out_field, num_halo=num_halo, extend=0)

        out_field[:, num_halo:-num_halo, num_halo:-num_halo] = (
            in_field[:, num_halo:-num_halo, num_halo:-num_halo]
            - alpha * out_field[:, num_halo:-num_halo, num_halo:-num_halo]
        )

        # End capturing the operations and create the first CUDA graph.
        graph_even = stream.end_capture()

    # Swap the input and output fields.
    in_field, out_field = out_field, in_field

    # Capture a CUDA graph for the second diffusion iteration (swapped fields).
    with stream:
        stream.begin_capture()

        halo_update(in_field, num_halo)
        laplacian(in_field, tmp_field, num_halo=num_halo, extend=1)
        laplacian(tmp_field, out_field, num_halo=num_halo, extend=0)

        out_field[:, num_halo:-num_halo, num_halo:-num_halo] = (
            in_field[:, num_halo:-num_halo, num_halo:-num_halo]
            - alpha * out_field[:, num_halo:-num_halo, num_halo:-num_halo]
        )

        # End capturing the operations and create the second CUDA graph.
        graph_odd = stream.end_capture()

    # Swap the input and output fields back.
    in_field, out_field = out_field, in_field

    # Launch the graph for all iterations except the last one.
    for n in range(num_iter - 1):
        if n % 2 == 0:
            graph_even.launch(stream)
        else:
            graph_odd.launch(stream)

        stream.synchronize()
        # Swap the input and output fields for the next iteration.
        in_field, out_field = out_field, in_field

    stream.synchronize()

    # Perform the last iteration without the CUDA graph.
    halo_update(in_field, num_halo)
    laplacian(in_field, tmp_field, num_halo=num_halo, extend=1)
    laplacian(tmp_field, out_field, num_halo=num_halo, extend=0)

    out_field[:, num_halo:-num_halo, num_halo:-num_halo] = (
        in_field[:, num_halo:-num_halo, num_halo:-num_halo]
        - alpha * out_field[:, num_halo:-num_halo, num_halo:-num_halo]
    )

    halo_update(out_field, num_halo)


def plot_field(field, filename):
    """Plot a 2D field and save it to a file."""
    plt.ioff()
    plt.imshow(field[field.shape[0] // 2, :, :], origin="lower")
    plt.colorbar()
    plt.savefig(filename)
    plt.close()


def main(
    nx,
    ny,
    nz,
    num_iter,
    num_halo=2,
    plot_result=False,
    save_result=True,
    return_result=False,
    benchmark=False,
):
    """Driver for apply_diffusion that sets up fields and does timings"""

    assert 0 < nx <= 1024 * 1024, "You have to specify a reasonable value for nx"
    assert 0 < ny <= 1024 * 1024, "You have to specify a reasonable value for ny"
    assert 0 < nz <= 1024, "You have to specify a reasonable value for nz"
    assert (
        -1 < num_iter <= 1024 * 1024
    ), "You have to specify a reasonable value for num_iter"
    assert (
        2 <= num_halo <= 256
    ), "Your have to specify a reasonable number of halo points"
    alpha = 1.0 / 32.0

    in_field = cp.zeros((nz, ny + 2 * num_halo, nx + 2 * num_halo))
    in_field[
        nz // 4 : 3 * nz // 4,
        num_halo + ny // 4 : num_halo + 3 * ny // 4,
        num_halo + nx // 4 : num_halo + 3 * nx // 4,
    ] = 1.0

    out_field = cp.copy(in_field)

    if save_result:
        np.save("in_field", in_field.get())

    if plot_result:
        plot_field(in_field.get(), "in_field.png")

    # warmup caches
    # print("warmup")
    apply_diffusion(in_field, out_field, alpha, num_halo)

    # time the actual work
    # print("starting actual work")
    tic = time.time()
    cp.cuda.Stream.null.synchronize()
    apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=num_iter)
    cp.cuda.Stream.null.synchronize()
    toc = time.time()

    # print(f"Elapsed time for work = {toc - tic} s")

    if save_result:
        np.save("out_field", out_field.get())

    if plot_result:
        plot_field(out_field.get(), "out_field.png")

    if return_result:
        assert not benchmark
        return out_field.get()

    if benchmark:
        return toc - tic
