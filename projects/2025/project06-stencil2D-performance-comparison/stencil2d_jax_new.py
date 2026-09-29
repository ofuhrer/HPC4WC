import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

# ---------------------------------------
# 1) Laplacian (halo-aware slicing)
# ---------------------------------------
@partial(jax.jit, static_argnames=("num_halo", "extend"))
def laplacian(in_field, num_halo: int, extend: int = 0):
    """
    Compute the 2D (x,y) 5-point Laplacian per z-slice.
    Returns a new array; no in-place mutation (JAX is functional).
    """
    h = num_halo
    ib = h - extend
    ie = -h + extend
    jb = h - extend
    je = -h + extend

    x_right_end = (ie + 1) if (ie != -1) else None
    y_up_end    = (je + 1) if (je != -1) else None

    lap = jnp.zeros_like(in_field)

    center = in_field[:, jb:je, ib:ie]
    left   = in_field[:, jb:je, ib - 1 : ie - 1]
    right  = in_field[:, jb:je, ib + 1 : x_right_end]
    down   = in_field[:, jb - 1 : je - 1, ib:ie]
    up     = in_field[:, jb + 1 : y_up_end,    ib:ie]

    interior_lap = -4.0 * center + left + right + down + up
    return lap.at[:, jb:je, ib:ie].set(interior_lap)


# ---------------------------------------
# 2) Halo update (up/down then left/right)
# ---------------------------------------
@partial(jax.jit, static_argnames=("num_halo",))
def update_halo(field, num_halo: int):
    """Periodic halo copy. Returns a new array."""
    h = num_halo
    if h == 0:
        return field
    out = field
    out = out.at[:, :h, h:-h].set(out[:, -2*h:-h, h:-h])  # bottom (no corners)
    out = out.at[:, -h:, h:-h].set(out[:, h:2*h, h:-h])   # top (no corners)
    out = out.at[:, :, :h].set(out[:, :, -2*h:-h])        # left  (incl corners)
    out = out.at[:, :, -h:].set(out[:, :, h:2*h])         # right (incl corners)
    return out


# ---------------------------------------
# 3) 4th-order diffusion apply (iterative)
# ---------------------------------------
@partial(jax.jit, static_argnames=("num_halo", "num_iter"))
def apply_diffusion(in_field, alpha: jnp.float32, num_halo: int, num_iter: int = 1):
    """
    Perform num_iter steps:
      tmp = Lap(in, extend=1); lap2 = Lap(tmp, extend=0);
      out[interior] = in[interior] - alpha * lap2[interior].
    Pointer-swap semantics via carry; final halo update on the result.
    Returns the final field.
    """
    h = num_halo

    def body(i, carry):
        inf, outf = carry
        inf = update_halo(inf, h)
        tmp  = laplacian(inf, num_halo=h, extend=1)
        lap2 = laplacian(tmp, num_halo=h, extend=0)

        interior = inf[:, h:-h, h:-h] - alpha * lap2[:, h:-h, h:-h]
        outf = outf.at[:, h:-h, h:-h].set(interior)

        # swap unless last iteration
        inf, outf = lax.cond(i < num_iter - 1,
                             lambda c: (c[1], c[0]),
                             lambda c: c,
                             (inf, outf))
        return (inf, outf)

    # start with explicit buffers (avoid shape changes inside loop)
    outf0 = jnp.zeros_like(in_field)
    inf, outf = lax.fori_loop(0, num_iter, body, (in_field, outf0))
    return update_halo(outf, h)


@click.command()
@click.option("--nx", type=int, required=True)
@click.option("--ny", type=int, required=True)
@click.option("--nz", type=int, required=True)
@click.option("--num_iter", type=int, required=True)
@click.option("--num_halo", type=int, default=2)
@click.option("--device", type=click.Choice(["cpu", "gpu"]), default="cpu",
              help="Select JAX device (gpu requires CUDA build).")
@click.option("--plot_result", is_flag=True, default=False,
              help="Save PNGs of mid-slice.")
def main(nx, ny, nz, num_iter, num_halo=2, device="cpu", plot_result=False):
    # sanity checks
    assert 0 < nx <= 1024 * 1024
    assert 0 < ny <= 1024 * 1024
    assert 0 < nz <= 1024
    assert 0 < num_iter <= 1024 * 1024
    assert 2 <= num_halo <= 256

    # device selection
    if device == "gpu":
        gpus = jax.devices("gpu")
        if not gpus:
            raise RuntimeError("GPU requested but no JAX GPU device is available.")
        dev = gpus[0]
    else:
        dev = jax.devices("cpu")[0]

    alpha = jnp.float32(1.0 / 32.0)

    # allocate on host then place on chosen device
    shape = (nz, ny + 2 * num_halo, nx + 2 * num_halo)
    in_field = np.zeros(shape, dtype=np.float32)
    in_field[
        nz // 4 : 3 * nz // 4,
        num_halo + ny // 4 : num_halo + 3 * ny // 4,
        num_halo + nx // 4 : num_halo + 3 * nx // 4,
    ] = 1.0
    in_field = jax.device_put(jnp.asarray(in_field), device=dev)

    # optional input plot (middle z-slice)
    if plot_result:
        plt.ioff()
        mid = in_field.shape[0] // 2
        plt.imshow(np.asarray(in_field[mid, :, :]), origin="lower")
        plt.colorbar()
        plt.savefig("in_field_jax.png")
        plt.close()

    # warmup (compile) and sync
    _ = apply_diffusion(in_field, alpha, num_halo, num_iter=1).block_until_ready()

    # timed run (sync after)
    tic = time.time()
    out_field = apply_diffusion(in_field, alpha, num_halo, num_iter=num_iter).block_until_ready()
    toc = time.time()
    print(f"Elapsed time for work = {toc - tic:.6f} s")

    # save results (host side)
    np.save("in_field_jax", np.asarray(in_field, dtype=np.float32))
    np.save("out_field_jax", np.asarray(out_field, dtype=np.float32))

    if plot_result:
        mid = out_field.shape[0] // 2
        plt.imshow(np.asarray(out_field[mid, :, :]), origin="lower")
        plt.colorbar()
        plt.savefig("out_field_jax.png")
        plt.close()


if __name__ == "__main__":
    main()
