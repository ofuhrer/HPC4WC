# ******************************************************
#     Program: stencil2d_jax
#      Author: Stefano Ubbiali, Oliver Fuhrer --> adapted to JAX by Nora Joss
#       Email: subbiali@phys.ethz.ch, ofuhrer@ethz.ch, nojoss@student.ethz.ch
#        Date: 04.06.2020 / 07-08.2026
# Description: JAX implementation of 4th-order diffusion
# ******************************************************
import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time
import jax.numpy as jnp
from jax import jit
import jax
from jax import lax


def laplacian(in_field, num_halo, extend=0):
    """Compute the Laplacian using 2nd-order centered differences.
        Arrays are immutable in JAX!!!

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


    lap = (
        -4.0 * in_field[:, jb:je, ib:ie]
        + in_field[:, jb:je, ib - 1 : ie - 1]
        + in_field[:, jb:je, ib + 1 : ie + 1 if ie != -1 else None]
        + in_field[:, jb - 1 : je - 1, ib:ie]
        + in_field[:, jb + 1 : je + 1 if je != -1 else None, ib:ie]
    )

    result = jnp.zeros_like(in_field)
    return result.at[:, jb:je, ib:ie].set(lap)


def update_halo(field, num_halo):
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
    field = field.at[:, :num_halo, num_halo:-num_halo].set(field[:, -2*num_halo:-num_halo, num_halo:-num_halo])

    # top edge (without corners)
    field = field.at[:, -num_halo:, num_halo:-num_halo].set(field[:, num_halo : 2 * num_halo, num_halo:-num_halo])

    # left edge (including corners)
    field = field.at[:, :, :num_halo].set(field[:, :, -2 * num_halo : -num_halo])

    # right edge (including corners)
    field = field.at[:, :, -num_halo:].set(field[:, :, num_halo : 2 * num_halo])

    return field

@jit(static_argnames=('num_halo','num_iter')) #static_argnames: tell jax that num_halo and num_iter are constants, to avoid unknown tracer problems in slicing
def apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=1):
    """Integrate 4th-order diffusion equation by a certain number of iterations.

    Parameters
    ----------
    in_field : array-like
        Input field (nz x ny x nx with halo in x- and y-direction).
    out_field : array-like
        Result (must be same size as ``in_field``).
    alpha : float
        Diffusion coefficient (dimensionless).
    num_iter : `int`, optional
        Number of iterations to execute.
    """

    def iteration_body(n, fields):
        in_field, out_field = fields
        in_field = update_halo(in_field, num_halo)
        
        tmp_field = laplacian(in_field, num_halo=num_halo, extend=1)
        out_field = laplacian(tmp_field, num_halo=num_halo, extend=0)
        
        out_field = out_field.at[:, num_halo:-num_halo, num_halo:-num_halo].set(
            in_field[:, num_halo:-num_halo, num_halo:-num_halo]
            - alpha * out_field[:, num_halo:-num_halo, num_halo:-num_halo]
            )

        return (out_field, in_field)

    field1, field2 = lax.fori_loop(0, num_iter, iteration_body, (in_field, out_field)) #fori_loop makes it possible to compile the iteration loop in a much more efficient way by the xla 

    if num_iter % 2 == 0:
        result = field2
    else: 
        result = field1

    result = update_halo(result, num_halo)
    return result

@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True
    )
)
@click.option("--in_field_path", type=str, required=True, help="location of numpy input grid file")
@click.option("--out_field_path", type=str, required=True, help="location of numpy output grid file")
@click.option("--num_iter", type=int, required=True, help="Number of iterations")
@click.option(
    "--num_halo",
    type=int,
    default=2,
    help="Number of halo points in x- and y-direction",
)
@click.option("--device", type=click.Choice(["cpu", "gpu"]), default="cpu", help="set cpu for CPU or gpu for GPU")

def main(in_field_path, out_field_path, num_iter, num_halo=2, device="cpu"):
    """Driver for apply_diffusion that sets up fields and does timings"""

    assert 0 < num_iter <= 1024 * 1024, "You have to specify a reasonable value for num_iter"
    assert 2 <= num_halo <= 256, "You have to specify a reasonable number of halo points"
    alpha = 1.0 / 64.0

    in_field = np.load(in_field_path)

    out_field = np.copy(in_field)

    if device is not None: 
        target_device = jax.devices(device)[0]
        in_field = jax.device_put(in_field, target_device)
        out_field = jax.device_put(out_field, target_device)


    # warmup caches
    _ = apply_diffusion(in_field, out_field, alpha, num_halo).block_until_ready()

    # time the actual work
    tic = time.time()
    out_field = apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=num_iter).block_until_ready()
    toc = time.time()

    #print(f"Elapsed time for work = {toc - tic} s")
    print(toc - tic)

    np.save(out_field_path, out_field)



if __name__ == "__main__":
    main()
