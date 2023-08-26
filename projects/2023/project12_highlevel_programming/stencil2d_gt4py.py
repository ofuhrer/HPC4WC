# Modified module from:
#
# ******************************************************
#     Program: stencil2d-gt4py
#      Author: Stefano Ubbiali
#       Email: subbiali@phys.ethz.ch
#        Date: 04.06.2020
# Description: GT4Py implementation of 4th-order diffusion
# ******************************************************
import gt4py as gt
from gt4py.cartesian import gtscript
import numpy as np


@gtscript.function
def laplacian(in_field):
    lap_field = (
        -4.0 * in_field[0, 0, 0]
        + in_field[-1, 0, 0]
        + in_field[1, 0, 0]
        + in_field[0, -1, 0]
        + in_field[0, 1, 0]
    )
    return lap_field


def diffusion_defs(
    in_field: gtscript.Field["dtype"],
    out_field: gtscript.Field["dtype"],
    *,
    alpha: float,
):

    with computation(PARALLEL), interval(...):
        lap1 = laplacian(in_field)
        lap2 = laplacian(lap1)
        out_field = in_field - alpha * lap2


def copy_defs(
    src: gtscript.Field["dtype"],
    dst: gtscript.Field["dtype"]
):

    with computation(PARALLEL), interval(...):
        dst = src


def update_halo(copy_stencil, field, num_halo):
    nx = field.shape[0] - 2 * num_halo
    ny = field.shape[1] - 2 * num_halo
    nz = field.shape[2]

    # bottom edge (without corners)
    copy_stencil(
        src=field,
        dst=field,
        origin={"src": (num_halo, ny, 0), "dst": (num_halo, 0, 0)},
        domain=(nx, num_halo, nz),
    )

    # top edge (without corners)
    copy_stencil(
        src=field,
        dst=field,
        origin={"src": (num_halo, num_halo, 0), "dst": (num_halo, ny + num_halo, 0)},
        domain=(nx, num_halo, nz),
    )

    # left edge (including corners)
    copy_stencil(
        src=field,
        dst=field,
        origin={"src": (nx, 0, 0), "dst": (0, 0, 0)},
        domain=(num_halo, ny + 2 * num_halo, nz),
    )

    # right edge (including corners)
    copy_stencil(
        src=field,
        dst=field,
        origin={"src": (num_halo, 0, 0), "dst": (nx + num_halo, 0, 0)},
        domain=(num_halo, ny + 2 * num_halo, nz),
    )


def apply_diffusion(
    diffusion_stencil, copy_stencil, in_field, out_field, alpha, num_halo, num_iter=1
):
    # origin and extent of the computational domain
    origin = (num_halo, num_halo, 0)
    domain = (
        in_field.shape[0] - 2 * num_halo,
        in_field.shape[1] - 2 * num_halo,
        in_field.shape[2],
    )

    for n in range(num_iter):
        # halo update
        update_halo(copy_stencil, in_field, num_halo)

        # run the stencil
        diffusion_stencil(
            in_field=in_field,
            out_field=out_field,
            alpha=alpha,
            origin=origin,
            domain=domain,
        )

        if n < num_iter - 1:
            # swap input and output fields
            in_field, out_field = out_field, in_field
        else:
            # halo update
            update_halo(copy_stencil, out_field, num_halo)