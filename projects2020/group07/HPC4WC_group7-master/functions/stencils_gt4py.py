import numpy as np
import gt4py
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage

backend = "numpy"
dtype = np.float64

# def gt4p_stencil(in_field):


def test_gt4py(
    in_field: gtscript.Field[dtype], out_field: gtscript.Field[dtype],
):
    with computation(PARALLEL), interval(...):

        out_field = in_field[0, 0, 0] - coeff[0, 0, 0] * (
            in_field[-1, -1, -1] - in_field[1, 1, 1] + in_field[0, 0, 0]
        )


def laplacian1d(
    in_field: gtscript.Field[dtype], out_field: gtscript.Field[dtype],
):
    with computation(PARALLEL), interval(...):

        out_field[0, 0, 0] = (
            -2.0 * in_field[0, 0, 0] + in_field[-1, 0, 0] + in_field[1, 0, 0]
        )


def laplacian2d(
    in_field: gtscript.Field[dtype], out_field: gtscript.Field[dtype],
):
    with computation(PARALLEL), interval(...):

        out_field[0, 0, 0] = (
            -4.0 * in_field[0, 0, 0]
            + in_field[-1, 0, 0]
            + in_field[1, 0, 0]
            + in_field[0, -1, 0]
            + in_field[0, 1, 0]
        )


def laplacian3d(
    in_field: gtscript.Field[dtype], out_field: gtscript.Field[dtype],
):
    with computation(FORWARD), interval(0,1):
        out_field = 1
    with computation(PARALLEL), interval(1,-1):
        out_field[0, 0, 0] = (
            -6.0 * in_field[0, 0, 0]
            + in_field[-1, 0, 0]
            + in_field[1, 0, 0]
            + in_field[0, -1, 0]
            + in_field[0, 1, 0]
            + in_field[0, 0, -1]
            + in_field[0, 0, 1]
        )
    with computation(PARALLEL), interval(-1,None):
        out_field = 1

def FMA(
    in_field: gtscript.Field[dtype],
    in_field2: gtscript.Field[dtype],
    in_field3: gtscript.Field[dtype],
    out_field: gtscript.Field[dtype],
):
    with computation(PARALLEL), interval(...):

        out_field[0, 0, 0] = in_field[0, 0, 0] + in_field2[0, 0, 0] * in_field3[0, 0, 0]


def lapoflap1d(
    in_field: gtscript.Field[dtype],
    out_field: gtscript.Field[dtype],
):
    with computation(PARALLEL), interval(...):
        tmp_field= (
            -2.0 * in_field[0, 0, 0] + in_field[-1, 0, 0] + in_field[1, 0, 0]
        )

        out_field[0, 0, 0] = (
            -2.0 * tmp_field[0, 0, 0] + tmp_field[-1, 0, 0] + tmp_field[1, 0, 0]
        )


def lapoflap2d(
    in_field: gtscript.Field[dtype],
    out_field: gtscript.Field[dtype],
):
    with computation(PARALLEL), interval(...):

        tmp_field = (
            -4.0 * in_field[0, 0, 0]
            + in_field[-1, 0, 0]
            + in_field[1, 0, 0]
            + in_field[0, -1, 0]
            + in_field[0, 1, 0]
        )

        out_field[0, 0, 0] = (
            -4.0 * tmp_field[0, 0, 0]
            + tmp_field[-1, 0, 0]
            + tmp_field[1, 0, 0]
            + tmp_field[0, -1, 0]
            + tmp_field[0, 1, 0]
        )


def lapoflap3d(
    in_field: gtscript.Field[dtype],
    tmp_field: gtscript.Field[dtype],
    out_field: gtscript.Field[dtype],
):
    with computation(FORWARD), interval(0,1):
        tmp_field = 1
    with computation(PARALLEL), interval(-1,None):
        tmp_field = 1
        
    with computation(PARALLEL), interval(1,-1):
        tmp_field = (
            -6.0 * in_field[0, 0, 0]
            + in_field[-1, 0, 0]
            + in_field[1, 0, 0]
            + in_field[0, -1, 0]
            + in_field[0, 1, 0]
            + in_field[0, 0, -1]
            + in_field[0, 0, 1]
        )

        
    with computation(FORWARD), interval(0,2):
        out_field = 1
    with computation(PARALLEL), interval(-2,None):
        out_field = 1
        
    with computation(PARALLEL), interval(2,-2):
        out_field = (
            -6.0 * tmp_field[0, 0, 0]
            + tmp_field[-1, 0, 0]
            + tmp_field[1, 0, 0]
            + tmp_field[0, -1, 0]
            + tmp_field[0, 1, 0]
            + tmp_field[0, 0, -1]
            + tmp_field[0, 0, 1]
        )

        
