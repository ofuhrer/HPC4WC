"""GT4Py implementation of the order-generic monotonic diffusion filter.

Implements the same filter as the Fortran variants, with the flux limiter
of Xue (2000) Eq. 10. Field operators admit no runtime loops, so a
template emits one GT4Py program per (order, limiter) pair. Output uses
the Fortran binary field format, so `tools/read_field.py` compares the two
implementations directly.

Usage:
    stencil2d-filter-gt4py.py --nx NX --ny NY --nz NZ --num_iter N
        [--order {2,4,6,8}] [--limiter] [--alpha A] [--backend {cpu,gpu}]

Based on the course day-5 GT4Py example.

Authors:
    Stefanie Boersig <stefanie.boersig@env.ethz.ch>
    Boaz Ko <boazko@student.ethz.ch>
    Ben Bullinger <ben.bullinger@inf.ethz.ch>
"""

import linecache
import sys
import time

sys.setrecursionlimit(200_000)  # gt4py's passes recurse deeply for n >= 6

import click  # noqa: E402
import numpy as np  # noqa: E402
import gt4py.next as gtx  # noqa: E402
from gt4py.next import where  # noqa: E402

backend_str_to_backend = {
    "none": None, "cpu": gtx.gtfn_cpu, "gpu": gtx.gtfn_gpu}

I = gtx.Dimension("I")  # noqa: E741
J = gtx.Dimension("J")
K = gtx.Dimension("K")

F32 = gtx.float32
IJKField = gtx.Field[gtx.Dims[I, J, K], F32]
OFFSET_PROVIDER = {"_IOff": I, "_JOff": J}


@gtx.field_operator
def laplacian(f: IJKField) -> IJKField:
    return F32(-4.0) * f + f(I - 1) + f(I + 1) + f(J - 1) + f(J + 1)


@gtx.field_operator
def limited_update(
    in_field: IJKField,
    src: IJKField,
    alpha_eff: F32,
    sgn: F32,
    zero: F32,
) -> IJKField:
    # interface fluxes of src around each cell
    fxw = src - src(I - 1)
    fxe = src(I + 1) - src
    fys = src - src(J - 1)
    fyn = src(J + 1) - src
    # Xue (2000) Eq. 10 with sign(0)=0: zero unless strictly down-gradient
    fxw = where(sgn * fxw * (in_field - in_field(I - 1)) <= zero, zero, fxw)
    fxe = where(sgn * fxe * (in_field(I + 1) - in_field) <= zero, zero, fxe)
    fys = where(sgn * fys * (in_field - in_field(J - 1)) <= zero, zero, fys)
    fyn = where(sgn * fyn * (in_field(J + 1) - in_field) <= zero, zero, fyn)
    return in_field + alpha_eff * (fxe - fxw + fyn - fys)


@gtx.field_operator
def unlimited_update(
    in_field: IJKField,
    src: IJKField,
    alpha_eff: F32,
) -> IJKField:
    fxw = src - src(I - 1)
    fxe = src(I + 1) - src
    fys = src - src(J - 1)
    fyn = src(J + 1) - src
    return in_field + alpha_eff * (fxe - fxw + fyn - fys)


def _gen_program_source(m: int, lim: bool) -> str:
    """Returns the source of one GT4Py program.

    Args:
        m: Number of Laplacian applications, order/2.
        lim: Whether the limiter is enabled.

    Returns:
        Python source defining the program, with the pre-sweeps as
        separate statements over the extend-cascade domains.
    """
    tag = f"m{m}_{'lim' if lim else 'off'}"
    extra_sig = ", sgn: F32, zero: F32" if lim else ""
    extra_call = ", sgn, zero" if lim else ""
    update = "limited_update" if lim else "unlimited_update"

    tmps = []
    if m >= 2:
        tmps.append("tmp1")
    if m >= 3:
        tmps.append("tmp2")
    tmp_sig = "".join(f", {t}: IJKField" for t in tmps)

    bounds_sig = "".join(
        f", lo{e}: gtx.int32, xhi{e}: gtx.int32, yhi{e}: gtx.int32"
        for e in range(m - 1, 0, -1)
    )

    stmts = []
    src = "in_field"
    dsts = ["tmp1", "tmp2", "tmp1"]  # ping-pong for up to 3 pre-sweeps
    for s in range(1, m):
        e = m - s
        dst = dsts[s - 1]
        stmts.append(
            f"    laplacian({src}, out={dst}, "
            f"domain={{I: (lo{e}, xhi{e}), J: (lo{e}, yhi{e}), K: (0, nz)}})"
        )
        src = dst
    stmts.append(
        f"    {update}(in_field, {src}, alpha_eff{extra_call}, out=out_field, "
        f"domain={{I: (h, nxh), J: (h, nyh), K: (0, nz)}})"
    )
    body = "\n".join(stmts)

    return f'''
@gtx.program
def prog_{tag}(
    in_field: IJKField,
    out_field: IJKField{tmp_sig},
    alpha_eff: F32{extra_sig}{bounds_sig},
    h: gtx.int32,
    nxh: gtx.int32,
    nyh: gtx.int32,
    nz: gtx.int32,
):
{body}
'''


for _m in (1, 2, 3, 4):
    for _lim in (False, True):
        _code = _gen_program_source(_m, _lim)
        _fname = f"<gen:m{_m}_{_lim}>"
        linecache.cache[_fname] = (
            len(_code), None, _code.splitlines(True), _fname)
        exec(compile(_code, _fname, "exec"), globals())


def get_program(order: int, limiter: bool):
    return globals()[f"prog_m{order // 2}_{'lim' if limiter else 'off'}"]


def update_halo(field, num_halo: int):
    f = field.ndarray
    # bottom/top edges (without corners), periodic in J
    f[num_halo:-num_halo, :num_halo] = (
        f[num_halo:-num_halo, -2 * num_halo:-num_halo])
    f[num_halo:-num_halo, -num_halo:] = (
        f[num_halo:-num_halo, num_halo:2 * num_halo])
    # left/right edges (including corners), periodic in I
    f[:num_halo, :] = f[-2 * num_halo:-num_halo, :]
    f[-num_halo:, :] = f[num_halo:2 * num_halo, :]


def apply_diffusion(prog, in_field, out_field, num_halo, num_iter, tail):
    for n in range(num_iter):
        update_halo(in_field, num_halo)
        prog(in_field, out_field, *tail, offset_provider=OFFSET_PROVIDER)
        if n < num_iter - 1:
            in_field, out_field = out_field, in_field
    return in_field, out_field


def write_field_to_file(arr_xyz, num_halo, filename):
    """Writes a field in the binary format of `m_utils.write_field_to_file`.

    Args:
        arr_xyz: Field indexed (i, j, k).
        num_halo: Halo width to record in the header.
        filename: Destination path.
    """
    nx, ny, nz = arr_xyz.shape
    header = np.array([3, 32, num_halo, nx, ny, nz], dtype=np.int32)
    with open(filename, "wb") as f:
        header.tofile(f)
        np.ascontiguousarray(
            arr_xyz.transpose(2, 1, 0)).astype(np.float32).tofile(f)


@click.command()
@click.option("--nx", type=int, required=True)
@click.option("--ny", type=int, required=True)
@click.option("--nz", type=int, required=True)
@click.option("--num_iter", type=int, required=True)
@click.option("--order", type=int, default=4)
@click.option("--limiter", is_flag=True, default=False)
@click.option("--alpha", type=float, default=-1.0)
@click.option("--backend", type=str, default="cpu")
@click.option("--prefix", type=str, default="py_", help="output file prefix")
def main(nx, ny, nz, num_iter, order, limiter, alpha, backend, prefix):
    assert order in (2, 4, 6, 8)
    m = order // 2
    num_halo = m
    if alpha < 0.0:
        alpha = 1.0 / 8.0**m
    sgn = 1.0 if (m + 1) % 2 == 0 else -1.0

    be = backend_str_to_backend[backend]
    domain = gtx.domain({I: nx + 2 * num_halo, J: ny + 2 * num_halo, K: nz})
    in_field = gtx.zeros(domain, dtype=F32, allocator=be)
    # box IC identical to the Fortran setup()
    f = in_field.ndarray
    f[
        num_halo + nx // 4: num_halo + 3 * nx // 4,
        num_halo + ny // 4: num_halo + 3 * ny // 4,
        nz // 4: 3 * nz // 4,
    ] = 1.0
    out_field = gtx.zeros(domain, dtype=F32, allocator=be)
    out_field.ndarray[...] = in_field.ndarray

    prog = get_program(order, limiter)
    if be is not None:
        prog = prog.with_backend(be)

    tmps = []
    if m >= 2:
        tmps.append(gtx.zeros(domain, dtype=F32, allocator=be))
    if m >= 3:
        tmps.append(gtx.zeros(domain, dtype=F32, allocator=be))
    scalars = ((F32(sgn * alpha), F32(sgn), F32(0.0)) if limiter
               else (F32(sgn * alpha),))
    bounds = []
    for e in range(m - 1, 0, -1):
        bounds += [
            gtx.int32(num_halo - e),
            gtx.int32(nx + num_halo + e),
            gtx.int32(ny + num_halo + e),
        ]
    tail = (
        *tmps,
        *scalars,
        *bounds,
        gtx.int32(num_halo),
        gtx.int32(nx + num_halo),
        gtx.int32(ny + num_halo),
        gtx.int32(nz),
    )

    def to_np(a):
        return a.get() if hasattr(a, "get") else np.asarray(a)

    def gpu_sync():
        if backend == "gpu":
            import cupy

            cupy.cuda.runtime.deviceSynchronize()

    write_field_to_file(to_np(in_field.ndarray), num_halo,
                        prefix + "in_field.dat")

    # warmup (compiles the stencil)
    apply_diffusion(prog, in_field, out_field, num_halo, 1, tail)

    gpu_sync()
    tic = time.perf_counter()
    fin, fout = apply_diffusion(prog, in_field, out_field, num_halo,
                                num_iter, tail)
    gpu_sync()
    toc = time.perf_counter()

    print("# ranks nx ny nz num_iter time")
    print("data = np.array( [ \\")
    print(f"[ 1, {nx}, {ny}, {nz}, {num_iter}, {toc - tic:.7E}], \\")
    print("] )")

    update_halo(fout, num_halo)
    write_field_to_file(to_np(fout.ndarray), num_halo, prefix + "out_field.dat")


if __name__ == "__main__":
    main()
