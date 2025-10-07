import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import time

# -----------------------------
# Torch kernels (no autograd)
# -----------------------------

@torch.no_grad()
def update_halo(field: torch.Tensor, num_halo: int):
    """
    Periodic BC via halo copy. Works on CPU or GPU (in-place on device).
    field shape: [nz, ny+2h, nx+2h]
    """
    h = num_halo
    # bottom/top (without corners)
    field[:, :h, h:-h] = field[:, -2*h:-h, h:-h]
    field[:, -h:, h:-h] = field[:, h:2*h, h:-h]
    # left/right (including corners)
    field[:, :, :h] = field[:, :, -2*h:-h]
    field[:, :, -h:] = field[:, :, h:2*h]


@torch.no_grad()
def laplacian(in_field: torch.Tensor, lap_field: torch.Tensor, num_halo: int, extend: int = 0):
    """
    5-point Laplacian, writes into lap_field. Only the target slice is written.
    extend=1: one cell into halo; extend=0: interior only.
    """
    h = num_halo
    ib = h - extend
    ie = -h + extend
    jb = h - extend
    je = -h + extend

    lap_field[:, jb:je, ib:ie] = (
        -4.0 * in_field[:, jb:je, ib:ie]
        + in_field[:, jb:je, ib - 1 : ie - 1]
        + in_field[:, jb:je, ib + 1 : ie + 1 if ie != -1 else None]
        + in_field[:, jb - 1 : je - 1, ib:ie]
        + in_field[:, jb + 1 : je + 1 if je != -1 else None, ib:ie]
    )


@torch.no_grad()
def axpy_diffusion_step(in_field: torch.Tensor, lap2_field: torch.Tensor,
                        out_field: torch.Tensor, alpha: float, num_halo: int):
    """
    out = in - alpha * lap2  on the interior only.
    """
    h = num_halo
    out_field[:, h:-h, h:-h] = in_field[:, h:-h, h:-h] - alpha * lap2_field[:, h:-h, h:-h]


@torch.no_grad()
def apply_diffusion(in_field: torch.Tensor, out_field: torch.Tensor,
                    alpha: float, num_halo: int, num_iter: int = 1):
    """
    Perform num_iter steps. Ensures out_field holds the final result when done.
    Uses pointer swapping; only copies if num_iter is even.
    """
    # local "pointers"
    a = in_field
    b = out_field
    tmp = torch.empty_like(in_field)

    for n in range(num_iter):
        update_halo(a, num_halo)
        laplacian(a, tmp, num_halo=num_halo, extend=1)
        laplacian(tmp, b, num_halo=num_halo, extend=0)
        axpy_diffusion_step(a, b, b, alpha, num_halo)
        if n < num_iter - 1:
            a, b = b, a

    # update halos of the final field (currently in b)
    update_halo(b, num_halo)

    # Guarantee: final result ends up in out_field
    if (num_iter % 2) == 0:
        out_field.copy_(in_field)  # even num_iter -> result resides in 'in_field'


# -----------------------------
# Driver
# -----------------------------
@click.command()
@click.option("--nx", type=int, required=True)
@click.option("--ny", type=int, required=True)
@click.option("--nz", type=int, required=True)
@click.option("--num_iter", type=int, required=True)
@click.option("--num_halo", type=int, default=2)
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cpu",
              help="Set to 'cuda' to run on GPU.")
@click.option("--plot_result", is_flag=True, default=False)  # flag, no argument needed
def main(nx, ny, nz, num_iter, num_halo=2, device="cpu", plot_result=False):
    assert 0 < nx <= 1024 * 1024
    assert 0 < ny <= 1024 * 1024
    assert 0 < nz <= 1024
    assert 0 < num_iter <= 1024 * 1024
    assert 2 <= num_halo <= 256

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        torch.cuda.set_device(0)

    dev = torch.device(device)
    dtype = torch.float32
    alpha = 1.0 / 32.0  # python float is fine (no autograd)

    # allocate with halos, float32 (matches Fortran wp=4)
    in_field = torch.zeros((nz, ny + 2 * num_halo, nx + 2 * num_halo), dtype=dtype, device=dev)
    in_field[
        nz // 4 : 3 * nz // 4,
        num_halo + ny // 4 : num_halo + 3 * ny // 4,
        num_halo + nx // 4 : num_halo + 3 * nx // 4,
    ] = 1.0
    out_field = in_field.clone()

    # save input
    torch.save(in_field, "in_field_torch.pt")

    # optional plot (middle z-slice)
    if plot_result:
        plt.ioff()
        mid = in_field.shape[0] // 2
        plt.imshow(in_field[mid].detach().cpu().numpy(), origin="lower")
        plt.colorbar()
        plt.savefig("in_field_torch.png")
        plt.close()

    # warmup (compile kernels/caches)
    _in_w = in_field.clone()
    _out_w = out_field.clone()
    apply_diffusion(_in_w, _out_w, alpha, num_halo, num_iter=1)
    if device == "cuda":
        torch.cuda.synchronize()

    # timed run
    tic = time.time()
    apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=num_iter)
    if device == "cuda":
        torch.cuda.synchronize()
    toc = time.time()

    print(f"Elapsed time for work = {toc - tic:.6f} s")

    # save output
    torch.save(out_field, "out_field_torch.pt")

    if plot_result:
        mid = out_field.shape[0] // 2
        plt.imshow(out_field[mid].detach().cpu().numpy(), origin="lower")
        plt.colorbar()
        plt.savefig("out_field_torch.png")
        plt.close()


if __name__ == "__main__":
    # Disable autograd globally (just to be safe)
    torch.set_grad_enabled(False)
    # Optional: control CPU threads from env (e.g., OMP_NUM_THREADS)
    # torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "0")) or torch.get_num_threads())
    main()
