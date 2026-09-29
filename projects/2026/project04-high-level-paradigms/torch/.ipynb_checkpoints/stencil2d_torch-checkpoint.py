# ******************************************************
#     Program: stencil2d
#      Author: Stefano Ubbiali, Oliver Fuhrer
#       Email: subbiali@phys.ethz.ch, ofuhrer@ethz.ch
#        Date: 04.06.2020
# Description: PyTorch implementation of 4th-order diffusion
# ******************************************************
import click
import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time


#--------------------------------
#setting up PyTorch device (torch.no_grad() used to avoid tracking of gradients)
#--------------------------------


@torch.no_grad()
def laplacian(in_field: torch.Tensor, lap_field: torch.Tensor, num_halo: int, extend: int=0):
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

@torch.no_grad()
def update_halo(field: torch.Tensor, num_halo: int):
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
    field[:, -num_halo:, num_halo:-num_halo] = field[:, num_halo : 2 * num_halo, num_halo:-num_halo]

    # left edge (including corners)
    field[:, :, :num_halo] = field[:, :, -2 * num_halo : -num_halo]

    # right edge (including corners)
    field[:, :, -num_halo:] = field[:, :, num_halo : 2 * num_halo]

@torch.no_grad()
def apply_diffusion(in_field: torch.Tensor, out_field: torch.Tensor, alpha: float, num_halo: int, num_iter: int=1):
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
    tmp_field = torch.empty_like(in_field)

    for n in range(num_iter):
        update_halo(in_field, num_halo)

        laplacian(in_field, tmp_field, num_halo=num_halo, extend=1)
        laplacian(tmp_field, out_field, num_halo=num_halo, extend=0)

        out_field[:, num_halo:-num_halo, num_halo:-num_halo] = (
            in_field[:, num_halo:-num_halo, num_halo:-num_halo]
            - alpha * out_field[:, num_halo:-num_halo, num_halo:-num_halo]
        )

        if n < num_iter - 1:
            in_field, out_field = out_field, in_field
        else:
            update_halo(out_field, num_halo)

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
@torch.no_grad()

def main(in_field_path, out_field_path, num_iter, num_halo=2, device="cpu"):
    """Driver for apply_diffusion that sets up fields and does timings"""

    assert 0 < num_iter <= 1024 * 1024, "You have to specify a reasonable value for num_iter"
    assert 2 <= num_halo <= 256, "You have to specify a reasonable number of halo points"

    #ensure device works on the current machine
    if device == "gpu":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            print("Warning: CUDA not available, using MPS instead")
        else:
            device = "cpu"
            print("Warning: CUDA not available, using CPU instead")
    
    dev = torch.device(device)      
    dtype = torch.float32    #default type for torch tensors
    alpha = 1.0 / 64.0

    in_field = torch.from_numpy(np.load(in_field_path))
    in_field = in_field.to(dev)    

    out_field = in_field.clone()


    # warmup caches
    apply_diffusion(in_field, out_field, alpha, num_halo)
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


    # time the actual work
    tic = time.time()
    apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=num_iter)
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    toc = time.time()

    print(toc - tic)

    # Saving the out field:
    out_field_cpu = out_field.cpu()
    out_field_np = out_field_cpu.detach().numpy()
    np.save(out_field_path, out_field_np)


if __name__ == "__main__":
    main()


