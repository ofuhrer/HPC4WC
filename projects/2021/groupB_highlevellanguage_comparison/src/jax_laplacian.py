from stencil import Stencil
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap, device_put
from jax.ops import index, index_add, index_update


def axis_slice(ndim, index, axis):
	slices = [slice(None)] * ndim
	slices[axis] = index
	return tuple(slices)

def slice_along_axis(array, index, axis):
	return array[axis_slice(array.ndim, index, axis)]

def shift(array, offset, axis):
	index = slice(offset, None) if offset >= 0 else slice(None, offset)
	sliced = slice_along_axis(array, index, axis)
	padding = [(0, 0)] * array.ndim
	padding[axis] = (-min(offset, 0), max(offset, 0))
	return jnp.pad(sliced, padding, mode='constant', constant_values=0)

def laplacian(in_field, lap_field, num_halo, extend=0):
    """ Compute the Laplacian using 2nd-order centered differences.
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

    # lap_field[:, jb:je, ib:ie] = (
        # -4.0 * in_field[:, jb:je, ib:ie]
        # + in_field[:, jb:je, ib - 1 : ie - 1]
        # + in_field[:, jb:je, ib + 1 : ie + 1 if ie != -1 else None]
        # + in_field[:, jb - 1 : je - 1, ib:ie]
        # + in_field[:, jb + 1 : je + 1 if je != -1 else None, ib:ie]
    # )

    left = shift(in_field, +jb, axis=0)
    right = shift(in_field, -je, axis=0)
    up = shift(in_field, +ib, axis=1)
    down = shift(in_field, -ie, axis=1)
    lap_field = (left + right + up + down - 4 * in_field)

def halo_update(field, num_halo):
    """ Update the halo-zone using an up/down and left/right strategy.
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
    """ Integrate 4th-order diffusion equation by a certain number of iterations.
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
    tmp_field = jnp.empty_like(in_field)

    for n in range(num_iter):

        laplacian(in_field, tmp_field, num_halo=num_halo, extend=1)
        laplacian(tmp_field, out_field, num_halo=num_halo, extend=0)

        out_field = index_update(out_field, index[:, num_halo:-num_halo, num_halo:-num_halo],
            in_field[:, num_halo:-num_halo, num_halo:-num_halo]
            - alpha * out_field[:, num_halo:-num_halo, num_halo:-num_halo]
        )

        in_field, out_field = out_field, in_field


class Numpy_diffusion(Stencil):
    def __init__(self, n) -> None:
        super().__init__(n)
    
    def __str__(self) -> str:
        try:
            s = f"Diffusion Scheme with Numpy version {np.version.version}"
        except AttributeError:
            s = f"Diffusion Scheme with Alternate Numpy Backend"
        return s

    def activate(self, backend="Numpy"):

        nx = self.n
        ny = self.n
        self.num_halo = 2
        nz = 30
        self.in_field = jnp.zeros((nz, ny + 2 * self.num_halo, nx + 2 * self.num_halo))
        self.in_field = index_add(self.in_field, index[
        nz // 4 : 3 * nz // 4,
            self.num_halo + ny // 4 : self.num_halo + 3 * ny // 4,
            self.num_halo + nx // 4 : self.num_halo + 3 * nx // 4,
        ], 1.0)
        self.out_field = jnp.array(self.in_field)
        self.alpha = 1.0 / 32.0

    def deactivate(self):
        pass

    def run(self):
        apply_diffusion(self.in_field, self.out_field, self.alpha, self.num_halo)
        # jnp.save("out_field", self.out_field)

    def save(self):
        jnp.save("out_field", self.out_field)

    
    
