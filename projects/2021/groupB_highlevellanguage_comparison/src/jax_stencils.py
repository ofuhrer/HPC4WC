# from functools import partial
from stencil import Stencil
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap, device_put, lax
from jax.ops import index, index_add, index_update
import jax


@jax.partial(jit, static_argnums=(1, 2))
def shift(array, offset, axis):
    index = (
        (offset, array.shape[axis] - offset)
        if offset >= 0
        else (0, array.shape[axis] - offset)
    )
    sliced = lax.dynamic_slice_in_dim(array, index[0], index[1], axis)
    padding = [(0, 0)] * array.ndim
    padding[axis] = (-min(offset, 0), max(offset, 0))
    return jnp.pad(sliced, padding, mode="constant", constant_values=0)


@jit
def apply_lap_stencil2D(left, right, up, down, in_field):
    return vmap(jnp.subtract)(
        jnp.add(jnp.add(left, right), jnp.add(up, down)), 4 * in_field
    )


@jit
def apply_lap_stencil3D(left, right, up, down, front, back, in_field):
    return vmap(jnp.subtract)(
        jnp.add(jnp.add(jnp.add(left, right), jnp.add(up, down)), jnp.add(front, back)),
        6 * in_field,
    )


class Laplacian2D(Stencil):
    def __init__(self, n) -> None:
        super().__init__(n)
        self.backend_name = f"JAX"
        self.stencil_name = "Laplacian2D"

    def activate(self):
        self.num_halo = 2
        self.in_field = jnp.zeros((self.n, self.n))
        self.in_field = index_add(
            self.in_field,
            index[
                self.num_halo + self.n // 4 : self.num_halo + 3 * self.n // 4,
                self.num_halo + self.n // 4 : self.num_halo + 3 * self.n // 4,
            ],
            1.0,
        )
        self.out_field = jnp.array(self.in_field)

    def deactivate(self):
        jnp.save("out_field", self.out_field)

    def run(self):

        ib = self.num_halo
        ie = -self.num_halo
        jb = self.num_halo
        je = -self.num_halo

        left = shift(self.in_field, +jb, axis=1)
        right = shift(self.in_field, -je, axis=1)
        up = shift(self.in_field, +ib, axis=0)
        down = shift(self.in_field, -ie, axis=0)
        self.out_field = apply_lap_stencil2D(
            left, right, up, down, self.in_field
        ).block_until_ready()

        self.in_field, self.out_field = self.out_field, self.in_field

    def sync(self):
        pass

class Laplacian3D(Stencil):
    def __init__(self, n) -> None:
        super().__init__(n)
        self.backend_name = f"JAX"
        self.stencil_name = "Laplacian3D"

    def activate(self):
        self.num_halo = 2
        self.in_field = jnp.zeros((self.n, self.n, self.n))
        self.out_field = np.copy(self.in_field)

        self.in_field = index_add(
            self.in_field,
            index[
                self.num_halo + self.n // 4 : self.num_halo + 3 * self.n // 4,
                self.num_halo + self.n // 4 : self.num_halo + 3 * self.n // 4,
                self.num_halo + self.n // 4 : self.num_halo + 3 * self.n // 4,
            ],
            1.0,
        )

    def deactivate(self):
        np.save("out_field", self.out_field)

    def run(self):

        ib = self.num_halo
        ie = -self.num_halo
        jb = self.num_halo
        je = -self.num_halo
        kb = self.num_halo
        ke = -self.num_halo

        left = shift(self.in_field, +jb, axis=2)
        right = shift(self.in_field, -je, axis=2)
        up = shift(self.in_field, +ib, axis=1)
        down = shift(self.in_field, -ie, axis=1)
        front = shift(self.in_field, +kb, axis=0)
        back = shift(self.in_field, -ke, axis=0)
        self.out_field = apply_lap_stencil3D(
            left, right, up, down, front, back, self.in_field
        ).block_until_ready()

        self.in_field, self.out_field = self.out_field, self.in_field


class Biharmonic2D(Stencil):
    def __init__(self, n) -> None:
        super().__init__(n)
        self.backend_name = f"JAX"
        self.stencil_name = "Biharmonic2D"

    def activate(self):
        self.num_halo = 2
        self.in_field = jnp.zeros((self.n, self.n))
        self.in_field = index_add(
            self.in_field,
            index[
                self.num_halo + self.n // 4 : self.num_halo + 3 * self.n // 4,
                self.num_halo + self.n // 4 : self.num_halo + 3 * self.n // 4,
            ],
            1.0,
        )
        self.out_field = jnp.array(self.in_field)

    def deactivate(self):
        np.save("out_field", self.out_field)

    def run(self):

        ib = self.num_halo - 1
        ie = -self.num_halo + 1
        jb = self.num_halo - 1
        je = -self.num_halo + 1

        tmp_field = jnp.zeros_like(self.in_field)
        left = shift(self.in_field, +jb, axis=1)
        right = shift(self.in_field, -je, axis=1)
        up = shift(self.in_field, +ib, axis=0)
        down = shift(self.in_field, -ie, axis=0)
        tmp_field = apply_lap_stencil2D(
            left, right, up, down, self.in_field
        ).block_until_ready()

        ib += 1
        ie -= 1
        jb += 1
        je -= 1

        left = shift(tmp_field, +jb, axis=1)
        right = shift(tmp_field, -je, axis=1)
        up = shift(tmp_field, +ib, axis=0)
        down = shift(tmp_field, -ie, axis=0)
        self.out_field = apply_lap_stencil2D(
            left, right, up, down, tmp_field
        ).block_until_ready()

        self.in_field, self.out_field = self.out_field, self.in_field


class Biharmonic3D(Stencil):
    def __init__(self, n) -> None:
        super().__init__(n)
        self.backend_name = f"JAX"
        self.stencil_name = "Biharmonic3D"

    def activate(self):
        self.num_halo = 2
        self.in_field = jnp.zeros((self.n, self.n, self.n))
        self.in_field = index_add(
            self.in_field,
            index[
                self.num_halo + self.n // 4 : self.num_halo + 3 * self.n // 4,
                self.num_halo + self.n // 4 : self.num_halo + 3 * self.n // 4,
                self.num_halo + self.n // 4 : self.num_halo + 3 * self.n // 4,
            ],
            1.0,
        )
        self.out_field = jnp.array(self.in_field)

    def deactivate(self):
        np.save("out_field", self.out_field)

    def run(self):

        ib = self.num_halo - 1
        ie = -self.num_halo + 1
        jb = self.num_halo - 1
        je = -self.num_halo + 1
        kb = self.num_halo - 1
        ke = -self.num_halo + 1

        tmp_field = jnp.zeros_like(self.in_field)
        left = shift(self.in_field, +jb, axis=2)
        right = shift(self.in_field, -je, axis=2)
        up = shift(self.in_field, +ib, axis=1)
        down = shift(self.in_field, -ie, axis=1)
        front = shift(self.in_field, +kb, axis=0)
        back = shift(self.in_field, -ke, axis=0)
        tmp_field = apply_lap_stencil3D(
            left, right, up, down, front, back, self.in_field
        ).block_until_ready()

        ib += 1
        ie -= 1
        jb += 1
        je -= 1
        kb += 1
        ke -= 1

        left = shift(tmp_field, +jb, axis=2)
        right = shift(tmp_field, -je, axis=2)
        up = shift(tmp_field, +ib, axis=1)
        down = shift(tmp_field, -ie, axis=1)
        front = shift(tmp_field, +kb, axis=0)
        back = shift(tmp_field, -ke, axis=0)
        self.out_field = apply_lap_stencil3D(
            left, right, up, down, front, back, tmp_field
        ).block_until_ready()

        self.in_field, self.out_field = self.out_field, self.in_field
