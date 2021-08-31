from stencil import Stencil
import numpy as np
import gt4py
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage

from gt4py.gtscript import (
    PARALLEL,
    computation,
    interval,
    Field,
    FORWARD,
    PARALLEL,

)

backend = "gtx86"
F_TYPE = np.float64
I_TYPE = np.int64

class Laplacian2D(Stencil):
    def __init__(self, n, backend) -> None:
        super().__init__(n)
        self.backend_name = f"gt4py:{backend}"
        self.stencil_name = "Laplacian2D"
        self.backend = backend

    def activate(self):
        self.num_halo = 2
        in_field = np.zeros((self.n, self.n, 1))
        in_field[
            self.num_halo + self.n // 4: self.num_halo + 3 * self.n // 4,
            self.num_halo + self.n // 4: self.num_halo + 3 * self.n // 4, 0
        ] = 1.0

        self.in_field  = gt_storage.from_array(in_field, backend=self.backend, dtype=F_TYPE, shape=(self.n,self.n,1), default_origin=(0,0,0))
        self.out_field  = gt_storage.zeros(backend=self.backend, dtype=F_TYPE, shape=(self.n,self.n,1), default_origin=(0,0,0))

        self.laplacian2D = gtscript.stencil(backend=self.backend, definition=laplacian2D_stencil)


    def deactivate(self):
        out_field = self.out_field.view(np.ndarray)
        np.reshape(out_field, (out_field.shape[0], out_field.shape[1]))
        np.save("out_file", out_field)


    def run(self):

        origin = (self.num_halo, self.num_halo, 0)
        domain = (
            self.in_field.shape[0] - 2 * self.num_halo,
            self.in_field.shape[1] - 2 * self.num_halo,
            self.in_field.shape[2]
        )

        self.laplacian2D(in_field=self.in_field, out_field=self.out_field, origin=origin, domain=domain)

        self.in_field, self.out_field = self.out_field, self.in_field

    def sync(self):
         if self.backend == "gtcuda":
            self.in_field.synchronize()
            self.out_field.synchronize()


def laplacian2D_stencil(in_field: Field[F_TYPE], out_field: Field[F_TYPE]):
    with computation(PARALLEL), interval(...):
        out_field = (
            -4.0 * in_field[ 0,  0, 0]
                 + in_field[-1,  0, 0]
                 + in_field[ 1,  0, 0]
                 + in_field[ 0, -1, 0]
                 + in_field[ 0,  1, 0]
        )

class Laplacian3D(Stencil):
    def __init__(self, n, backend) -> None:
        super().__init__(n)
        self.backend_name = f"gt4py:{backend}"
        self.stencil_name = "Laplacian3D"
        self.backend = backend

    def activate(self):
        self.num_halo = 2
        in_field = np.zeros((self.n, self.n, self.n))
        in_field[
            self.num_halo + self.n // 4: self.num_halo + 3 * self.n // 4,
            self.num_halo + self.n // 4: self.num_halo + 3 * self.n // 4,
            self.num_halo + self.n // 4: self.num_halo + 3 * self.n // 4,
        ] = 1.0

        self.in_field  = gt_storage.from_array(in_field, backend=backend, dtype=F_TYPE, shape=(self.n,self.n,self.n), default_origin=(0,0,0))
        self.out_field  = gt_storage.zeros(backend=backend, dtype=F_TYPE, shape=(self.n,self.n,self.n), default_origin=(0,0,0))

        self.laplacian3D = gtscript.stencil(backend=self.backend, definition=laplacian3D_stencil)
    
    def deactivate(self):
        out_field = self.out_field.view(np.ndarray)
        np.save("out_file", out_field)

    def run(self):

        origin = (self.num_halo, self.num_halo, self.num_halo)
        domain = (
            self.in_field.shape[0] - 2 * self.num_halo,
            self.in_field.shape[1] - 2 * self.num_halo,
            self.in_field.shape[2] - 2 * self.num_halo,
        )

        self.laplacian3D(in_field=self.in_field, out_field=self.out_field, origin=origin, domain=domain)
        
        self.in_field, self.out_field = self.out_field, self.in_field

        if self.backend == "gtcuda":
            self.in_field.synchronize()
            self.out_field.synchronize()


def laplacian3D_stencil(in_field: Field[F_TYPE], out_field: Field[F_TYPE]):
    with computation(PARALLEL), interval(...):
        out_field = (
            -6.0 * in_field[ 0,  0,  0]
                 + in_field[-1,  0,  0]
                 + in_field[ 1,  0,  0]
                 + in_field[ 0, -1,  0]
                 + in_field[ 0,  1,  0]
                 + in_field[ 0,  0, -1]
                 + in_field[ 0,  0,  1]
        )

class Biharmonic2D(Stencil):
    def __init__(self, n, backend) -> None:
        super().__init__(n)
        self.backend_name = f"gt4py:{backend}"
        self.stencil_name = "Biharmonic2D"
        self.backend = backend

    def activate(self):
        self.num_halo = 2
        in_field = np.zeros((self.n, self.n, 1))
        in_field[
            self.num_halo + self.n // 4: self.num_halo + 3 * self.n // 4,
            self.num_halo + self.n // 4: self.num_halo + 3 * self.n // 4, 0
        ] = 1.0

        self.in_field  = gt_storage.from_array(in_field, backend=self.backend, dtype=F_TYPE, shape=(self.n,self.n,1), default_origin=(0,0,0))
        self.out_field  = gt_storage.zeros(backend=self.backend, dtype=F_TYPE, shape=(self.n,self.n,1), default_origin=(0,0,0))

        self.biharmonic2D = gtscript.stencil(backend=self.backend, definition=biharmonic2D_stencil)

    def deactivate(self):
        out_field = self.out_field.view(np.ndarray)
        np.reshape(out_field, (out_field.shape[0], out_field.shape[1]))
        np.save("out_file", out_field)

    def run(self):

        origin = (self.num_halo+1, self.num_halo+1, 0)
        domain = (
            self.in_field.shape[0] - 2 * (self.num_halo+1),
            self.in_field.shape[1] - 2 * (self.num_halo+1),
            self.in_field.shape[2]
        )

        self.biharmonic2D(in_field=self.in_field, out_field=self.out_field, origin=origin, domain=domain)
        self.in_field, self.out_field = self.out_field, self.in_field

    def sync(self):
         if self.backend == "gtcuda":
            self.in_field.synchronize()
            self.out_field.synchronize()


def biharmonic2D_stencil(in_field: Field[F_TYPE], out_field: Field[F_TYPE]):
    with computation(PARALLEL), interval(...):
        tmp_field = (
            -4.0 * in_field[ 0,  0, 0]
                 + in_field[-1,  0, 0]
                 + in_field[ 1,  0, 0]
                 + in_field[ 0, -1, 0]
                 + in_field[ 0,  1, 0]
        )

        out_field = (
            -4.0 * tmp_field[ 0,  0, 0]
                 + tmp_field[-1,  0, 0]
                 + tmp_field[ 1,  0, 0]
                 + tmp_field[ 0, -1, 0]
                 + tmp_field[ 0,  1, 0]
        )


        

class Biharmonic3D(Stencil):
    def __init__(self, n, backend) -> None:
        super().__init__(n)
        self.backend_name = f"gt4py:{backend}"
        self.stencil_name = "Biharmonic3D"
        self.backend = backend

    def activate(self):
        self.num_halo = 2
        in_field = np.zeros((self.n, self.n, self.n))
        in_field[
            self.num_halo + self.n // 4: self.num_halo + 3 * self.n // 4,
            self.num_halo + self.n // 4: self.num_halo + 3 * self.n // 4,
            self.num_halo + self.n // 4: self.num_halo + 3 * self.n // 4,
        ] = 1.0

        self.in_field  = gt_storage.from_array(in_field, backend=self.backend, dtype=F_TYPE, shape=(self.n,self.n,self.n), default_origin=(0,0,0))
        self.out_field  = gt_storage.zeros(backend=self.backend, dtype=F_TYPE, shape=(self.n,self.n,self.n), default_origin=(0,0,0))
        self.biharmonic3D = gtscript.stencil(backend=self.backend, definition=biharmonic3D_stencil)

    def deactivate(self):
        out_field = self.out_field.view(np.ndarray)
        np.save("out_file", out_field)

    def run(self):

        origin = (self.num_halo+1, self.num_halo+1, self.num_halo+1)
        domain = (
            self.in_field.shape[0] - 2 * (self.num_halo+1),
            self.in_field.shape[1] - 2 * (self.num_halo+1),
            self.in_field.shape[2] - 2 * (self.num_halo+1),
        )

        self.biharmonic3D(in_field=self.in_field, out_field=self.out_field, origin=origin, domain=domain)
        self.in_field, self.out_field = self.out_field, self.in_field

        if self.backend == "gtcuda":
            self.in_field.synchronize()
            self.out_field.synchronize()


def biharmonic3D_stencil(in_field: Field[F_TYPE], out_field: Field[F_TYPE]):
    with computation(PARALLEL), interval(...):
        tmp_field = (
            -6.0 * in_field[ 0,  0,  0]
                 + in_field[-1,  0,  0]
                 + in_field[ 1,  0,  0]
                 + in_field[ 0, -1,  0]
                 + in_field[ 0,  1,  0]
                 + in_field[ 0,  0, -1]
                 + in_field[ 0,  0,  1]
        )

        out_field = (
            -6.0 * tmp_field[ 0,  0,  0]
                 + tmp_field[-1,  0,  0]
                 + tmp_field[ 1,  0,  0]
                 + tmp_field[ 0, -1,  0]
                 + tmp_field[ 0,  1,  0]
                 + tmp_field[ 0,  0, -1]
                 + tmp_field[ 0,  0,  1]
        )

