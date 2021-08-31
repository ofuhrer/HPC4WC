import os
from stencil import Stencil

backend = os.getenv("NUMPY_BACKEND")
if backend == None:
    backend = "NUMPY"
    import numpy as np
elif backend == "LEGATE":
    try:
        import legate.numpy as np
    except:
        print("LEGATE NUMPY NOT FOUND\n defaulting to numpy")
        import numpy as np
elif backend == "BOHRIUM":
    try:
        import bohrium as np
    except:
        print("BOHRIUM NUMPY NOT FOUND\n defaulting to numpy")
        import numpy as np
else:
    print("NUMPY BACKEND NOT FOUND\n defaulting to numpy")
    import numpy as np
    

class Laplacian2D(Stencil):
    def __init__(self, n) -> None:
        super().__init__(n)
        self.stencil_name = "Laplacian2D"
        self.backend_name = backend

    def activate(self):
        self.num_halo = 2
        self.in_field = np.zeros((self.n, self.n))
        self.out_field = np.copy(self.in_field)

        self.in_field[
            self.num_halo + self.n // 4: self.num_halo + 3 * self.n // 4,
            self.num_halo + self.n // 4: self.num_halo + 3 * self.n // 4,
        ] = 1.0

    def deactivate(self):
        np.save("out_field", self.out_field)

    def run(self):
        I, J = self.in_field.shape

        ib = self.num_halo
        ie = I - self.num_halo
        jb = self.num_halo
        je = J - self.num_halo
        
        self.out_field[ib:ie, jb:je] = (
            -4.0 * self.in_field[ib:ie, jb:je]
            + self.in_field[ib - 1: ie - 1, jb:je]
            + self.in_field[ib + 1: ie + 1, jb:je]
            + self.in_field[ib:ie, jb - 1: je - 1]
            + self.in_field[ib:ie, jb + 1: je + 1]
        )
        self.in_field, self.out_field = self.out_field, self.in_field
    
    def sync(self):
        #force evaluation
        I, J = self.in_field.shape

        ib = self.num_halo
        ie = I - self.num_halo
        jb = self.num_halo
        je = J - self.num_halo
        print(self.in_field[int((ie-ib)/2),int((je-jb)/2)])


class Laplacian3D(Stencil):
    def __init__(self, n) -> None:
        super().__init__(n)
        self.stencil_name = "Laplacian3D"
        self.backend_name = backend

    def activate(self):
        self.num_halo = 2
        self.in_field = np.zeros((self.n, self.n, self.n))
        self.out_field = np.copy(self.in_field)

        self.in_field[
            self.num_halo + self.n // 4: self.num_halo + 3 * self.n // 4,
            self.num_halo + self.n // 4: self.num_halo + 3 * self.n // 4,
            self.num_halo + self.n // 4: self.num_halo + 3 * self.n // 4,
        ] = 1.0

    def deactivate(self):
        np.save("out_field", self.out_field)

    def run(self):
        I, J, K = self.in_field.shape

        ib = self.num_halo
        ie = I - self.num_halo
        jb = self.num_halo
        je = J - self.num_halo
        kb = self.num_halo
        ke = K - self.num_halo

        self.out_field[ib:ie:, jb:je, kb:ke] = (
            -6.0 * self.in_field[ib:ie, jb:je, kb:ke]
            + self.in_field[ib - 1: ie - 1, jb:je, kb:ke]
            + self.in_field[ib + 1: ie + 1, jb:je, kb:ke]
            + self.in_field[ib:ie, jb - 1: je - 1, kb:ke]
            + self.in_field[ib:ie, jb + 1: je + 1, kb:ke]
            + self.in_field[ib:ie, jb:je, kb - 1: ke - 1]
            + self.in_field[ib:ie, jb:je, kb + 1: ke + 1]
        )

        self.in_field, self.out_field = self.out_field, self.in_field


class Copy2D(Stencil):
    def __init__(self, n) -> None:
        super().__init__(n)
        self.stencil_name = "Copy2D"
        self.backend_name = backend

    def activate(self):
        self.in_field = np.ones((self.n, self.n))
        self.out_field = np.zeros_like(self.in_field)

    def deactivate(self):
        np.save("out_field", self.out_field)

    def run(self):
        self.out_field = np.copy(self.in_field)


class Biharmonic2D(Stencil):
    def __init__(self, n) -> None:
        super().__init__(n)
        self.stencil_name = "Biharmonic2D"
        self.backend_name = backend

    def activate(self):
        self.num_halo = 2
        self.in_field = np.zeros((self.n, self.n))
        self.out_field = np.copy(self.in_field)

        self.in_field[
            self.num_halo + self.n // 4: self.num_halo + 3 * self.n // 4,
            self.num_halo + self.n // 4: self.num_halo + 3 * self.n // 4,
        ] = 1.0

    def deactivate(self):
        np.save("out_field", self.out_field)

    def run(self):
        I, J = self.in_field.shape

        ib = self.num_halo - 1
        ie = I - self.num_halo + 1
        jb = self.num_halo - 1
        je = J - self.num_halo + 1

        tmp_field = np.zeros_like(self.in_field)
        tmp_field[ib:ie, jb:je] = (
            -4.0 * self.in_field[ib:ie, jb:je]
            + self.in_field[ib - 1: ie - 1, jb:je]
            + self.in_field[ib + 1: ie + 1, jb:je]
            + self.in_field[ib:ie, jb - 1: je - 1]
            + self.in_field[ib:ie, jb + 1: je + 1]
        )

        ib += 1
        ie -= 1
        jb += 1
        je -= 1

        self.out_field[ib:ie, jb:je] = (
            -4.0 * tmp_field[ib:ie, jb:je]
            + tmp_field[ib - 1: ie - 1, jb:je]
            + tmp_field[ib + 1: ie + 1, jb:je]
            + tmp_field[ib:ie, jb - 1: je - 1]
            + tmp_field[ib:ie, jb + 1: je + 1]
        )


        self.in_field, self.out_field = self.out_field, self.in_field

class Biharmonic3D(Stencil):
    def __init__(self, n) -> None:
        super().__init__(n)
        self.stencil_name = "Biharmonic3D"
        self.backend_name = backend

    def activate(self):
        self.num_halo = 2
        self.in_field = np.zeros((self.n, self.n, self.n))
        self.out_field = np.copy(self.in_field)

        self.in_field[
            self.num_halo + self.n // 4: self.num_halo + 3 * self.n // 4,
            self.num_halo + self.n // 4: self.num_halo + 3 * self.n // 4,
            self.num_halo + self.n // 4: self.num_halo + 3 * self.n // 4,
        ] = 1.0

    def deactivate(self):
        np.save("out_field", self.out_field)

    def run(self):
        I, J, K = self.in_field.shape

        ib = self.num_halo - 1
        ie = I - self.num_halo + 1
        jb = self.num_halo - 1
        je = J - self.num_halo + 1
        kb = self.num_halo - 1
        ke = K - self.num_halo + 1

        tmp_field = np.zeros_like(self.in_field)
        tmp_field[ib:ie:, jb:je, kb:ke] = (
            -6.0 * self.in_field[ib:ie, jb:je, kb:ke]
            + self.in_field[ib - 1: ie - 1, jb:je, kb:ke]
            + self.in_field[ib + 1: ie + 1, jb:je, kb:ke]
            + self.in_field[ib:ie, jb - 1: je - 1, kb:ke]
            + self.in_field[ib:ie, jb + 1: je + 1, kb:ke]
            + self.in_field[ib:ie, jb:je, kb - 1: ke - 1]
            + self.in_field[ib:ie, jb:je, kb + 1: ke + 1]
        )

        ib += 1
        ie -= 1
        jb += 1
        je -= 1
        kb += 1
        ke -= 1

        self.out_field[ib:ie:, jb:je, kb:ke] = (
            -6.0 * tmp_field[ib:ie, jb:je, kb:ke]
            + tmp_field[ib - 1: ie - 1, jb:je, kb:ke]
            + tmp_field[ib + 1: ie + 1, jb:je, kb:ke]
            + tmp_field[ib:ie, jb - 1: je - 1, kb:ke]
            + tmp_field[ib:ie, jb + 1: je + 1, kb:ke]
            + tmp_field[ib:ie, jb:je, kb - 1: ke - 1]
            + tmp_field[ib:ie, jb:je, kb + 1: ke + 1]
        )
        self.in_field, self.out_field = self.out_field, self.in_field
