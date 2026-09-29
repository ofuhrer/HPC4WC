import numpy as np

import yasmin as yasi

x, y = yasi.Dimension("x", "y")

u = yasi.Field("u", dims=(x, y), dtype=yasi.float64)
out = yasi.Field("out", dims=(x, y), dtype=yasi.float64)


@yasi.stencil
def laplace(f: yasi.Field) -> yasi.SymbolicExpr:
    return f[-1, 0] + f[1, 0] + f[0, -1] + f[0, 1] - 4.0 * f[0, 0]


@yasi.operator
def apply(out: yasi.Field, u: yasi.Field) -> None:
    out[0, 0] = laplace(u)


def main() -> None:
    u_data = np.arange(64, dtype=np.float64).reshape(8, 8)
    out_data = np.zeros_like(u_data)

    yasi.execute(
        apply(out, u),
        backend="numpy",  # | "cpp" | "openmp"
        fields={u: u_data, out: out_data},
    )

    # For advanced OpenMP, compile with e.g.
    # config = yasi.CompileConfig(
    #     backend="openmp",
    #     options=yasi.OpenMPOptions(num_threads=4),
    # )
    # kernel = yasi.compile(apply(out, u), config=config)

    print(out_data)


if __name__ == "__main__":
    main()
