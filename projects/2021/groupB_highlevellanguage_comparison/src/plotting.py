import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


def compute_flops(n, kind):
    d = kind[-2]
    if kind[:-2] == "Laplacian":
        return (5 if d == 2 else 7) * (n - 4) ** float(d)
    elif kind[:-2] == "Biharmonic":
        return 2 * (5 if d == 2 else 7) * (n - 4) ** float(d)


def change_lib_names(df):
    names = [col.split("|")[1] for col in df.columns]
    for i in range(len(names)):
        if names[i] == "NUMPY":
            names[i] = "NumPy"
        if names[i] == "BOHRIUM":
            names[i] = "Bohrium"
        if names[i] == "JAX":
            names[i] = "JAX"
        if names[i] == "LEGATE":
            names[i] = "Legate"
        if names[i] == "gt4py:gtx86" or names[i] == "gt4py:gtcuda":
            names[i] = "GT4Py"
    return names


parser = argparse.ArgumentParser()
parser.add_argument(
    "--libs",
    type=str,
    nargs="+",
    default=None,
    choices=["NUMPY", "gt4py:gtx86", "gt4py:gtcuda", "JAX", "BOHRIUM"],
)
parser.add_argument(
    "--stencils",
    type=str,
    nargs="+",
    default=None,
    choices=["Laplacian2D", "Laplacian3D", "Biharmonic2D", "Biharmonic3D"],
)
parser.add_argument("--output", type=str, default=None, help="Output pdf file")
parser.add_argument(
    "--input",
    type=str,
    nargs="+",
    default="timings.cvs",
    help="One or multiple path to csv files",
)
parser.add_argument(
    "--title", type=str, default=None, help="Custom title shown on  plot"
)
parser.add_argument(
    "--logx", help="Set x-axis to logarithmic. Default: False", action="store_true"
)
parser.add_argument(
    "--logy", help="Set y-axis to logarithmic. Default: False", action="store_true"
)
parser.add_argument(
    "--speedup",
    type=str,
    default=None,
    help="speedup in barchart or plot form",
    choices=["None", "bar", "line"],
)
parser.add_argument(
    "--perfplot",
    help="create a performance plot normalized by number of flops",
    action="store_true",
)

args = parser.parse_args()

# Collect data (possibly from multiple files)
dfs = []
for i, file in enumerate(args.input):
    dfs.append(pd.read_csv(file))
    if i != 0:
        dfs[i] = dfs[i].iloc[:, 1:]

df_init = pd.concat(dfs, axis=1, ignore_index=False)
# Manually set first column
df_init["n"] = [2 ** n for n in range(3, 15)]
print("Initial dataframe:")
print(df_init)

fig = plt.figure()
ax = fig.add_subplot(111)

# Set axis type according to args give
if args.logx and args.speedup != "bar":
    ax.set_xscale("log", basex=2)
if args.logy and args.speedup != "bar":
    ax.set_yscale("log", basey=10)

# Escape whitespaces if custom title is given as arg
if args.title is not None:
    args.title = args.title.replace(" ", "\ ")

# Populate args.stencils or args.libs if not set with the values found in table
type_set = [set(), set()]
for col in df_init.columns:
    if col == "n" or col == "cores":
        continue
    s, l = col.split("|")
    type_set[0].add(s)
    type_set[1].add(l)

if args.stencils is None:
    args.stencils = list(type_set[0])
if args.libs is None:
    args.libs = list(type_set[1])

print(f"libs: {args.libs}")
print(f"stencils: {args.stencils}")

# Create selection of columns depending on library and/or stencil
data_cols = ["n"] + [f"{s}|{l}" for s in args.stencils for l in args.libs]
if "cores" in df_init.columns:
    data_cols += ["cores"]
df = df_init.filter(data_cols, axis=1)
print("Filtered dataframe according to args")
print(df)

if args.speedup is not None:
    for s in args.stencils:
        df.iloc[:, 1:] = 1.0 / df.iloc[:, 1:].div(df[f"{s}|NUMPY"], axis=0)

        if args.speedup == "bar":
            df_barplot = pd.DataFrame(
                {
                    "Libraries": change_lib_names(df.loc[:, df.columns != "n"]),
                    "Speedup": df[[col for col in df.columns if s in col]].iloc[10, :],
                }
            )
            # Drop Reference NUMPY column
            df_barplot = df_barplot[df_barplot.Libraries != "NumPy"]
            print(df_barplot)
            ax = df_barplot.plot.bar(x="Libraries", rot=0, ax=ax, legend=False)
            if args.title is not None:
                title = f"$\\bf{args.title}$"
            else:
                title = r"$\bf{Speedup\ with\ respect\ to\ NumPy}$"
            title += "\n Speedup [-]"
            ax.set_title(title, loc="left")

        elif args.speedup == "line":
            # Drop Reference NUMPY column
            df = df[df.columns.drop(list(df.filter(regex="NUMPY")))]
            # Remove stencil in column-names
            df.columns = ["n"] + change_lib_names(df.loc[:, df.columns != "n"])
            # Reorder columns alphabetically
            df = df.reindex(sorted(df.columns), axis=1)
            ax = df.plot(x="n", ax=ax, marker="o", markersize=4)
            ax.legend()
            ax.set_xlabel("n per dimension")
            if args.title is not None:
                title = f"$\\bf{args.title}$"
            else:
                title = r"$\bf{Speedup\ with\ respect\ to\ NumPy}$"
            title += "\n Speedup [-]"
            ax.set_title(title, loc="left")
elif args.perfplot:
    print(df["n"])
    for col in df.columns[1:]:
        print(f"altering col {col}")
        df[col] = (1e-9 / df[col]) * compute_flops(df["n"], col.split("|")[0])
    print("After normalization:")
    print(df)
    # Remove stencil in column-names
    df.columns = ["n"] + change_lib_names(df.loc[:, df.columns != "n"])
    # Reorder columns alphabetically
    df = df.reindex(sorted(df.columns), axis=1)
    ax = df.plot(x="n", ax=ax, marker="o", markersize=4)
    if args.title is not None:
        title = f"$\\bf{args.title}$"
    else:
        title = r"$\bf{Approximate\ performance\ normalized\ per\ stencil}$"
    title += "\n [Gflop / s]"
    ax.set_title(title, loc="left")
else:
    # Remove stencil in column-names
    df.columns = ["n"] + change_lib_names(df.loc[:, df.columns != "n"])
    # Reorder columns alphabetically
    df = df.reindex(sorted(df.columns), axis=1)
    ax = df.plot(x="n", ax=ax, marker="o", markersize=4)
    ax.legend()
    ax.set_xlabel("n per dimension")
    if args.title is not None:
        title = f"$\\bf{args.title}$"
    else:
        title = r"$\bf{Runtime\ plot}$"
    title += "\n Runtime [s]"
    ax.set_title(title, loc="left")


if args.output is None:
    plt.show()
else:
    fig.savefig(args.output, dpi=150)
