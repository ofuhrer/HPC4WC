import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme()


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
df = pd.read_csv(*args.input)
# Manually set first column
df["n"] = [2 ** n for n in range(3, 15)]
print("Initial dataframe:")
print(df)

fig = plt.figure()
ax = fig.add_subplot(111)

# Set axis type according to args give
if args.logx:
    ax.set_xscale("log", basex=2)
if args.logy:
    ax.set_yscale("log", basey=10)

# Escape whitespaces if custom title is given as arg
if args.title is not None:
    args.title = args.title.replace(" ", "\ ")

df.iloc[:, 1:] = 1.0 / df.iloc[:, 1:].div(df[f"NumPy"], axis=0)

#### Lineplot [begin]
# df = df[[col for col in df.columns if col != "NumPy"]]
# ax = df.plot(x="n", ax=ax, marker="o", markersize=4)
# ax.legend()
# plt.hlines(1.0, 8, 16384, linestyles="dashed", colors="black")
#### Lineplot [end]

#### Barplot [begin]
df_barplot = pd.DataFrame(
{
"Libraries": df.columns[1:],
"Speedup": df[[col for col in df.columns[1:]]].iloc[9, :],
}
)
print(df_barplot)
# Drop Reference NUMPY column
df_barplot = df_barplot[df_barplot.Libraries != "NumPy"]
df_barplot = df_barplot[df_barplot.Libraries != "Bohrium (CPU)"]
print(df_barplot)
ax = df_barplot.plot.bar(x="Libraries", rot=0, ax=ax, legend=False)
ax.set_xlabel("Libraries", labelpad=10.0)
ax.set_aspect(aspect="auto")
#### Barplot [end]

if args.title is not None:
    title = f"$\\bf{args.title}$"
else:
    title = r"$\bf{Speedup\ with\ respect\ to\ NumPy}$"
    title += "\n Speedup [-]"
ax.set_title(title, loc="left")


if args.output is None:
    plt.show()
else:
    fig.savefig(args.output, dpi=150)
