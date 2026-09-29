# Report

The report follows the LaTeX setup used by the `rare_weather_sampling` final
report, with reproducible build commands for different operating systems.

## Build

From this directory, choose either option:

```sh
# Local TeX installation (TeX Live/MacTeX with latexmk)
make

# No TeX installation required; Docker downloads a shared TeX Live image
make docker

# On the cluster (santis): rootless podman, arm64-compatible image
make podman
```

`make podman` exists because the login nodes have no TeX installation and
`make docker` does not work there: `ghcr.io/xu-cheng/texlive-full` is published
for amd64 only, while the nodes are arm64. The `podman` target uses the
multi-arch `texlive/texlive:latest-medium` image (which covers every package in
the preamble) and keeps its image store on node-local `/tmp` with the `vfs`
driver, because podman's default overlay store on the network home filesystem
cannot set the xattrs it needs. Since `/tmp` is a tmpfs, the image is re-pulled
the first time you build on a different node.

The PDF is written to `build/report.pdf`. Use `make watch` to rebuild whenever
a source file changes, and `make clean` to remove generated files.

On Windows, use Docker Desktop with WSL or run `latexmk report.tex` from a TeX
Live terminal. In VS Code, install **LaTeX Workshop**; the repository settings
build into the same `build/` directory.

## Collaborating

- The report source is `report.tex`. Coordinate ownership by section and keep
  commits focused so concurrent changes are straightforward to merge.
- Put plots and diagrams in `figures/` and reference them without a path, e.g.
  `\includegraphics{strong-scaling.pdf}`.
- Add sources to `references.bib` and cite them with `\cite{key}`.
- Commit source files and figures, but not `build/` or generated auxiliary files.
- Run `make` (or `make docker`) before pushing so broken references and LaTeX
  errors are caught early.

Missing images render as labeled placeholders, so everyone can build the draft
before all plots are available. Adding a file at the displayed path replaces
the placeholder automatically on the next build.

Please use vector PDF figures for plots where possible; PNG is suitable for
raster images.
