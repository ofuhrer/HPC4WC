# Project/report consistency audit — 2026-09-15

## Follow-up status

This section supersedes the original review below. The original findings and
line numbers are retained as a historical record, not as a current checklist.

- **Blocking CSV provenance corrected:** all 120 `source_file` paths in
  `plots/blocking_non_blocking/{strong,weak}/comm_strategy_results.csv` now
  resolve under `results/blocking_non_blocking/`. Measurement values, recorded
  commands, revisions, and experiment labels are unchanged.
- **README corrected:** it now identifies the archived blocking plots and uses
  the existing `comm_strategy_results.csv` / `comm_strategy_results_agg.csv`
  filenames in regeneration commands.
- **Source snapshots are trackable:** `.gitignore` no longer excludes
  `code_state.diff` or `code_state.status`. No historical snapshots were found
  in this checkout, so this does not recover the missing dirty-source history.
  Patches capture tracked edits; untracked source files need separate retention.
- **Validation launcher updated:** `mpi_leftright` now runs through the same
  comparison as the other implementations. Its validation has not been rerun;
  no new passing result is claimed.
- **One-rank archive cleanup completed:** the strong-scaling `corners` and
  `mpi_leftright` one-rank results, failed-run files, and corresponding raw and
  aggregate CSV rows were removed. Existing plot images were not regenerated.
- **Runtime terminology clarified:** Experimental Setup describes mean
  diffusion-loop elapsed time across ranks, matching `timer_get`.
- **Review items closed by the author:** figure integration, validation-count
  wording, corner-baseline wording, repeat-count wording, and plot titles.
  The launcher grid restriction was accepted for the intended configurations.
  These are review decisions, not a claim of fresh build or benchmark validation.

Remaining review topics are the overlap source/schedule description and missing
historical overlap evidence, the pipelined timing-model illustration, the
strong-speedup prose/caption mismatch, and draft text. Unused routines and
dependencies are optional cleanup. No cluster benchmarks, numerical validation,
or plot regeneration were performed for this provenance/documentation update.

## Original review (historical)

The following findings describe the checkout at the time of the initial review.
That review did not change report prose, experiments, or archived data.

## Highest-priority findings

1. **The report does not build: 12 referenced figures are absent.**
   `make -C report` fails at `report.tex:421`. Missing files in `report/figures/`:
   `comm_vs_compute_avg_weak_nico.png`, `total_time_scaling_weak_nico.png`,
   `weak_scaling_efficiency_nico.png`, `COMM_COMP_STRONG.png`, and the strong/weak
   pairs of `comm_relative_bars`, `packing_relative_bars`,
   `sending_relative_bars`, and `waiting_relative_bars`.
   Corresponding blocking and corner plots exist under `plots/`; wire them into
   the report reproducibly. The overlap plot needs regeneration from its archive.
   The two existing strong-scaling images in the report are not byte-identical
   to the corresponding current images anywhere under `plots/`; their provenance
   needs checking before treating them as the new blocking results.

2. **The overlap method and diagnostic narrative do not describe the current source.**
   `report.tex:195` names `stencil2d-comm_comp_opt.F90`, which is absent.
   `stencil2d/stencil2d-comm_comp.F90:182` starts top/bottom communication,
   computes the interior, then calls `update_halo_middle` to complete top/bottom
   and launch left/right sends. It computes top/bottom strips before waiting for
   left/right and computing the remaining strips. The report's generic all-face
   send schedule and the revised left/right-before-interior schedule at line 511
   therefore need to distinguish historical experiments from the checked-in code.
   Current `comm_comp` prints total runtime only, not the claimed wait diagnostics.

3. **Historical overlap claims lack supporting artifacts in this checkout.**
   `results/comm_comp_experiments/` contains only the strong comparison, with
   six basic metadata fields and no recorded launch command or source revision.
   There is no archived overlap weak sweep or the eight-node diagnostic runs
   supporting the 30/70 split, 0.3/3.2 seconds, and 0.16 to 0.02–0.12 seconds
   in `report.tex:492–511`. Those may be valid historical measurements, but cannot
   be verified here. Add their artifacts and historical source or qualify/remove
   the claims. README already acknowledges that a new weak sweep would not
   reproduce the historical diagnostics.

4. **The sweep does not enforce the report's even-grid requirement.**
   `experiment_scaling_all_strategies.sh:65` says blocking methods are safe from
   four ranks onward; its launch loop only checks a minimum rank count.
   However, `sendrecv_evenodd` requires even dimensions on the periodic process
   grid. For example, `--ranks 9` produces a 3×3 grid and is unsafe; a prime rank
   count can also leave a self-neighbor axis for `brokencycles`. Validate process
   grid dimensions per strategy. The archived blocking sweep at 4, 8, 16, 32,
   and 64 ranks does satisfy the requirements.

## Claims and data

5. **Corner conclusions contain stale framing.** The abstract TODO says corners
   do not outperform the baseline, while Section 3.3 correctly reports gains at
   high rank counts. The outlook still refers generally to corner underperformance.
   The final conclusion calls `irecvisend` a corner strategy, although the corner
   comparison uses `mpi_leftright` from `stencil2d-mpi.F90`. Keep the two baselines
   distinct: they come from different executables and experiment sets.

   Current corner CSV means support the reported halo-time winners:

   | Mode/ranks | mpi_leftright (s) | Fastest halo strategy | Halo time (s) |
   | --- | ---: | --- | ---: |
   | strong/64 | 0.2385 | pipelined | 0.2120 |
   | strong/96 | 0.2051 | waitany | 0.1753 |
   | weak/64 | 0.2204 | pipelined | 0.2053 |
   | weak/96 | 0.2505 | waitany | 0.2203 |

   These winners also have the lowest mean total runtime at those configurations.
   Keep measured ordering separate from hypotheses about arrival skew and MPI
   progress; the timers do not directly measure network arrivals or cache misses.

6. **Repeat counts are not uniformly three.** `report.tex:375` says every case
   has three repeats. The weak 96-rank `waitany` aggregate has five. The full
   strong corner archive also has two repeats for the one-rank `corners` case,
   outside the report's plotted 8–96 range. Describe the exceptions and explain
   the additional/replacement runs.

7. **Strong-speedup wording and caption disagree.** At `report.tex:470` the prose
   says all four strategies are superlinear, whereas the caption singles out
   three and says `brokencycles` tracks the ideal line. Using current means,
   4-to-64-rank speedups are 16.24× (`brokencycles`), 19.32× (`irecvisend`),
   19.17× (`sendrecv`), and 18.36× (`sendrecv_evenodd`), versus ideal 16×.
   Describe `brokencycles` as approximately linear and the others as clearly
   superlinear. Cache effects are a plausible explanation, not demonstrated by
   these runtime measurements alone. Also avoid an absolute fastest-everywhere
   claim for `irecvisend`: at eight ranks, `sendrecv` has a slightly lower strong
   mean (9.9329 vs 9.9419 seconds).

8. **MPI explanations overstate what calls guarantee.** Around `report.tex:449`,
   blocking send completion is described as completion of the entire message,
   and nonblocking calls as immediate handoff to the network with guaranteed
   hardware overlap. MPI_Send completion means the send buffer is reusable;
   it can precede matching receive completion. Nonblocking initiation permits
   overlap but does not prove it happened. Likewise, standard corners can begin
   transfers while subsequent sends are being posted, not only after all sends
   are posted as claimed later in Section 3.3.
   Sources: [MPI communication modes](https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node57.htm)
   and [MPI nonblocking communication](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node71.htm).
   The broken-cycle implementation exchanges neighbor data; it does not forward
   each payload around the ring. It is the dependency that propagates.

## Validation and reproducibility

9. **Numerical validation is supported, but exact source preservation is not.**
   The manifest confirms 128×128×4, 1024 iterations, 16 ranks on four nodes.
   All ten current strategy fields have identical SHA-256 hashes; logs report
   the stated 6.204e-5 differences from the serial field. The archive also
   contains an eleventh passing historical `comm_comp_opt` result.
   However, the manifest records a dirty checkout, and `git_status.txt` is a
   list of changed paths, not a source snapshot or patch. The report's claim
   that the code state is preserved is too strong. This evidence validates the
   archived outputs; it does not establish validation of every current edit.

10. **Moved blocking CSV provenance paths are stale.** Both
    `plots/blocking_non_blocking/{strong,weak}/comm_strategy_results.csv`
    retain `source_file` paths beginning at the former root directories.
    Regenerate these CSVs against the new results paths. README's canonical
    names are `runs.csv` and `aggregated.csv`, while the moved files retain
    `comm_strategy_results.csv` and `comm_strategy_results_agg.csv`.
    Historical commands should remain intact as provenance.

11. **Draft/build documentation still needs reconciliation.**
    `sec:results-ordering` is an undefined reference. Abstract, conclusion,
    references, and setup contain TODOs, including a stale 512-rank example
    outside these datasets. Sentences promising launch commands in the report
    have no following commands. `report/README.md` incorrectly promises missing
    image placeholders; the actual build fails. It also recommends a BibTeX
    file absent from the manual `thebibliography` setup and path-free figure
    references without a configured graphics search path.

## Checks performed and limits

- Read report, build configuration, README, launch/analysis scripts, relevant
  Fortran communication schedules and buffer sizes, and archived metadata.
- Checked all 19 graphics references: 12 missing; checked label references.
- Ran the local LaTeX build; failure reproduced at the first missing image.
- Independently recomputed blocking total/communication/computation means and
  repeat counts from all 120 raw runs; all match the two aggregate CSVs.
- Inspected corner aggregates and repeat counts. Equal payload volume is
  supported by buffer sizes: both schemes send
  `2*nz*h*(nx+ny) + 4*nz*h*h` values per rank per update.
- Recomputed archived overlap strong means: `comm_comp` is slower at every
  sampled rank count, from 93.3747 vs 92.2029 seconds at one rank to 3.6665 vs
  3.4871 seconds at 64 ranks. These records do not establish original placement.
- Checked validation logs and hashed archived fields.
- Did not rerun cluster benchmarks or current-source numerical validation.
  Plot regeneration was not run; the default Python environment lacks matplotlib.
