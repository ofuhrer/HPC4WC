# ******************************************************
#     Library: lib_experiment.sh
# Description: Shared helpers for every stencil2d experiment script, so
#              that all runs -- comm strategies, corners variants,
#              comm/compute overlap -- land in ONE comparable format.
#
#              Source it, don't execute it:
#                  source "$(dirname "$0")/lib_experiment.sh"
#
#              The point of this file is that every result_*.py is
#              self-describing: it records not just the timing the binary
#              printed, but the exact command, environment, binary, git
#              commit, wallclock and Slurm job id behind that timing. That
#              is what lets analyze_benchmark_results.py merge several
#              results_*/ directories into one comparative plot without
#              anyone having to remember which label meant what.
# ******************************************************

# ---- fixed cluster settings -------------------------------------------------
# The uenv image must be given to srun explicitly: running `uenv start` in
# the login shell does NOT propagate to job steps, and the binary then dies
# with "error while loading shared libraries" even though it built fine.
: "${SRUN_ACCOUNT:=hpc4wc-course2026-ethz}"
: "${SRUN_UENV:=prgenv-gnu/26.3:v1}"
: "${SRUN_VIEW:=default}"

# ---- weak-scaling grid factorization ----------------------------------------
# Mirrors m_partitioner.F90: ranks_y is the
# largest divisor of n that is <= sqrt(n); ranks_x = n / ranks_y. Callers use
# it both to size the global domain and to check neither dim collapses below
# 3 (with < 3 ranks in a dim and periodic BCs, left()==right() and the corner
# neighbours degenerate, which makes a corners-vs-baseline comparison
# meaningless).
grid_for_ranks() {
    local n=$1 rx ry
    rx=$(awk -v n="${n}" 'BEGIN{r=int(sqrt(n)); for(i=r;i>=1;i--){if(n%i==0){print i; exit}}}')
    ry=$((n / rx))
    echo "${ry} ${rx}"
}

# ---- python string quoting --------------------------------------------------
# The trailer we append is executed by analyze_benchmark_results.py, so any
# value that ends up inside "..." has to survive exec().
py_str() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    printf '"%s"' "${s}"
}

# ---- code-state capture -----------------------------------------------------
# The commit alone is not enough: these sweeps are routinely launched from a
# working tree with uncommitted edits to a stencil2d-*.F90, and six weeks
# later "which code produced this number?" has to be answerable. So we snapshot
# the state ONCE at sweep start (experiment_init) rather than per run -- every
# run of a sweep then reports the same, correct, starting state even if
# somebody edits a file while the sweep is queued.
#
#   EXP_GIT_COMMIT  -- full sha at sweep start
#   EXP_GIT_DIRTY   -- 1 if tracked or untracked files existed at sweep start
#   EXP_GIT_DIFF    -- tracked-file patch (only when dirty), or ""
#   EXP_GIT_STATUS  -- exact porcelain status, including untracked paths
#
# experiment_init <results_dir>  -- call once, after mkdir -p "${results_dir}"
experiment_init() {
    local results_dir="$1"
    shift
    local repo
    repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    EXP_GIT_COMMIT="$(git -C "${repo}" rev-parse HEAD 2>/dev/null || echo unknown)"
    EXP_GIT_BRANCH="$(git -C "${repo}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    EXP_GIT_DIRTY=0
    EXP_GIT_DIFF=""
    EXP_GIT_STATUS=""
    EXP_SWEEP_START="$(date -Is)"
    printf -v EXP_SWEEP_CMD '%q ' "$0" "$@"
    EXP_SWEEP_CMD="${EXP_SWEEP_CMD% }"

    local git_status
    git_status="$(git -C "${repo}" status --porcelain --untracked-files=all 2>/dev/null || true)"
    if [ "${EXP_GIT_COMMIT}" != "unknown" ] && [ -n "${git_status}" ]; then
        EXP_GIT_DIRTY=1
        EXP_GIT_DIFF="${results_dir}/code_state.diff"
        EXP_GIT_STATUS="${results_dir}/code_state.status"
        git -C "${repo}" diff HEAD -- > "${EXP_GIT_DIFF}" 2>/dev/null || true
        printf '%s\n' "${git_status}" > "${EXP_GIT_STATUS}"
        echo "!! working tree is DIRTY at sweep start; saved ${EXP_GIT_DIFF} and ${EXP_GIT_STATUS}" >&2
    fi

    # A human-readable manifest for the whole sweep, next to its results.
    {
        echo "sweep_start   = ${EXP_SWEEP_START}"
        echo "sweep_command = ${EXP_SWEEP_CMD}"
        echo "git_branch    = ${EXP_GIT_BRANCH}"
        echo "git_commit    = ${EXP_GIT_COMMIT}"
        echo "git_dirty     = ${EXP_GIT_DIRTY}"
        echo "git_diff      = ${EXP_GIT_DIFF:-(clean)}"
        echo "git_status    = ${EXP_GIT_STATUS:-(clean)}"
        echo "host          = $(hostname)"
        echo "srun_account  = ${SRUN_ACCOUNT}"
        echo "srun_uenv     = ${SRUN_UENV}"
    } > "${results_dir}/sweep_manifest.txt"

    echo "sweep starting from ${EXP_GIT_BRANCH}@${EXP_GIT_COMMIT:0:8}$([ "${EXP_GIT_DIRTY}" = 1 ] && echo ' (dirty)')"
}

# ---- run_case ---------------------------------------------------------------
# One srun of one binary at one config, written to the canonical filename
#
#     <results_dir>/result_<label>_N<nodes>_n<ranks>_c<cores>_rep<rep>.py
#
# Usage (all flags; --local-nx/--local-ny/--env/--srun-extra/--app-extra
# are optional):
#
#   run_case --label corners --binary ./stencil2d-mpi-corners.x \
#            --results-dir results_ceiling --mode weak --rep 1 \
#            --nodes 9 --ranks 9 --cores 1 \
#            --nx 192 --ny 192 --nz 64 --num_iter 1024 \
#            --local-nx 64 --local-ny 64 \
#            --env "OMP_NUM_THREADS=1 MPICH_ASYNC_PROGRESS=1" \
#            --srun-extra "--ntasks-per-node=1 --cpu-bind=cores" \
#            --app-extra "--comm_strategy sendrecv" \
#            --tag local64_nz256
#
# Sets RUN_CASE_OUTFILE to the file written. Returns non-zero if srun failed
# (the caller decides whether to abort the whole sweep or carry on).
run_case() {
    local label="" binary="" results_dir="" mode="" rep=1
    local nodes="" ranks="" cores=1
    local nx="" ny="" nz="" num_iter=""
    local local_nx="" local_ny=""
    local env_overrides="" srun_extra="" app_extra="" tag=""

    while [ $# -gt 0 ]; do
        case "$1" in
            --label)       label="$2"; shift 2 ;;
            --binary)      binary="$2"; shift 2 ;;
            --results-dir) results_dir="$2"; shift 2 ;;
            --mode)        mode="$2"; shift 2 ;;
            --rep)         rep="$2"; shift 2 ;;
            --nodes)       nodes="$2"; shift 2 ;;
            --ranks)       ranks="$2"; shift 2 ;;
            --cores)       cores="$2"; shift 2 ;;
            --nx)          nx="$2"; shift 2 ;;
            --ny)          ny="$2"; shift 2 ;;
            --nz)          nz="$2"; shift 2 ;;
            --num_iter)    num_iter="$2"; shift 2 ;;
            --local-nx)    local_nx="$2"; shift 2 ;;
            --local-ny)    local_ny="$2"; shift 2 ;;
            --env)         env_overrides="$2"; shift 2 ;;
            --srun-extra)  srun_extra="$2"; shift 2 ;;
            --app-extra)   app_extra="$2"; shift 2 ;;
            --tag)         tag="$2"; shift 2 ;;
            *) echo "run_case: unknown argument: $1" >&2; return 2 ;;
        esac
    done

    # The default filename identifies a run by its rank layout, which is all
    # a scaling sweep varies. A sweep that instead varies the problem size at
    # a FIXED rank count would collide on that name, so such sweeps pass
    # --tag (e.g. "local64_nz256") to disambiguate. The analysis does not
    # depend on this -- aggregate() keys on the real nx/ny/nz/num_iter -- it
    # only keeps files from overwriting each other.
    local outfile="${results_dir}/result_${label}${tag:+_${tag}}_N${nodes}_n${ranks}_c${cores}_rep${rep}.py"
    RUN_CASE_OUTFILE="${outfile}"

    if [ -e "${outfile}" ]; then
        echo "run_case: refusing to overwrite existing result: ${outfile}" >&2
        echo "Use a new --results-dir or increase --repeats." >&2
        return 2
    fi

    # Build the full command as an array first, then keep a printable copy of
    # it for the trailer. What we record is exactly what we run -- no
    # reconstruction after the fact.
    local -a srun_args=(
        -A "${SRUN_ACCOUNT}" --uenv="${SRUN_UENV}" --view="${SRUN_VIEW}"
        -N "${nodes}" -n "${ranks}" -c "${cores}"
    )
    # shellcheck disable=SC2206 -- deliberate word splitting of a flag string
    [ -n "${srun_extra}" ] && srun_args+=(${srun_extra})

    local -a app_args=("${binary}" --nx "${nx}" --ny "${ny}" --nz "${nz}" --num_iter "${num_iter}")
    # shellcheck disable=SC2206
    [ -n "${app_extra}" ] && app_args+=(${app_extra})

    # cmd_str is the LOGICAL command -- what a human would retype to reproduce
    # this run. The command actually executed wraps the binary in a one-line
    # shell (below) purely to report the job id; that wrapper is plumbing, not
    # part of the experiment, so it is deliberately kept out of the record.
    local cmd_str="srun ${srun_args[*]} ${app_args[*]}"
    [ -n "${env_overrides}" ] && cmd_str="env ${env_overrides} ${cmd_str}"

    # srun prints "srun: job N queued and waiting" ONLY when the allocation
    # has to wait -- a run that starts immediately reports no job id at all,
    # and the link to sacct (real node list, start/end) is lost. So ask the
    # allocation itself: rank 0 echoes SLURM_JOB_ID to stderr, then execs the
    # binary in place, which costs one fork and no measured time.
    local -a cmd=(
        srun "${srun_args[@]}"
        bash -c 'if [ "${SLURM_PROCID:-0}" = 0 ]; then echo "run_case_jobid=${SLURM_JOB_ID}" >&2; fi; exec "$@"' _
        "${app_args[@]}"
    )

    local stderr_file
    stderr_file="$(mktemp)"

    local t_start_iso t0 t1 wall_s status=0
    t_start_iso="$(date -Is)"
    t0="$(date +%s.%N)"

    if [ -n "${env_overrides}" ]; then
        # shellcheck disable=SC2086 -- env_overrides is a space-separated VAR=VAL list
        env ${env_overrides} "${cmd[@]}" > "${outfile}" 2> "${stderr_file}" || status=$?
    else
        "${cmd[@]}" > "${outfile}" 2> "${stderr_file}" || status=$?
    fi

    t1="$(date +%s.%N)"
    wall_s="$(awk -v a="${t0}" -v b="${t1}" 'BEGIN{printf "%.3f", b-a}')"

    # srun announces the allocation on stderr; keep it visible in the sweep
    # log the way it always was, but also capture the job id so a run can be
    # traced back to sacct (node list, real start/end) months later.
    # Keep srun's own chatter visible in the sweep log the way it always was,
    # but strip our jobid probe -- it is metadata, not something a reader of
    # the log needs to see on every line.
    grep -v '^run_case_jobid=' "${stderr_file}" >&2 || true
    local jobid
    jobid="$(sed -n 's/^run_case_jobid=\([0-9]\+\).*/\1/p' "${stderr_file}" | head -1)"
    if [ -z "${jobid}" ]; then
        # Fallback for the queued case, and for any binary launched without
        # the wrapper.
        jobid="$(grep -oE 'job [0-9]+' "${stderr_file}" | head -1 | awk '{print $2}' || true)"
    fi
    rm -f "${stderr_file}"

    if [ "${status}" -ne 0 ]; then
        mv "${outfile}" "${outfile}.failed"
        echo "  !! srun failed (status ${status}) for ${outfile}" >&2
        echo "     partial stdout kept as ${outfile}.failed" >&2
        return "${status}"
    fi

    {
        echo ""
        echo "# ---- run metadata (appended by lib_experiment.sh run_case) ----"
        echo "strategy = $(py_str "${label}")"
        echo "nodes = ${nodes}"
        echo "ranks = ${ranks}"
        echo "cores_per_rank = ${cores}"
        echo "mode = $(py_str "${mode}")"
        echo "rep = ${rep}"
        echo "experiment = $(py_str "${results_dir}")"
        echo "binary = $(py_str "${binary}")"
        echo "command = $(py_str "${cmd_str}")"
        echo "env_overrides = $(py_str "${env_overrides}")"
        echo "srun_extra = $(py_str "${srun_extra}")"
        echo "app_extra = $(py_str "${app_extra}")"
        echo "git_commit = $(py_str "${EXP_GIT_COMMIT:-unknown}")"
        echo "git_branch = $(py_str "${EXP_GIT_BRANCH:-unknown}")"
        echo "git_dirty = ${EXP_GIT_DIRTY:-0}"
        echo "git_diff = $(py_str "${EXP_GIT_DIFF:-}")"
        echo "git_status = $(py_str "${EXP_GIT_STATUS:-}")"
        echo "sweep_start = $(py_str "${EXP_SWEEP_START:-}")"
        echo "sweep_command = $(py_str "${EXP_SWEEP_CMD:-}")"
        echo "slurm_jobid = $(py_str "${jobid}")"
        echo "t_start = $(py_str "${t_start_iso}")"
        echo "wall_s = ${wall_s}"
        echo "t_start_approx = 0"
        echo "metadata_source = \"recorded\""
        [ -n "${local_nx}" ] && echo "local_nx = ${local_nx}"
        [ -n "${local_ny}" ] && echo "local_ny = ${local_ny}"
    } >> "${outfile}"

    echo "  -> ${outfile}  (job ${jobid:-?}, ${wall_s}s wall)"
    return 0
}
