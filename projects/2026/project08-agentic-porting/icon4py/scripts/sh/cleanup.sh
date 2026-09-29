#!/usr/bin/env bash
# [help] Remove common build/cache artifacts from the repo tree.
# Invoked via:  ./scripts/run cleanup [--dry-run]

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run|-n) DRY_RUN=1 ;;
        --help|-h)
            echo "Usage: cleanup [--dry-run]"
            echo "  Remove build/cache artifacts from the repository."
            exit 0
            ;;
        *)
            log_error "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

PATTERNS=(
    "__pycache__"
    ".gt4py_cache"
    ".mypy_cache"
    ".pytest_cache"
    ".ruff_cache"
    "*.egg-info"
    "dist"
    "build"
)

EXCLUDE_PATTERNS=(
    ".git"
    ".venv"
)

for pat in "${PATTERNS[@]}"; do
    # Build find command with exclude patterns
    find_cmd=("find" "$REPO_ROOT" "-name" "$pat")
    for ex_pat in "${EXCLUDE_PATTERNS[@]}"; do
        find_cmd+=("-not" "-path" "*/$ex_pat/*")
    done
    find_cmd+=("-print0")

    while IFS= read -r -d '' target; do
        if [[ "$DRY_RUN" -eq 1 ]]; then
            log_info "[dry-run] would remove: $target"
        else
            log_info "Removing: $target"
            rm -rf "$target"
        fi
    done < <("${find_cmd[@]}" 2>/dev/null)
done

[[ "$DRY_RUN" -eq 0 ]] && log_info "Cleanup complete."
