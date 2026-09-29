#!/usr/bin/env bats
# Tests for sh/_lib.sh shared bash functions.
# Run with:  bats scripts/tests/sh/

setup() {
    # Source the library under test
    source "$BATS_TEST_DIRNAME/../../sh/_lib.sh"
}

@test "REPO_ROOT points to a directory containing scripts/" {
    [ -d "$REPO_ROOT" ]
    [ -d "$REPO_ROOT/scripts" ]
}

@test "SCRIPTS_DIR points to the scripts directory" {
    [ -d "$SCRIPTS_DIR" ]
    [ -f "$SCRIPTS_DIR/run" ]
}

@test "require_cmd succeeds for a command that exists" {
    run require_cmd bash
    [ "$status" -eq 0 ]
}

@test "require_cmd fails for a nonexistent command" {
    run require_cmd __nonexistent_command_xyz__
    [ "$status" -ne 0 ]
    [[ "$output" == *"not found"* ]]
}

@test "require_var succeeds when variable is set" {
    export MY_TEST_VAR="hello"
    run require_var MY_TEST_VAR
    [ "$status" -eq 0 ]
}

@test "require_var fails when variable is unset" {
    unset MY_MISSING_VAR 2>/dev/null || true
    run require_var MY_MISSING_VAR
    [ "$status" -ne 0 ]
    [[ "$output" == *"not set"* ]]
}

@test "log_info outputs to stdout" {
    run log_info "test message"
    [ "$status" -eq 0 ]
    [[ "$output" == *"INFO"* ]]
    [[ "$output" == *"test message"* ]]
}
