#!/bin/zsh

# ================================================================================
# Requirements: 
# Have rust running, cargo installed, python installed
# ================================================================================


# set to desired number of threads, will mainly be use in stencil2d_rayon_iterators.rs
# suppress all warnings for the duration of the benchmarks
# export RUSTFLAGS=-Awarnings

echo "==================== Running rust benchmarks: Look at generated report /target/criterion/report/index.html"
# cargo build
cargo bench

echo "==================== collecting mean execution times into rust_benchmarks.csv"
python ./data/get_estimates.py


# unset RUSTFLAGS
echo "==================== Done. All variables unset."