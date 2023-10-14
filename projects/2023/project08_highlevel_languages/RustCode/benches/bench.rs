#![allow(unused_imports)]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::prelude::*;
use std::time::Duration;
use RustCode::helper_functions;
use RustCode::input_loader;
use RustCode::stencil2d_iterators as iterators;
use RustCode::stencil2d_naive as naive;
use RustCode::stencil2d_rayon_iterators as rayiterators;
use RustCode::stencil2d_vectorized as vectorized;

pub fn stencil_benchmarks_criterion(c: &mut Criterion) {
    let mut group = c.benchmark_group("stencil_computations");
    group.measurement_time(Duration::new(10, 0)).sample_size(10);

    let mut dims: Vec<(usize, usize, usize, f32, usize)> = Vec::new();
    if let Err(err) = input_loader::load_input(&mut dims) {
        eprintln!("Error loading input: {}", err);
    }
    println!("There are {:?} problem sizes available. They are:", dims.len());
    println!("idx (nx, ny, nz, alpha, num_iter, num_halo)");
    for (i, d) in dims.iter().enumerate() {
        println!("{}: {:?}", i, d);
    }

    // Iterate through problem sizes. If you only want to do one just run loop from 2..3 (e.g. to run problem size 2)
    // The number of the problem size is just the row (starting from 0) in universal_input/input_dimensions.csv
    // THIS NEEDS TO MATCH: will produce errors if you iterate up to a too large row index
    for i in 0..4 {
        let nx: usize = dims[i].0;
        let ny: usize = dims[i].1;
        let nz: usize = dims[i].2;
        let num_iter: usize = dims[i].4;
        let num_halo: usize = 2;
        let alpha = dims[i].3;
        println!("running size {:?} [nx: {:?}, ny: {:?}, nz: {:?}, alpha: {:?}, num_iter: {:?},  num_halo: {:?}]",
                i, nx, ny, nz, alpha, num_iter, num_halo);

        // ================================================================================
        // naive version
        // ================================================================================
        let mut in_field = Array3::<f32>::zeros((nx + 2 * num_halo, ny + 2 * num_halo, nz).f());
        let mut out_field = Array3::<f32>::zeros((nx + 2 * num_halo, ny + 2 * num_halo, nz).f());
        helper_functions::initialize_in_field(&mut in_field, nx, ny, nz, num_halo);
        group.bench_function(BenchmarkId::new("naive", i), |b| {
            b.iter(|| naive::apply_diffusion(&mut in_field, &mut out_field, nx, ny, nz, num_halo, num_iter, alpha))
        });

        //================================================================================
        // vectorized version
        //================================================================================
        let mut in_field = Array3::<f32>::zeros((nx + 2 * num_halo, ny + 2 * num_halo, nz));
        let mut out_field = Array3::<f32>::zeros((nx + 2 * num_halo, ny + 2 * num_halo, nz));
        helper_functions::initialize_in_field(&mut in_field, nx, ny, nz, num_halo);
        group.bench_function(BenchmarkId::new("vectorized", i), |b| {
            b.iter(|| vectorized::apply_diffusion(&mut in_field, &mut out_field, nx, ny, nz, num_halo, num_iter, alpha))
        });

        //================================================================================
        // iterator version
        //================================================================================
        let mut in_field = Array3::<f32>::zeros((nx + 2 * num_halo, ny + 2 * num_halo, nz).f());
        let mut out_field = Array3::<f32>::zeros((nx + 2 * num_halo, ny + 2 * num_halo, nz).f());
        helper_functions::initialize_in_field(&mut in_field, nx, ny, nz, num_halo);
        group.bench_function(BenchmarkId::new("iterators", i), |b| {
            b.iter(|| iterators::apply_diffusion(&mut in_field, &mut out_field, nx, ny, nz, num_halo, num_iter, alpha))
        });

        //================================================================================
        // parallel iteration version
        //================================================================================
        println!("Rayon iterators with {:?} threads.", rayon::current_num_threads());
        let mut in_field = Array3::<f32>::zeros((nx + 2 * num_halo, ny + 2 * num_halo, nz).f());
        let mut out_field = Array3::<f32>::zeros((nx + 2 * num_halo, ny + 2 * num_halo, nz).f());
        helper_functions::initialize_in_field(&mut in_field, nx, ny, nz, num_halo);
        group.bench_function(BenchmarkId::new("rayiterators", i), |b| {
            b.iter(|| rayiterators::apply_diffusion(&mut in_field, &mut out_field, nx, ny, nz, num_halo, num_iter, alpha))
        });

        //================================================================================
        // views iterator version
        //================================================================================
        let mut in_field = Array3::<f32>::zeros((nx + 2 * num_halo, ny + 2 * num_halo, nz).f());
        let mut out_field = Array3::<f32>::zeros((nx + 2 * num_halo, ny + 2 * num_halo, nz).f());
        helper_functions::initialize_in_field(&mut in_field, nx, ny, nz, num_halo);
        group.bench_function(BenchmarkId::new("view_iterators", i), |b| {
            b.iter(|| iterators::apply_diffusion_view(&mut in_field.view_mut(), &mut out_field.view_mut(), nx, ny, nz, num_halo, num_iter, alpha))
        });

        //================================================================================
        // views parallel iteration version
        //================================================================================
        println!("Rayon iterators with {:?} threads.", rayon::current_num_threads());
        let mut in_field = Array3::<f32>::zeros((nx + 2 * num_halo, ny + 2 * num_halo, nz).f());
        let mut out_field = Array3::<f32>::zeros((nx + 2 * num_halo, ny + 2 * num_halo, nz).f());
        helper_functions::initialize_in_field(&mut in_field, nx, ny, nz, num_halo);
        group.bench_function(BenchmarkId::new("view_rayiterators", i), |b| {
            b.iter(|| rayiterators::apply_diffusion_view(&mut in_field.view_mut(), &mut out_field.view_mut(), nx, ny, nz, num_halo, num_iter, alpha))
        });
    }

    group.finish();
}

criterion_group!(benches, stencil_benchmarks_criterion);
criterion_main!(benches);
