#![allow(nonstandard_style)] // to allow non-snake-case name RustCode
use ndarray::prelude::*;
// use rayon::prelude::*;
use std::env;
use std::time::Instant;

use RustCode::helper_functions;
use RustCode::stencil2d_naive::update_halo as update_halo_naive;

// Different implementations of the stencil computations:
// use RustCode::stencil2d_iterators as stencil;
use RustCode::stencil2d_naive as stencil;
// use RustCode::stencil2d_rayon_iterators as stencil;
// use RustCode::stencil2d_vectorized as stencil;

/// # Diffusion Simulation
/// ### Running the simulation:
/// Use this function to run the simulation from command line, i.e. when you are not performing benchmarks. \
/// The easiest way to run the simulation is to use cargo run and supply cmd line arguments: \
/// ```cargo run -- 64 64 64 100 2``` or ```cargo run --release  -- 64 64 64 100 2``` \
/// The arguments are -- nx ny nz num_iter num_halo, --release builds the optimized (instead of default debug) target. \
/// The different versions of the stencil computation are listed above, simply uncomment the use statement
/// of the version you want to run. \
///
/// ### Plotting:
/// This function will save 2D slices of the 3D array as csv files which then can be plotted using the provided
/// python script ```./data/plots.py.``` \
/// The array slices will be saved to ./data/in_field.csv and ./data/out_field.csv and can both be plotted using: \
/// ```py data/plots.py in_field out_field```.
///
/// ### Benchmarks:
/// The benchmarks in ```./benches/bench.rs``` will time all of the above mentioned stencil implementations and 
/// save the report in ```./target/criterion/report/index.html```. This includes a variety of usefull plots.
/// To extract all means (mean times over some number of repeated benchmarks for a certain stencil implementation),
/// run ```py3 ./data/get_means.py```. For more information see the comments in that file.
fn main() {
    //================================================================================
    // Get input: array dimensions and num_iter
    //================================================================================
    env::set_var("RUST_BACKTRACE", "1");
    let args: Vec<_> = env::args().collect();
    assert!(args.len() == 6); // 6 = file path + five args

    // get all values from cmd line, note they all need to be usize to be used as indices
    let nx: usize = args[1].parse::<usize>().unwrap();
    let ny: usize = args[2].parse::<usize>().unwrap();
    let nz: usize = args[3].parse::<usize>().unwrap();
    let num_iter: usize = args[4].parse::<usize>().unwrap();
    let num_halo: usize = args[5].parse::<usize>().unwrap();
    let alpha = 1.0 / 32.0;

    assert!(0 < nx && nx <= 4096, "Unreasonable value for nx");
    assert!(0 < ny && ny <= 4096, "Unreasonable value for ny");
    assert!(0 < nz && nz <= 1024, "Unreasonable value for nz");
    assert!(0 < num_iter && num_iter <= 4096, "Unreasonable value for num_iter");
    assert!(2 <= num_halo && num_halo <= 256, "Unreasonable number of halo points");

    print!(
        "Running with: nx={}, ny={}, nz={}, num_iter={}, num_halo={}\n",
        nx, ny, nz, num_iter, num_halo
    );

    //================================================================================
    // allocate arrays and initialize all with 0
    //================================================================================
    // Note on the shape of these arrays:
    // The shape generally is (rows, cols, depth) for ndarray. Througout RustCode/ this will be interpreted as
    // (rows, cols, depth) = (nx, ny, nz), which means rows are x, cols are in y. This is manly because of the way the
    // arrays are accessed in the fortran and c codes as well as for the convenience of element access being [[x,y,z]]
    // To get comparable output the arrays are transposed before saving, see helper_functions::save_slice_to_csv
    let mut in_field = Array3::<f32>::zeros((nx + 2 * num_halo, ny + 2 * num_halo, nz).f());
    let mut out_field = Array3::<f32>::zeros((nx + 2 * num_halo, ny + 2 * num_halo, nz).f());

    // output stride lengths to check loop order
    let stride_lengths: [isize; 3] = helper_functions::get_stride_lengths(&in_field);
    println!("Stride lengths: {:?}", stride_lengths);

    // check dimensions/shape of the two arrays and print shape
    assert!(in_field.shape() == out_field.shape());
    println!("in_field shape: {:?}", in_field.shape());

    // fill parts of in_field with some initial values, use _test for testing
    helper_functions::initialize_in_field(&mut in_field, nx, ny, nz, num_halo);

    // save in_field and handle errors if there are any
    // we pass in array.view() as the function only needs a read-only view of the array
    let in_file_path = "./data/in_field.csv";
    match helper_functions::save_slice_to_csv(in_file_path, in_field.view()) {
        Ok(()) => println!("saved to {}", in_file_path),
        Err(err) => eprintln!("Error: {}", err),
    }

    //================================================================================
    // warm up caches
    //================================================================================
    stencil::apply_diffusion(&mut in_field, &mut out_field, nx, ny, nz, num_halo, 1, alpha);
    // s_iter::apply_diffusion(&mut in_field, &mut out_field, nx, ny, nz, num_halo, num_iter, alpha);

    //================================================================================
    // START TIMING: time the calculations
    //================================================================================
    let now = Instant::now();
    stencil::apply_diffusion(&mut in_field, &mut out_field, nx, ny, nz, num_halo, num_iter, alpha);
    let elapsed_time = now.elapsed().as_nanos() as f64;
    println!("\nTook: {} ms", elapsed_time * 1e-6);
    //================================================================================
    // END TIMING: done with time
    //================================================================================

    // update halo around out_field: if not called the halo will remain all zeros as initialized
    // and only the interior (without halo points) will show effect of diffusion
    // if called out_field will display final state of field as is i.e. interior + halo points
    update_halo_naive(&mut out_field, nx, ny, nz, num_halo);
    // save out_field
    let out_file_path = "./data/out_field.csv";
    match helper_functions::save_slice_to_csv(out_file_path, out_field.view()) {
        Ok(()) => println!("saved to {}", out_file_path),
        Err(err) => eprintln!("Error: {}", err),
    }
}

//================================================================================
// NOTE ON SETTING THE NUMBER OF THREADS:
//================================================================================
// setting number of threads not working as expected using thread pool:
//  https://docs.rs/rayon/latest/rayon/struct.ThreadPoolBuilder.html
// https://stackoverflow.com/questions/59205184/how-can-i-change-the-number-of-threads-rayon-uses
//
// fn main() -> Result<(), rayon::ThreadPoolBuildError> {
// // set number of threads for rayon version
// rayon::ThreadPoolBuilder::new().num_threads(22).build_global().unwrap();
//
// // as return
// Ok(())
// }
//
// What does work is setting the environment variable RAYON_NUM_THREADS:
// export RAYON_NUM_THREADS=4
// (and later unset RAYON_NUM_THREADS if you want)
//
