use ndarray::prelude::*;
// use rayon::prelude::*;
use std::io;
use std::io::Write;

pub fn update_halo(in_field: &mut Array3<f32>, nx: usize, ny: usize, nz: usize, num_halo: usize) {
    assert!(nx % 4 == 0);
    assert!(num_halo % 2 == 0);
    // bottom edge (without corners)
    for k in 0..nz {
        for j in 0..num_halo {
            for i in ((num_halo)..(nx + num_halo)).step_by(4) {
                in_field[[i, j, k]] = in_field[[i, j + ny, k]];
                in_field[[i + 1, j, k]] = in_field[[i + 1, j + ny, k]];
                in_field[[i + 2, j, k]] = in_field[[i + 2, j + ny, k]];
                in_field[[i + 3, j, k]] = in_field[[i + 3, j + ny, k]];
            }
        }
    }

    // top edge (without corners)
    for k in 0..nz {
        for j in (ny + num_halo)..(ny + 2 * num_halo) {
            for i in (num_halo..(num_halo + nx)).step_by(4) {
                in_field[[i, j, k]] = in_field[[i, j - ny, k]];
                in_field[[i + 1, j, k]] = in_field[[i + 1, j - ny, k]];
                in_field[[i + 2, j, k]] = in_field[[i + 2, j - ny, k]];
                in_field[[i + 3, j, k]] = in_field[[i + 3, j - ny, k]];
            }
        }
    }

    // left edge (including corners)
    for k in 0..nz {
        for j in 0..(ny + 2 * num_halo) {
            for i in (0..num_halo).step_by(2) {
                in_field[[i, j, k]] = in_field[[i + nx, j, k]];
                in_field[[i + 1, j, k]] = in_field[[i + 1 + nx, j, k]];
            }
        }
    }

    // right edge (including corners)
    for k in 0..nz {
        for j in 0..(ny + 2 * num_halo) {
            for i in (nx + num_halo..(nx + 2 * num_halo)).step_by(2) {
                in_field[[i, j, k]] = in_field[[i - nx, j, k]];
                in_field[[i + 1, j, k]] = in_field[[i + 1 - nx, j, k]];
            }
        }
    }
}

pub fn apply_diffusion(
    in_field: &mut Array3<f32>,
    out_field: &mut Array3<f32>,
    nx: usize,
    ny: usize,
    nz: usize,
    num_halo: usize,
    num_iter: usize,
    alpha: f32,
) { 
    println!("running unrolled version.");
    let mut tmp1_field = Array2::<f32>::zeros((nx + 2 * num_halo, ny + 2 * num_halo).f());

    for iter in 0..num_iter {
        // print progress every num_iter/x iterations. REMOVE for benchmarking
        if num_iter > 10 && iter % ((num_iter / 50) as usize) == 0 {
            print!("."); // println!("{iter}");
            io::stdout().flush().unwrap()
        }
        update_halo(in_field, nx, ny, nz, num_halo);
        // laplacian(in_field, out_field, nx, ny, nz, num_halo, alpha);
        // laplacian(in_field, out_field, nx, ny, nz, num_halo, alpha);

        // laplacians
        for k in 0..nz {
            for j in (num_halo - 1)..(ny + num_halo + 1) {
                for i in (num_halo - 1)..(nx + num_halo + 1) {
                    tmp1_field[[i, j]] = -4.0f32 * in_field[[i, j, k]]
                        + in_field[[i - 1, j, k]]
                        + in_field[[i + 1, j, k]]
                        + in_field[[i, j - 1, k]]
                        + in_field[[i, j + 1, k]];
                }
            }

            for j in num_halo..(ny + num_halo) {
                for i in num_halo..(nx + num_halo) {
                    let laplap: f32 = -4.0f32 * tmp1_field[[i, j]]
                        + tmp1_field[[i - 1, j]]
                        + tmp1_field[[i + 1, j]]
                        + tmp1_field[[i, j - 1]]
                        + tmp1_field[[i, j + 1]];
                    if iter != num_iter - 1 {
                        in_field[[i, j, k]] = in_field[[i, j, k]] - alpha * laplap;
                    } else {
                        out_field[[i, j, k]] = in_field[[i, j, k]] - alpha * laplap;
                    }
                }
            }
        }
    }
}

/// Unit tests for the sencil functions and involved helper functions.
#[cfg(test)]
mod stencil2d_naive_tests {

    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::super::helper_functions;
    use super::*; // cannot see the functions to be tested otherwise

    // constants for testing, used in all tests so all tests work on the same size
    const NX: usize = 8;
    const NY: usize = 16;
    const NZ: usize = 16;
    const NUM_HALO: usize = 2;

    #[test]
    /// This test checks if the halo update adds or changes any values when it should not.
    fn test_update_halo_no_new_values() {
        let mut in_field = Array3::<f32>::zeros((NX + 2 * NUM_HALO, NY + 2 * NUM_HALO, NZ));
        let mut zero_field = Array3::<f32>::zeros((NX + 2 * NUM_HALO, NY + 2 * NUM_HALO, NZ));

        helper_functions::initialize_in_field(&mut in_field, NX, NY, NZ, NUM_HALO);
        helper_functions::initialize_in_field(&mut zero_field, NX, NY, NZ, NUM_HALO);

        update_halo(&mut in_field, NX, NY, NZ, NUM_HALO);

        assert_eq!(in_field.shape(), zero_field.shape());

        // CHECK that update_halo does not introduce any new values
        for k in 0..in_field.shape()[2] {
            for j in 0..in_field.shape()[1] {
                for i in 0..in_field.shape()[0] {
                    assert!(helper_functions::almost_equal(
                        in_field[[i, j, k]],
                        zero_field[[i, j, k]]
                    ));
                }
            }
        }
    }

    #[test]
    /// This test checks if the halo update happens correctly. Currently this has no automated chesk implemented,
    /// you need to check the .csv and plots yourself.
    fn test_update_halo_correct_update() {
        let mut in_field = Array3::<f32>::zeros((NX + 2 * NUM_HALO, NY + 2 * NUM_HALO, NZ));
        let mut init_field = Array3::<f32>::zeros((NX + 2 * NUM_HALO, NY + 2 * NUM_HALO, NZ));

        helper_functions::initialize_in_field_test(&mut in_field, NX, NY, NZ, NUM_HALO);
        helper_functions::initialize_in_field_test(&mut init_field, NX, NY, NZ, NUM_HALO);

        update_halo(&mut in_field, NX, NY, NZ, NUM_HALO);

        assert_eq!(in_field.shape(), init_field.shape());

        // Uncomment if you want to check manually:
        let path1 = "./data/uhc_in_field.csv";
        match helper_functions::save_slice_to_csv(path1, in_field.view()) {
            Ok(()) => println!("Center slice saved to {}", path1),
            Err(err) => eprintln!("Error: {}", err),
        }

        let path2 = "./data/uhc_init_field.csv";
        match helper_functions::save_slice_to_csv(path2, init_field.view()) {
            Ok(()) => println!("Center slice saved to {}", path2),
            Err(err) => eprintln!("Error: {}", err),
        }
    }
}
