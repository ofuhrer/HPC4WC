use ndarray::prelude::*;
// use rayon::prelude::*;
// use std::io;
// use std::io::Write;

pub fn update_halo(in_field: &mut Array3<f32>, nx: usize, ny: usize, nz: usize, num_halo: usize) {
    // update bottom edge (without corners)
    for k in 0..nz {
        for j in 0..num_halo {
            for i in (num_halo)..(nx + num_halo) {
                in_field[[i, j, k]] = in_field[[i, j + ny, k]];
            }
        }
    }

    // update top edge (without corners)
    for k in 0..nz {
        for j in (ny + num_halo)..(ny + 2 * num_halo) {
            for i in (num_halo)..(num_halo + nx) {
                in_field[[i, j, k]] = in_field[[i, j - ny, k]];
            }
        }
    }

    // update left edge (including corners)
    for k in 0..nz {
        for j in 0..(ny + 2 * num_halo) {
            for i in 0..num_halo {
                in_field[[i, j, k]] = in_field[[i + nx, j, k]];
            }
        }
    }

    // update right edge (including corners)
    for k in 0..nz {
        for j in 0..(ny + 2 * num_halo) {
            for i in (nx + num_halo)..(nx + 2 * num_halo) {
                in_field[[i, j, k]] = in_field[[i - nx, j, k]];
            }
        }
    }
}

/// Integrate 4th-order diffusion equation by a certain number of iterations.
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
    // println!("running naive version.");
    let mut tmp_field_2d = Array2::<f32>::zeros((nx + 2 * num_halo, ny + 2 * num_halo).f());

    for iter in 0..num_iter {
        // print progress every num_iter/x iterations. REMOVE for benchmarking
        // if num_iter > 10 && iter % ((num_iter / 50) as usize) == 0 {
        //     print!("."); // println!("{iter}");
        //     io::stdout().flush().unwrap()
        // }
        update_halo(in_field, nx, ny, nz, num_halo);

        // laplacians
        for k in 0..nz {
            for j in (num_halo - 1)..(ny + num_halo + 1) {
                for i in (num_halo - 1)..(nx + num_halo + 1) {
                    tmp_field_2d[[i, j]] = -4.0f32 * in_field[[i, j, k]]
                        + in_field[[i - 1, j, k]]
                        + in_field[[i + 1, j, k]]
                        + in_field[[i, j - 1, k]]
                        + in_field[[i, j + 1, k]];
                }
            }

            for j in num_halo..(ny + num_halo) {
                for i in num_halo..(nx + num_halo) {
                    let laplap: f32 = -4.0f32 * tmp_field_2d[[i, j]]
                        + tmp_field_2d[[i - 1, j]]
                        + tmp_field_2d[[i + 1, j]]
                        + tmp_field_2d[[i, j - 1]]
                        + tmp_field_2d[[i, j + 1]];
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

/// Unit tests for the sencil functions
#[cfg(test)]
mod unit_tests {

    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::super::helper_functions;
    use super::*; // cannot see the functions to be tested otherwise
    use std::env;

    // constants for testing, used in all tests so all tests work on the same size
    const NX: usize = 8;
    const NY: usize = 16;
    const NZ: usize = 16;
    const NUM_HALO: usize = 2;

    #[test]
    /// This test checks if the halo update adds or changes any values when it should not.
    fn update_halo_no_new_values() {
        let mut in_field = Array3::<f32>::zeros((NX + 2 * NUM_HALO, NY + 2 * NUM_HALO, NZ));
        let mut zero_field = Array3::<f32>::zeros((NX + 2 * NUM_HALO, NY + 2 * NUM_HALO, NZ));

        helper_functions::initialize_in_field(&mut in_field, NX, NY, NZ, NUM_HALO);
        helper_functions::initialize_in_field(&mut zero_field, NX, NY, NZ, NUM_HALO);

        update_halo(&mut in_field, NX, NY, NZ, NUM_HALO);

        assert_eq!(in_field.shape(), zero_field.shape());

        // check that update_halo does not introduce any new values
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
    fn update_halo_correct_update_manually() {
        let mut in_field = Array3::<f32>::zeros((NX + 2 * NUM_HALO, NY + 2 * NUM_HALO, NZ));
        let mut init_field = Array3::<f32>::zeros((NX + 2 * NUM_HALO, NY + 2 * NUM_HALO, NZ));

        helper_functions::initialize_in_field_test(&mut in_field, NX, NY, NZ, NUM_HALO);
        helper_functions::initialize_in_field_test(&mut init_field, NX, NY, NZ, NUM_HALO);

        update_halo(&mut in_field, NX, NY, NZ, NUM_HALO);

        assert_eq!(in_field.shape(), init_field.shape());

        // Uncomment if you want to check manually:
        // let path1 = "./data/tests/uhc_in_field.csv";
        // match helper_functions::save_slice_to_csv(path1, in_field.view()) {
        //     Ok(()) => println!("Center slice saved to {}", path1),
        //     Err(err) => eprintln!("Error: {}", err),
        // }

        // let path2 = "./data/tests/uhc_init_field.csv";
        // match helper_functions::save_slice_to_csv(path2, init_field.view()) {
        //     Ok(()) => println!("Center slice saved to {}", path2),
        //     Err(err) => eprintln!("Error: {}", err),
        // }
    }

    #[test]
    fn update_halo_correct_update() {
        env::set_var("RUST_BACKTRACE", "1");
        let nx: usize = 2;
        let ny: usize = 2;
        let nz: usize = 1;
        let num_halo: usize = 1;
        let size = (nx + 2 * num_halo) * (ny + 2 * num_halo) * nz;

        let mut in_field: Array3<f32> = Array3::from_shape_vec(
            (nx + 2 * num_halo, ny + 2 * num_halo, nz),
            (1..=size).map(|x| x as f32).collect::<Vec<f32>>(),
        )
        .unwrap();
        let ground_truth: Array3<f32> = Array3::from_shape_vec(
            (4, 4, 1),
            vec![11., 10., 11., 10., 7., 6., 7., 6., 11., 10., 11., 10., 7., 6., 7., 6.],
        )
        .unwrap();

        update_halo(&mut in_field, nx, ny, nz, num_halo);

        // let path2 = "./data/tests/ina.csv";
        // match helper_functions::save_slice_to_csv(path2, in_field.view()) {
        //     Ok(()) => println!("Center slice saved to {}", path2),
        //     Err(err) => eprintln!("Error: {}", err),
        // }
        // let path2 = "./data/tests/out.csv";
        // match helper_functions::save_slice_to_csv(path2, out_field.view()) {
        //     Ok(()) => println!("Center slice saved to {}", path2),
        //     Err(err) => eprintln!("Error: {}", err),
        // }

        for k in 0..in_field.shape()[2] {
            for j in 0..in_field.shape()[1] {
                for i in 0..in_field.shape()[0] {
                    assert!(helper_functions::almost_equal(
                        in_field[[i, j, k]],
                        ground_truth[[i, j, k]]
                    ));
                }
            }
        }
    }

    #[test]
    fn correct_apply_diffusion() {
        let nx: usize = 4;
        let ny: usize = 4;
        let nz: usize = 1;
        let num_halo: usize = 2;

        #[rustfmt::skip]
        let mut in_field: Array3<f32> = Array3::from_shape_vec(
            (nx + 2 * num_halo, nx + 2 * num_halo, nz),
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
        )
        .unwrap();
        let mut out_field = in_field.clone();

        #[rustfmt::skip]
        let ground_truth: Array3<f32> = Array3::from_shape_vec(
            (nx + 2 * num_halo, ny + 2 * num_halo, nz),
            vec![0.94, 0.04, 0.04, 0.94, 0.94, 0.04, 0.04, 0.94,
                    0.04, -0.02, -0.02, 0.04, 0.04, -0.02, -0.02, 0.04,
                    0.04, -0.02, -0.02, 0.04, 0.04, -0.02, -0.02, 0.04,
                    0.94, 0.04, 0.04, 0.94, 0.94, 0.04, 0.04, 0.94,
                    0.94, 0.04, 0.04, 0.94, 0.94, 0.04, 0.04, 0.94,
                    0.04, -0.02, -0.02, 0.04, 0.04, -0.02, -0.02, 0.04,
                    0.04, -0.02, -0.02, 0.04, 0.04, -0.02, -0.02, 0.04,
                    0.94, 0.04, 0.04, 0.94, 0.94, 0.04, 0.04, 0.94, ],
        )
        .unwrap();

        // let path2 = "./data/tests/in_ad.csv";
        // match helper_functions::save_slice_to_csv(path2, in_field.view()) {
        //     Ok(()) => println!("Center slice saved to {}", path2),
        //     Err(err) => eprintln!("Error: {}", err),
        // }

        apply_diffusion(&mut in_field, &mut out_field, nx, ny, nz, num_halo, 1, 0.01);
        update_halo(&mut out_field, nx, ny, nz, num_halo);

        // let path2 = "./data/tests/out_ad.csv";
        // match helper_functions::save_slice_to_csv(path2, out_field.view()) {
        //     Ok(()) => println!("Center slice saved to {}", path2),
        //     Err(err) => eprintln!("Error: {}", err),
        // }
        // let path2 = "./data/tests/in_ad_ad.csv";
        // match helper_functions::save_slice_to_csv(path2, ground_truth.view()) {
        //     Ok(()) => println!("Center slice saved to {}", path2),
        //     Err(err) => eprintln!("Error: {}", err),
        // }

        for k in 0..in_field.shape()[2] {
            for j in 0..in_field.shape()[1] {
                for i in 0..in_field.shape()[0] {
                    assert!(helper_functions::almost_equal(
                        out_field[[i, j, k]],
                        ground_truth[[i, j, k]]
                    ));
                }
            }
        }
    }
}
