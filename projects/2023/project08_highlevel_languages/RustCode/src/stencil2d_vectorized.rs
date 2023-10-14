use crate::stencil2d_naive::update_halo as update_halo_naive;
use ndarray::prelude::*;
// use std::io;
// use std::io::Write;

pub fn update_halo(
    field: &mut Array3<f32>,
    nx: usize,
    ny: usize,
    _nz: usize,
    num_halo: usize,
    x_slice: &mut Array3<f32>,
    y_slice: &mut Array3<f32>,
) {
    // Bottom halo edge (without corners)
    // let bottom_slice = field
    //     .slice(s![num_halo..(num_halo + nx), (ny)..(ny + num_halo), ..])
    //     .to_owned();
    x_slice.assign(&field.slice(s![num_halo..(num_halo + nx), (ny)..(ny + num_halo), ..]));
    field
        .slice_mut(s![num_halo..(num_halo + nx), ..num_halo, ..])
        .assign(&(x_slice));

    // // Top halo edge (without corners)
    // let top_slice = field
    //     .slice(s![num_halo..(num_halo + nx), num_halo..(2 * num_halo), ..])
    //     .to_owned();
    x_slice.assign(&field.slice(s![num_halo..(num_halo + nx), num_halo..(2 * num_halo), ..]));
    field
        .slice_mut(s![num_halo..(num_halo + nx), (ny + num_halo).., ..])
        .assign(&x_slice);

    // // Right halo edge (including corners)
    // let right_slice = field.slice(s![num_halo..(2 * num_halo), .., ..]).to_owned();
    y_slice.assign(&field.slice(s![num_halo..(2 * num_halo), .., ..]));
    field.slice_mut(s![(nx + num_halo).., .., ..]).assign(&y_slice);

    // // Left edge (including corners)
    // let left_slice = field.slice(s![nx..(nx + num_halo), .., ..]).to_owned();
    y_slice.assign(&field.slice(s![nx..(nx + num_halo), .., ..]));
    field.slice_mut(s![..num_halo, .., ..]).assign(&y_slice);
}

fn _laplacian(
    in_field: &ArrayView<f32, Ix3>,
    lap_field: &mut Array3<f32>,
    nx: usize,
    ny: usize,
    _nz: usize,
    num_halo: usize,
    extend: usize,
) {
    let sizex = nx + 2 * num_halo;
    let sizey = ny + 2 * num_halo;
    let ib = num_halo - extend;
    let ie = sizex - num_halo + extend;
    let jb = num_halo - extend;
    let je = sizey - num_halo + extend;

    let lap = -4.0f32 * &in_field.slice(s![ib..ie, jb..je, ..])
        + &in_field.slice(s![ib - 1..ie - 1, jb..je, ..])
        + &in_field.slice(s![ib..ie, jb - 1..je - 1, ..])
        + &in_field.slice(s![ib + 1..ie + 1, jb..je, ..])
        + &in_field.slice(s![ib..ie, jb + 1..je + 1, ..]);

    lap_field.slice_mut(s![ib..ie, jb..je, ..]).assign(&lap);
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
    // println!("running slicing version.");
    let mut tmp_field_3d = Array3::<f32>::zeros((nx + 2 * num_halo, ny + 2 * num_halo, nz).f());

    let sizex = nx + 2 * num_halo;
    let sizey = ny + 2 * num_halo;
    let ib1 = num_halo - 1;
    let ie1 = sizex - num_halo + 1;
    let jb1 = num_halo - 1;
    let je1 = sizey - num_halo + 1;
    let ib0 = num_halo;
    let ie0 = sizex - num_halo;
    let jb0 = num_halo;
    let je0 = sizey - num_halo;

    for iter in 0..num_iter {
        // Print progress every num_iter/x iterations. REMOVE for benchmarking
        // if num_iter > 10 && iter % ((num_iter / 50) as usize) == 0 {
        //     print!(".");
        //     io::stdout().flush().unwrap();
        // }
        update_halo_naive(in_field, nx, ny, nz, num_halo);

        // laplacian(&in_field.view(), &mut tmp_field_3d, nx, ny, nz, num_halo, 1);
        tmp_field_3d.slice_mut(s![ib1..ie1, jb1..je1, ..]).assign(
            &(-4.0f32 * &in_field.slice(s![ib1..ie1, jb1..je1, ..])
                + &in_field.slice(s![ib1 - 1..ie1 - 1, jb1..je1, ..])
                + &in_field.slice(s![ib1..ie1, jb1 - 1..je1 - 1, ..])
                + &in_field.slice(s![ib1 + 1..ie1 + 1, jb1..je1, ..])
                + &in_field.slice(s![ib1..ie1, jb1 + 1..je1 + 1, ..])),
        );

        // laplacian(&tmp_field_3d.view(), out_field, nx, ny, nz, num_halo, 0);
        out_field.slice_mut(s![ib0..ie0, jb0..je0, ..]).assign(
            &(-4.0f32 * &tmp_field_3d.slice(s![ib0..ie0, jb0..je0, ..])
                + &tmp_field_3d.slice(s![ib0 - 1..ie0 - 1, jb0..je0, ..])
                + &tmp_field_3d.slice(s![ib0..ie0, jb0 - 1..je0 - 1, ..])
                + &tmp_field_3d.slice(s![ib0 + 1..ie0 + 1, jb0..je0, ..])
                + &tmp_field_3d.slice(s![ib0..ie0, jb0 + 1..je0 + 1, ..])),
        );

        // Update tmp_in_field using broadcasting
        if iter != num_iter - 1 {
            *in_field = &in_field.view() - &(alpha * &out_field.view());
        } else {
            *out_field = &in_field.view() - &(alpha * &out_field.view());
        }
        // ONLY USE TO DEBUG, otherwise poor data folder
        // requires helper_functions
        // let in_file_path = format!("./data/ifie_{}.csv", iter);
        // match helper_functions::save_slice_to_csv(&in_file_path, in_field.view()) {
        //     Ok(()) => println!("saved to {}", in_file_path),
        //     Err(err) => eprintln!("Error: {}", err),
        // }
    }
}

/// Unit tests for the sencil functions and involved helper functions.
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
    fn test_update_halo_no_new_values() {
        let mut in_field = Array3::<f32>::zeros((NX + 2 * NUM_HALO, NY + 2 * NUM_HALO, NZ));
        let mut zero_field = Array3::<f32>::zeros((NX + 2 * NUM_HALO, NY + 2 * NUM_HALO, NZ));

        helper_functions::initialize_in_field(&mut in_field, NX, NY, NZ, NUM_HALO);
        helper_functions::initialize_in_field(&mut zero_field, NX, NY, NZ, NUM_HALO);

        let mut x_slice = Array3::<f32>::zeros((NX, NUM_HALO, NZ).f());
        let mut y_slice = Array3::<f32>::zeros((NUM_HALO, NY + 2 * NUM_HALO, NZ).f());
        update_halo(&mut in_field, NX, NY, NZ, NUM_HALO, &mut x_slice, &mut y_slice);

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

        let mut x_slice = Array3::<f32>::zeros((NX, NUM_HALO, NZ).f());
        let mut y_slice = Array3::<f32>::zeros((NUM_HALO, NY + 2 * NUM_HALO, NZ).f());
        update_halo(&mut in_field, NX, NY, NZ, NUM_HALO, &mut x_slice, &mut y_slice);

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
        let out_field: Array3<f32> = Array3::from_shape_vec(
            (4, 4, 1),
            vec![11., 10., 11., 10., 7., 6., 7., 6., 11., 10., 11., 10., 7., 6., 7., 6.],
        )
        .unwrap();

        let mut x_slice = Array3::<f32>::zeros((nx, num_halo, nz).f());
        let mut y_slice = Array3::<f32>::zeros((num_halo, ny + 2 * num_halo, nz).f());
        update_halo(&mut in_field, nx, ny, nz, num_halo, &mut x_slice, &mut y_slice);

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
                        out_field[[i, j, k]]
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
        let mut x_slice = Array3::<f32>::zeros((nx, num_halo, nz).f());
        let mut y_slice = Array3::<f32>::zeros((num_halo, ny + 2 * num_halo, nz).f());
        update_halo(&mut out_field, nx, ny, nz, num_halo, &mut x_slice, &mut y_slice);

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
