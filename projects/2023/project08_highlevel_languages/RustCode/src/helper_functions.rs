use csv::Writer;
use ndarray::prelude::*;
// use rayon::prelude::*;
use std::error::Error;
use std::fs::File;
// use std::io;
// use std::io::Write;

/// fills center cube with val=1.0
pub fn initialize_in_field(field: &mut Array3<f32>, nx: usize, ny: usize, nz: usize, num_halo: usize) -> () {
    let val: f32 = 1.0;
    field.fill(0.);
    for k in (nz / 4)..(3 * nz / 4) {
        for j in (num_halo + ny / 4)..(num_halo + 3 * ny / 4) {
            for i in (num_halo + nx / 4)..(num_halo + 3 * nx / 4) {
                field[[i, j, k]] = val;
            }
        }
    }
}

/// Good function to for testing: Try non-symmetric initializations to check if all dims are working
/// correctly. This function is not used in apply_diffusion.
pub fn initialize_in_field_test(field: &mut Array3<f32>, nx: usize, ny: usize, nz: usize, num_halo: usize) -> () {
    // // ====== fill central region with val ======
    // for k in (nz / 4)..(3 * nz / 4) {
    //     for j in (num_halo + ny / 4)..(num_halo + 3 * ny / 4) {
    //         for i in (num_halo + nx / 4)..(num_halo + 3 * nx / 4) {
    //             field[[i, j, k]] = 1.0 as f32;
    //         }
    //     }
    // }

    // ====== fill border region with val =======
    // for k in 0..nz {
    //     for j in 0..ny {
    //         for i in 0..(nx / 5) {
    //             field[[i, j, k]] = val;
    //         }
    //     }
    // }

    // ====== fill column wise ==================
    // i.e. fills everything except halo regions with the x resp i value it would have if there were no halo region
    for k in 0..(nz) {
        for j in num_halo..(ny + num_halo) {
            for i in num_halo..(nx + num_halo) {
                field[[i, j, k]] = (i - num_halo) as f32;
            }
        }
    }
}

/// Given a 3D array of size nx,ny,nz this saves the nx*ny 2D array at height nz/2 to
/// csv. No idea what happens when nz/2 is not integer so better watch out
/// Also: I think file_path has to have path relative to cargo.toml and file extension
pub fn save_slice_to_csv(file_path: &str, array: ArrayView3<f32>) -> Result<(), Box<dyn Error>> {
    let nz = array.shape()[2];
    let center_slice = array.index_axis(ndarray::Axis(2), nz / 2);

    // Transpose the middle slice: this is because we use (row, col, depth) as (x,y,z)

    let center_slice_t = center_slice.t();

    // Open the CSV file for writing
    let file = File::create(file_path)?;
    let mut wtr = Writer::from_writer(file);

    // Write the slice data to the CSV file
    for row in center_slice_t.rows() {
        let row_str: Vec<String> = row.iter().map(|&val| val.to_string()).collect();
        wtr.write_record(&row_str)?;
    }

    // Flush the writer to ensure all data is written
    wtr.flush()?;

    Ok(())
}

/// returns stride lengths of axis 0,1,2 i.e. rows, cols, depth
/// which in our case is x,y,z
pub fn get_stride_lengths(array: &Array3<f32>) -> [isize; 3] {
    let strides = array.strides();
    // println!(
    //     "0: {}, 1: {}, 2: {}",
    //     array.stride_of(ndarray::Axis(0)),
    //     array.stride_of(ndarray::Axis(1)),
    //     array.stride_of(ndarray::Axis(2))
    // );
    [strides[0], strides[1], strides[2]]
}

/// helper function to test for float equality
#[inline(always)]
pub fn almost_equal(a: f32, b: f32) -> bool {
    let epsilon: f32 = 1e-9;
    (a - b).abs() < epsilon
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    // constants for testing, used in all tests so all tests work on the same size
    const NX: usize = 8;
    const NY: usize = 16;
    const NZ: usize = 16;
    const NUM_HALO: usize = 2;

    #[test]
    /// This test checks if the fields are initialized correctly.
    fn initialize_field() {
        let mut in_field = Array3::<f32>::zeros((NX + 2 * NUM_HALO, NY + 2 * NUM_HALO, NZ));
        let mut zero_field = Array3::<f32>::zeros((NX + 2 * NUM_HALO, NY + 2 * NUM_HALO, NZ));

        initialize_in_field_test(&mut in_field, NX, NY, NZ, NUM_HALO);
        initialize_in_field_test(&mut zero_field, NX, NY, NZ, NUM_HALO);

        // let path1 = "./data/tests/tif_in_field.csv";
        // match save_slice_to_csv(path1, in_field.view()) {
        //     Ok(()) => println!("Center slice saved to {}", path1),
        //     Err(err) => eprintln!("Error: {}", err),
        // }

        // let path2 = "./data/tests/tif_zero_field.csv";
        // match save_slice_to_csv(path2, zero_field.view()) {
        //     Ok(()) => println!("Center slice saved to {}", path2),
        //     Err(err) => eprintln!("Error: {}", err),
        // }

    }

    #[test]
    fn strides() {
        // cargo test strides -- --nocapture
        // nocapture is to print output to cmd line, by default test output is supressed
        let mut a = Array3::<f32>::zeros((NX + 2 * NUM_HALO, NY + 2 * NUM_HALO, NZ));
        a[[5, 10, 8]] = 100 as f32; // 8 is NZ/2 i.e. the center slice wrt to height
        a[[6, 10, 8]] = 200 as f32;
        let path2 = "./data/strides.csv";
        match save_slice_to_csv(path2, a.view()) {
            Ok(()) => println!("Center slice saved to {}", path2),
            Err(err) => eprintln!("Error: {}", err),
        }
        let stride_lengths = get_stride_lengths(&a);
        println!("Stride lengths: {:?}", stride_lengths);

        // Get a raw pointer to the first element of the array
        let ptr = a.as_ptr();

        for x in 0..(NX + 2 * NUM_HALO) {
            let offset: isize = x as isize * a.strides()[0] + 10 * a.strides()[1] + 8 * a.strides()[2];
            let element_address = unsafe { ptr.add(offset as usize) };
            unsafe { println!("{}", *element_address) };
        }

        // // Calculate the offset based on indices (0, 0, 0)
        // let offset: isize = 5 * a.strides()[0] + 10 * a.strides()[1] + 8 * a.strides()[2];
        // let offset2: isize = 6 * a.strides()[0] + 10 * a.strides()[1] + 8 * a.strides()[2];
        // // Calculate the memory address of the element at (0, 0, 0)
        // let element_address = unsafe { ptr.add(offset as usize) };
        // let element_address2 = unsafe { ptr.add(offset2 as usize) };
        // // Print the memory address as a hexadecimal value
        // println!("{:?}", element_address);
        // println!("{:?}", element_address2);
    }
}
