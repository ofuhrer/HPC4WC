use csv::ReaderBuilder;
// use csv::Reader;
use std::error::Error;
// use std::io::Result; // why does this break everything?


// TODO: if universal input ads ydim, try to add it here as well
pub fn load_input(dims: &mut Vec<(usize, usize, usize, f32, usize)>) -> Result<(), Box<dyn Error>> {
    // Open the CSV file
    let path = std::path::Path::new("../universal_input/input_dimensions.csv");
    let file = std::fs::File::open(&path)?;

    // if fn is: load_input(file_path: &str)
    // let file = std::fs::File::open(file_path)?;

    // Create a CSV reader with flexible options
    let mut rdr = ReaderBuilder::new().flexible(true).from_reader(file);

    // Iterate over the CSV records and collect the pairs into a vector
    // let mut pairs = Vec::new();
    for result in rdr.records() {
        let record = result?;
        // Assuming the CSV file always has three columns (x_dim, y_dim, z_dim)
        let x_dim: usize = record[0].parse()?;
        let y_dim: usize = record[1].parse()?;
        let z_dim: usize = record[2].parse()?;
        let alpha: f32 = record[3].parse()?;
        let num_iter: usize = record[4].parse()?;

        // Add the pair (x_dim, y_dim) to the vector
        dims.push((x_dim, y_dim, z_dim, alpha, num_iter));
    }

    Ok(())
}