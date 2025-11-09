#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <ostream>
#include <vector>
#include "utils_Burger.h"


// Specify the precision here before compiling!
// Possible choices: long double (128 bit), double (64 bit), float (32 bit), __fp16 (16 bit,
// not working properly on personal machine).

// TODO: Specify the precision here before compiling: long double; double; float
using precision_t = float;

// TODO: Specify the precision here before compiling:
// long double 	-> longdouble
// double 		-> double
// float 		-> single
// __fp16 		-> half
std:: string precision_str = "single";

// ===========================================================================
// Function to write fields to disk
// ===========================================================================
void saveFields(Storage3D<precision_t> &u_input, Storage3D<precision_t> &v_input, int x, int y, int z) {

  // Opening output files for u and v fields
  // Requires '.../input_data' to exist
  std::string in_folder_path = "input_data/";
  std::ostringstream u_in_filename, v_in_filename;

  u_in_filename << in_folder_path << "u_in_field_nx_" << x << "_ny_" << y << "_nz_" << z << "_" << precision_str << ".dat";
  std::ofstream u_initial(u_in_filename.str(), std::ios::binary | std::ios::app);
  if (!u_initial.is_open()) {
      std::cerr << "Error: Could not open file '" << u_in_filename.str() << "' for writing." << std::endl;
      return;
  }
  u_input.writeFile(u_initial);
  u_initial.close();

  v_in_filename << in_folder_path << "v_in_field_nx_" << x << "_ny_" << y << "_nz_" << z << "_" << precision_str << ".dat";
  std::ofstream v_initial(v_in_filename.str(), std::ios::binary | std::ios::app);
  if (!v_initial.is_open()) {
      std::cerr << "Error: Could not open file '" << v_in_filename.str() << "' for writing." << std::endl;
      return;
  }
  v_input.writeFile(v_initial);
  v_initial.close();

  return;
}

int main(int argc, char const *argv[]) {

    // Call signature for compiled program: "./make_initial_fields --nx X --ny Y --nz Z"
    int x = atoi(argv[2]);        // x-dimensions
    int y = atoi(argv[4]);        // y-dimensions
    int z = atoi(argv[6]);        // z-dimensions

    // Halo size
    const int nHalo = 3;

    // Parameters for initialization of fields (initial conditions)
    precision_t u_max = 1;
    int width = y / 1, length = x / 1;

    // Initilize u component on largest field
    Storage3D<precision_t> u_input(x, y, z, nHalo);
    u_input.initialize(u_max, width, length, "u"); 

    // Initilize v component on largest field
    Storage3D<precision_t> v_input(x, y, z, nHalo);
    v_input.initialize(u_max, width, length, "v");

    // Create fields of smaller resolutions
    Storage3D<precision_t> u_input_2(x / 2, y / 2, z, nHalo);
    Storage3D<precision_t> v_input_2(x / 2, y / 2, z, nHalo);
    Storage3D<precision_t> u_input_4(x / 4, y / 4, z, nHalo);
    Storage3D<precision_t> v_input_4(x / 4, y / 4, z, nHalo);
    Storage3D<precision_t> u_input_8(x / 8, y / 8, z, nHalo);
    Storage3D<precision_t> v_input_8(x / 8, y / 8, z, nHalo);

    int j_inner = -1; // Variable to store j_inner value
    int i_inner = -1; // Variable to store i_inner value
    for (int k = 0; k < z; ++k) {
        for (int j = 0; j < u_input.ySize(); ++j) {
            for (int i = 0; i < u_input.xSize(); ++i) {
                // create inner indices for the non-halo region
                if (j < u_input.yMin() || j > u_input.yMax()) j_inner = -1; // Set j_inner in Halo to -1
                else j_inner = j - u_input.yMin();
                if (i < u_input.xMin() || i > u_input.xMax()) i_inner = -1; // Set i_inner in Halo to -1
                else i_inner = i - u_input.xMin();
                
                // Copy values to smaller resolutions in the non-halo region
                // In the halo region, all fields are already initialized to 0
                if (i_inner % 2 == 0 && j_inner % 2 == 0) {
                    u_input_2(u_input_2.xMin() + i_inner / 2, u_input_2.yMin() + j_inner / 2, k) = u_input(i, j, k);
                    v_input_2(v_input_2.xMin() + i_inner / 2, v_input_2.yMin() + j_inner / 2, k) = v_input(i, j, k);
                }
                if (i_inner % 4 == 0 && j_inner % 4 == 0) {
                    u_input_4(u_input_4.xMin() + i_inner / 4, u_input_4.yMin() + j_inner / 4, k) = u_input(i, j, k);
                    v_input_4(v_input_4.xMin() + i_inner / 4, v_input_4.yMin() + j_inner / 4, k) = v_input(i, j, k);
                }
                if (i_inner % 8 == 0 && j_inner % 8 == 0) {
                    u_input_8(u_input_8.xMin() + i_inner / 8, u_input_8.yMin() + j_inner / 8, k) = u_input(i, j, k);
                    v_input_8(v_input_8.xMin() + i_inner / 8, v_input_8.yMin() + j_inner / 8, k) = v_input(i, j, k);
                }
            }
        }   
    }
    // Save fields to files
    saveFields(u_input, v_input, x, y, z);
    saveFields(u_input_2, v_input_2, x / 2, y / 2, z);
    saveFields(u_input_4, v_input_4, x / 4, y / 4, z);
    saveFields(u_input_8, v_input_8, x / 8, y / 8, z);

    return 0;
}