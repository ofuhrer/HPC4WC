#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <omp.h>

#ifdef CRAYPAT
#include "pat_api.h"
#endif
#include "utils_Burger.h"

// Specify the precision here before compiling!
// Possible choices: long double (128 bit), double (64 bit), float (32 bit), __fp16 (16 bit,
// not working properly on personal machine).

// TODO: Specify the precision here before compiling: long double; double; float
using precision_t = long double;

// TODO: Specify the precision here before compiling:
// long double 	-> longdouble
// double 		-> double
// float 		-> single
// __fp16 		-> half
std:: string precision_str = "longdouble";

namespace {

// ===========================================================================
// Function to update halo regions and implement periodic boundary conditions
// ===========================================================================
void updateHalo(Storage3D<precision_t> &inField) {
    const int xInterior = inField.xMax() - inField.xMin();
    const int yInterior = inField.yMax() - inField.yMin();


    for (std::size_t k = 0; k < inField.zMax(); ++k) {
        // Bottom edge (without corners)
        for (std::size_t j = 0; j < inField.yMin(); ++j) {
            for (std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
                inField(i, j, k) = inField(i, j + yInterior, k);
            }
        }

        // Top edge (without corners)
        for (std::size_t j = inField.yMax(); j < inField.ySize(); ++j) {
            for (std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
                inField(i, j, k) = inField(i, j - yInterior, k);
            }
        }

        // Left edge (including corners)
        for (std::size_t j = 0; j < inField.ySize(); ++j) {
            for (std::size_t i = 0; i < inField.xMin(); ++i) {
                inField(i, j, k) = inField(i + xInterior, j, k);
            }
        }

        // Right edge (including corners)
        for (std::size_t j = 0; j < inField.ySize(); ++j) {
            for (std::size_t i = inField.xMax(); i < inField.xSize(); ++i) {
                inField(i, j, k) = inField(i - xInterior, j, k);
            }
        }
    }
}


// ===========================================================================
// Function to compute the upwind derivative based on sign of advection velocity
// ===========================================================================
precision_t upwindDerivative(precision_t adv_vel, precision_t u_center, precision_t u_left, precision_t u_right, precision_t dx) {
    if (adv_vel > 0){
        return (u_center - u_left) / dx;
    } else {
        return (u_right - u_center) / dx;
    }
}


// ===========================================================================
// Function to solve Burger's equation on horizontal slices using upwind scheme.
// ===========================================================================
void solve_Burger(Storage3D<precision_t> &u_inField, Storage3D<precision_t> &u_outField, 
                  Storage3D<precision_t> &v_inField, Storage3D<precision_t> &v_outField,
                     precision_t ni, precision_t dx, precision_t dt, unsigned numIter, int x,
                     int y, int z, int halo, const std::vector<precision_t> &save_times) {


  precision_t current_time = 0.0; 		// Initialize simulation time
  std::size_t save_index = 0; 			// Index to track the next save time

  // Opening output files for u and v fields
  std::string out_folder_path = "output_data/";
  std::ostringstream u_out_filename, v_out_filename;

  u_out_filename << out_folder_path << "u_out_field_nx_" << x << "_ny_" << y << "_nz_" << z << "_" << precision_str << ".dat";
  std::ofstream u_final(u_out_filename.str(), std::ios::binary | std::ios::app);
  if (!u_final.is_open()) {
      std::cerr << "Error: Could not open file '" << u_out_filename.str() << "' for writing." << std::endl;
      return;
  }
  v_out_filename << out_folder_path << "v_out_field_nx_" << x << "_ny_" << y << "_nz_" << z << "_" << precision_str << ".dat";
  std::ofstream v_final(v_out_filename.str(), std::ios::binary | std::ios::app);
  if (!v_final.is_open()) {
      std::cerr << "Error: Could not open file '" << v_out_filename.str() << "' for writing." << std::endl;
      return;
  }
    
    // Main loop for solving Burger's equation
    for (std::size_t iter = 0; iter < numIter; ++iter) {

      precision_t du_dx, dv_dx, du_dy, dv_dy, laplacian_u, laplacian_v;		// Declaing variables
      
      // Number of threads not fixed. Suggest putting OMP_NUM_THREADS=xx when executing the code (more flexible).
      #pragma omp parallel for collapse(2) default(none) private(du_dx, dv_dx, du_dy, dv_dy, laplacian_u, laplacian_v) shared(u_inField, v_inField, u_outField, v_outField, dt, dx, ni)
      for (std::size_t k = 0; k < u_inField.zMax(); ++k) {
        for (std::size_t j = u_inField.yMin(); j < u_inField.yMax(); ++j) {
          for (std::size_t i = u_inField.xMin(); i < u_inField.xMax(); ++i) {
            
            // Define local variables for u_inField and neighbors (to improve performance)
            precision_t u_center = u_inField(i, j, k);
            precision_t u_left = u_inField(i - 1, j, k);
            precision_t u_right = u_inField(i + 1, j, k);
            precision_t u_top = u_inField(i, j + 1, k);
            precision_t u_bottom = u_inField(i, j - 1, k);

            precision_t v_center = v_inField(i, j, k);
            precision_t v_left = v_inField(i - 1, j, k);
            precision_t v_right = v_inField(i + 1, j, k);
            precision_t v_top = v_inField(i, j + 1, k);
            precision_t v_bottom = v_inField(i, j - 1, k);

            // Compute derivatives with finite differences
            du_dx = upwindDerivative(u_center, u_center, u_left, u_right, dx);
            dv_dx = upwindDerivative(u_center, v_center, v_left, v_right, dx);
            du_dy = upwindDerivative(v_center, u_center, u_bottom, u_top, dx);
            dv_dy = upwindDerivative(v_center, v_center, v_bottom, v_top, dx);

            // Calculate the Laplacian for u and v
            laplacian_u = (u_left + u_right + u_top + u_bottom - 4 * u_center) / (dx * dx);
            laplacian_v = (v_left + v_right + v_top + v_bottom - 4 * v_center) / (dx * dx);

            // Update the u field
            u_outField(i, j, k) = u_center - dt * (u_center * du_dx + v_center * du_dy) + dt * ni * laplacian_u;

            // Update the v field
            v_outField(i, j, k) = v_center - dt * (u_center * dv_dx + v_center * dv_dy) + dt * ni * laplacian_v;
            }      
          }
        }

      current_time += dt;	 // time step

      // Save snapshots at fixed physical times
      while (save_index < save_times.size() && current_time >= save_times[save_index]) {
          u_outField.writeFile(u_final);
          v_outField.writeFile(v_final);
          save_index++;
      }

      // Swap input and output for next iteration
      if (iter != numIter - 1) {
          u_inField = u_outField;
          v_inField = v_outField;
          }

	  // Update Halo
      updateHalo(u_inField);
      updateHalo(v_inField);

    }

  // Close files
  u_final.close();
  v_final.close();

  // Print final message
  std::cout << "Simulation completed. Total iterations: " << numIter << std::endl;
}


// ===========================================================================
// Function to report time taken for computation
// ===========================================================================
void reportTime(const Storage3D<precision_t> &storage, int nIter, double diff) {
  std::cout << "# ranks nx ny nz num_iter time\ndata = np.array( [ \\\n";
  int size = 1.0;
#pragma omp parallel
  {
#pragma omp master
    { size = omp_get_num_threads(); }
  }
  std::cout << "[ " << size << ", " << storage.xMax() - storage.xMin() << ", "
            << storage.yMax() - storage.yMin() << ", " << storage.zMax() << ", "
            << nIter << ", " << diff << "],\n";
  std::cout << "] )" << std::endl;
}
} // namespace

int main(int argc, char const *argv[]) {
#ifdef CRAYPAT
  PAT_record(PAT_STATE_OFF);
#endif

  // Call signature for compiled program: "./2d_Burger --nx X --ny Y --nz Z --Tend TEND --nums NUMS"
  int x = atoi(argv[2]);                      // x-dimensions
  int y = atoi(argv[4]);                      // y-dimensions
  int z = atoi(argv[6]);                      // z-dimensions
  int T_end = atoi(argv[8]);                  // Duration of the simulation

  const int nHalo = 3;          			  // Size of Halo field outside of domain
  assert(x > 0 && y > 0 && z > 0);			  // Assert that all fields have positive size

  // Parameters for the simulation
  precision_t dx = 1.0 / (x-1);                         		// dx = dy (assuming physical domain size to be 1.0 m in both directions), gives delta space
  precision_t ni = 0.0001;                              		// viscosity; 1e-4 to allow Re=1e4 and resolve the non-linear effects, Re = vel_max / ni
  precision_t vel_max = 1;                            		// Follows from the parameters for Re=1e4
  precision_t dt_adv = 0.4 * dx / vel_max;            		// characteristic diffusion time
  precision_t dt_diff = 0.75 * dx * dx / (4 * ni);    		// characteristic advection time
  precision_t dt = std::min(dt_diff / 10, dt_adv / 10);   	// time step, defined to ensure stability (for diffusion and advection)
  std::size_t num_iter = static_cast<std::size_t>(T_end / dt); 	// Number of iterations based on end time and time step

  // Define the number of snapshots
  unsigned int num_snapshots = atoi(argv[10]); // Number of snapshots to save, i.e. the first num_snapshots timesteps
  std::vector<precision_t> save_times(num_snapshots);

  save_times[0] = 0.0; // Initial time is always 0 (to show initial condition)
  for (std::size_t i = 1; i < num_snapshots; ++i) { // save time steps in regular intervals
      save_times[i] = i * (static_cast<precision_t>(T_end) / (num_snapshots - 1));
  }

  std::string in_folder_path = "input_data/";

  std::ostringstream u_in_filename, v_in_filename;
  u_in_filename << in_folder_path << "u_in_field_nx_" << x << "_ny_" << y << "_nz_" << z <<  "_" << precision_str << ".dat";
  v_in_filename << in_folder_path << "v_in_field_nx_" << x << "_ny_" << y << "_nz_" << z <<  "_" << precision_str << ".dat";

  // Initilize u component
  Storage3D<precision_t> u_input(x, y, z, nHalo);
  Storage3D<precision_t> u_output(x, y, z, nHalo);
  std::ifstream u_initial(u_in_filename.str(), std::ios::binary);
  u_input.readFile(u_initial);
  u_initial.close();
  u_output = u_input;

  // Initilize v component
  Storage3D<precision_t> v_input(x, y, z, nHalo);
  Storage3D<precision_t> v_output(x, y, z, nHalo);
  std::ifstream v_initial(v_in_filename.str(), std::ios::binary);
  v_input.readFile(v_initial);
  v_initial.close();
  v_output = v_input;

  // Perform initial halo updates
  updateHalo(u_output);
  updateHalo(v_output);

#ifdef CRAYPAT
  PAT_record(PAT_STATE_ON);
#endif
  auto start = std::chrono::steady_clock::now();

  solve_Burger(u_input, u_output, v_input, v_output, ni, dx, dt, num_iter, x, y, z, nHalo, save_times);

  auto end = std::chrono::steady_clock::now();
#ifdef CRAYPAT
  PAT_record(PAT_STATE_OFF);
#endif

  // Report Time Differences
  auto diff = end - start;
  double timeDiff =
      std::chrono::duration<double, std::milli>(diff).count() / 1000.;
  reportTime(u_output, num_iter, timeDiff);

  return 0;
}