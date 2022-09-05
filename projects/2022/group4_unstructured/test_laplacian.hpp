#ifndef TEST_LAPLACIAN
#define TEST_LAPLACIAN
#include <iostream>
#include <Eigen/Dense>
//#include <cassert>
//#include "pat_api.h"
//#include "utils.h"


typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXd;

//if we want to work with irregular grids we need to store the gridinformation e.g. coordinates in a different matrix
// can we even use second order finite difference for irregular grids e.g. which stepsize to use? 
void compute_laplacian(MatrixXd& in_field, MatrixXd& out_field, const double step_size_x, const double step_size_y)
{
      
      //based on second order centered finite difference
      //due to halo update
      for(std::size_t i = 1; i < in_field.rows()-1; ++i){
	for(std::size_t j = 1; j < in_field.cols()-1; ++j){
		//out_field(i-1,j-1) = (in_field(i,j+1) - 4*in_field(i,j) + in_field(i,j-1) + in_field(i+1,j) + in_field(i-1,j))/step_size;
                 out_field(i,j) = (in_field(i,j+1) - 2*in_field(i,j) + in_field(i,j-1))/step_size_y + (in_field(i+1,j)- 2*in_field(i,j) + in_field(i-1,j))/step_size_x;	

	}

      } 

}

MatrixXd add_halo_points(MatrixXd const& in_field, const unsigned int n_halo, const double dirichlet_value) {

	const std::size_t num_rows = in_field.rows(); 
	const std::size_t num_cols = in_field.cols();

	MatrixXd updated_matrix = MatrixXd::Constant(num_rows+2*n_halo, num_cols+2*n_halo, dirichlet_value);

	updated_matrix.block(1,1, num_rows, num_cols) = in_field;

	return updated_matrix;
}



/*
int main(int argc, char const* argv[]) {

  int x = atoi(argv[2]);
  int y = atoi(argv[4]);
  //int z = atoi(argv[6]);
  int iter = atoi(argv[8]);
  unsigned int halo_points = 1;
  assert(x > 0 && y > 0 && iter > 0);
  const double dirichlet_value = 0.0;
  //Storage3D<double> input(x, y, z, nHalo);
  //input.initialize();
  MatrixXd input(x,y);

  input = add_halo_points(in_field, halo_points, dirichlet_value);
  
  MatrixXd output(x,y); 
  //Storage3D<double> output(x, y, z, nHalo);
  //output.initialize();
  
  //std::ofstream fout;
  //fout.open("in_field.dat", std::ios::binary | std::ofstream::trunc);
  //input.writeFile(fout);
  //fout.close();
  

  compute_laplacian(input, output, step_size);

  //updateHalo(output);
  //fout.open("out_field.dat", std::ios::binary | std::ofstream::trunc);
  //output.writeFile(fout);
  //fout.close();
  

  return 0;
}
*/

#endif /*TEST_LAPLACIAN*/
