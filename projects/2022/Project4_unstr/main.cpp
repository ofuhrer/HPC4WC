#include <iostream>
#include <memory>
#include <gmsh.h>
#include <fstream>
#include <iomanip>
#include "mesh.hpp"
#include <Eigen/Dense>
#include "interpolation.hpp"
#include "test_laplacian.hpp"
#include <stdio.h> //for sscanf

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXd;

void diffusion(std::shared_ptr<Mesh>& meshptr,const unsigned int& num_iterations, const double& time_step, Eigen::VectorXd& initial_field) 
{
    Eigen::VectorXd out_field(initial_field.size()); 
    for(unsigned int i = 0; i < num_iterations; ++i) {
        meshptr->laplacian(initial_field, out_field);
        initial_field = initial_field + time_step*out_field; 
    }
}

Eigen::MatrixXd diffusion2(std::shared_ptr<Mesh>& meshptr, const unsigned int& num_iterations, const double& time_step, Eigen::VectorXd const& initial_field) 
{
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(initial_field.rows(), num_iterations);
  out.col(0) = initial_field;
  Eigen::VectorXd out_field(initial_field.size()); 
  for(unsigned int i = 1; i < num_iterations; ++i) {
      meshptr->laplacian(out.col(i - 1), out_field);
      out.col(i) = out.col(i - 1) + time_step * out_field; 
  }
        
  return out; 
}

void write_mesh_to_file(std::shared_ptr<Mesh>& mesh_ptr)
{

  std::ofstream myfile; 
  myfile.open("centers.csv");
  myfile << mesh_ptr->get_centers();
  myfile.close();

  myfile.open("nodes.csv");
  myfile << mesh_ptr->get_nodes();
  myfile.close();
 
  myfile.open("lookup_table.csv");
  
  for(unsigned int i = 0; i < mesh_ptr->get_num_elements(); ++i)
  {
    myfile << mesh_ptr->get_all_neighbours(i).transpose(); 
    myfile << "\n";
  }
  myfile.close();
  
  myfile.open("element_vertices.csv");
  /* loop over all the element IDs to write the corresponding vertices in a line each */
  for (int i = 0; i < mesh_ptr->get_num_elements(); i++) {
    myfile << mesh_ptr->get_cellvertices(i); 
    myfile << "\n";
  }
  myfile.close();
	
}

void benchmark(std::shared_ptr<Mesh>& meshptr, const std::size_t nx, const std::size_t ny) 
{

    MatrixXd in_field = MatrixXd::Random(nx,ny);
    MatrixXd out_field = MatrixXd::Zero(in_field.rows(), in_field.cols());
    
    Eigen::VectorXd in_field_unstr(nx*ny);
    Eigen::VectorXd out_field_unstr(in_field_unstr.size());
    std::cout << in_field_unstr.size() << std::endl;

    const unsigned int halo_points = 1;
    const double dirichlet_value = 0.0;
    double lx = 100; 
    double ly = 100; 
    const double step_size_x = lx/nx;  
    const double step_size_y = ly/ny;

    MatrixXd input = add_halo_points(in_field, halo_points, dirichlet_value);
    MatrixXd output_field(in_field.rows(),in_field.cols());

    //compute laplacian using finite differences
    double t1, t2, time_finite = 0, time_gauss = 0;    
    for(unsigned int i = 0; i < 10; ++i) {
       t1 = wall_time();
       compute_laplacian(in_field, output_field, step_size_x, step_size_y);
       t2 = wall_time();
       time_finite += (t2-t1);	

       t1 = wall_time(); 
       meshptr->laplacian(in_field_unstr, out_field_unstr);
       t2 = wall_time();
       time_gauss += (t2-t1);  
    }
    std::cout << "time for laplacian using finite differences: " << time_finite/10 << std::endl; 
    std::cout << "time for laplacian with lookup table: " << time_gauss/10 << std::endl; 

    std::ofstream myfile("benchmark.csv", std::ios::app);
    myfile << nx << " " << ny << " " << time_finite/10 << " " << time_gauss/10 << "\n";
    myfile.close();

}


void benchmark_cache(std::shared_ptr<Mesh>& meshptr, const std::size_t nx, const std::size_t ny) 
{

    MatrixXd in_field = MatrixXd::Random(nx,ny);
    MatrixXd out_field = MatrixXd::Zero(in_field.rows(), in_field.cols());
    
    Eigen::VectorXd in_field_unstr(nx*ny);
    Eigen::VectorXd out_field_unstr(in_field_unstr.size());
    std::cout << in_field_unstr.size() << std::endl;

    

    const unsigned int halo_points = 1;
    const double dirichlet_value = 0.0;
    double lx = 100; //100;
    double ly = 100; //100;
    const double step_size_x = lx/nx;  
    const double step_size_y = ly/ny;

    //compute laplacian using finite differences
    double t1, t2, time_finite = 0, time_gauss = 0;    
    for(unsigned int i = 0; i < 10; ++i) {
       t1 = wall_time(); 
       meshptr->laplacian(in_field_unstr, out_field_unstr);
       t2 = wall_time();
       time_gauss += (t2-t1); 

       t1 = wall_time();
       compute_laplacian(in_field, out_field, step_size_x, step_size_y);
       t2 = wall_time();
       time_finite += (t2-t1);
    }
    std::cout << "time for laplacian using finite differences: " << time_finite/10 << std::endl; 
    std::cout << "time for laplacian with lookup table: " << time_gauss/10 << std::endl; 

    std::ofstream myfile("benchmark_cache.csv", std::ios::app);
    myfile << nx << " " << " " << ny << time_finite/10 << " " << time_gauss/10 << "\n";
    myfile.close();

}

void benchmark_reindexing(std::shared_ptr<Mesh>& meshptr)
{
        int num_elements = meshptr->get_num_elements();
	Eigen::VectorXd in_field(meshptr->get_num_elements());
        Eigen::VectorXd out_field(meshptr->get_num_elements());

 	double t1,t2,time=0, time_reindex = 0;

        for(unsigned int i = 0; i < 10; ++i) {
		t1 = wall_time();
		meshptr->laplacian(in_field,out_field);
   		t2 = wall_time();
		time += (t2-t1);
	}
        meshptr->reindex_cleverly(-50, -50, 50, 50, 8);
	
	for(unsigned int i = 0; i < 10; ++i) {
		t1 = wall_time();
		meshptr->laplacian(in_field,out_field);
   		t2 = wall_time();
		time_reindex += (t2-t1);
	}     
    std::cout << "time before reindexings: " << time/10 << std::endl; 
    std::cout << "time after reindexing: " << time_reindex/10 << std::endl; 

    std::ofstream myfile("benchmark_reindex.csv", std::ios::app);
    myfile << num_elements << " " << time/10 << " " << time_reindex/10 << "\n";
    myfile.close();

}


void test_diffusion_triangular(std::shared_ptr<Mesh>& meshptr, const unsigned int num_iter, const double time_step, bool reindex=false)
{
  size_t nx = 100, ny = 100;
  double lx = 100, ly = 100, x0 = -50, y0 = -50;
  //creation of initial field
  if (reindex) {
    meshptr->reindex_cleverly(x0, y0, x0 + lx, y0 + ly, 8);
  }
  std::cout << "reindexing worked" << std::endl;
  MatrixXd in_field = MatrixXd::Zero(nx, ny);
  const size_t bx0 = 24, by0 = 24, lbx = 52, lby = 52; // where the block of ones is
  in_field.block(bx0, by0, lbx, lby) = Eigen::MatrixXd::Ones(lbx, lby);
  Eigen::VectorXd in_field_unstr = interpolate(x0, lx, y0, lx, in_field, meshptr); //hardcoded
  std::cout << "interpolation worked" << std::endl;

  std::ofstream myfile; 
  myfile.open("initial_field.csv");
  myfile << in_field_unstr;
  myfile.close();

  Eigen::MatrixXd diffusion_out = diffusion2(meshptr, num_iter, time_step, in_field_unstr); 
  std::cout << "diffusion?" << std::endl;
  /*
  std::ofstream myfile;
  const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
  myfile.open("all_diffusion.csv");
  myfile << diffusion_out.format(CSVFormat);
  myfile.close();
  */
  
  //std::cout << diffusion_out.col(50) << std::endl;
 
  myfile.open("output_diffusion.csv");
  myfile << diffusion_out.col(num_iter - 1);
  myfile.close();

  
  
}

void test_diffusion_rectangular(std::shared_ptr<Mesh>& meshptr, const unsigned int num_iter, const double time_step)
{
    size_t nx = 100, ny = 100;
    double lx = 100, ly = 100, x0 = -50, y0 = -50;   
    MatrixXd in_field = MatrixXd::Zero(nx, ny);
    const size_t bx0 = 24, by0 = 24, lbx = 52, lby = 52; // where the block of ones is
    in_field.block(bx0, by0, lbx, lby) = Eigen::MatrixXd::Ones(lbx, lby);
    Eigen::VectorXd in_field_unstr = interpolate(x0, lx, y0, ly, in_field, meshptr);
    std::cout << "created initial_field" << std::endl;

    std::ofstream myfile;
    myfile.open("initial_field.csv");
    myfile << in_field_unstr;
    myfile.close();

    diffusion(meshptr, num_iter, time_step, in_field_unstr);
    
    myfile.open("output_diffusion.csv");
    myfile << in_field_unstr;
    myfile.close();

}

//command line arguments nx, ny
int main(int argc, char *argv[])
{

  //default
  size_t nx = 50;
  size_t ny = 50;

  if(argc > 1)
  {
    sscanf(argv[1], "%zu", &nx);
	  sscanf(argv[2], "%zu", &ny);
  }

  std::cout << "nx: " << nx << std::endl;
  std::cout << "ny: " << ny << std::endl;

  double lx = 100; 
  double ly = 100; 
  double x0 = -50; 
  double y0 = -50;

  const unsigned int num_iter = 100;
  const double time_step = 0.5;  // lower for fine meshes
  
  
  // Mesh mesh(nx, ny, x0, lx, y0, ly);
  // std::shared_ptr<Mesh> meshptr = std::make_shared<Mesh>(mesh);
  // std::cout << "mesh construction worked" << std::endl;
  //benchmark(meshptr, nx, ny);
  // benchmark_cache(meshptr, nx, ny);
   
  //Mesh mesh("../testmesh.msh");

  Mesh mesh(x0, lx, y0, ly, .4, false, "../testmesh.msh");
  std::shared_ptr<Mesh> meshptr = std::make_shared<Mesh>(mesh);
  std::cout << "construction of mesh worked" << std::endl;
  meshptr->reindex_cleverly(x0, y0, x0 + lx, y0 + ly, 8);
  // std::cout << "number of elements: " << meshptr->get_num_elements() << std::endl;

  // test_diffusion_triangular(meshptr, num_iter, time_step, true);

  write_mesh_to_file(meshptr);
    
  return 0;
  
}

