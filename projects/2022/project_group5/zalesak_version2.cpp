#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>

#include <cmath>

#include "pat_api.h"
#include "utils.h"

namespace {

void updateHalo(Storage3D<double>& inField) {
  const int xInterior = inField.xMax() - inField.xMin();
  const int yInterior = inField.yMax() - inField.yMin();

  // bottom edge (without corners)
  for(std::size_t k = 0; k < inField.zMin(); ++k) {
    for(std::size_t j = 0; j < inField.yMin(); ++j) {
      for(std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
        inField(i, j, k) = inField(i, j + yInterior, k);
      }
    }
  }

  // top edge (without corners)
  for(std::size_t k = 0; k < inField.zMin(); ++k) {
    for(std::size_t j = inField.yMax(); j < inField.ySize(); ++j) {
      for(std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
        inField(i, j, k) = inField(i, j - yInterior, k);
      }
    }
  }

  // left edge (including corners)
  for(std::size_t k = 0; k < inField.zMin(); ++k) {
    for(std::size_t j = inField.yMin(); j < inField.yMax(); ++j) {
      for(std::size_t i = 0; i < inField.xMin(); ++i) {
        inField(i, j, k) = inField(i + xInterior, j, k);
      }
    }
  }

  // right edge (including corners)
  for(std::size_t k = 0; k < inField.zMin(); ++k) {
    for(std::size_t j = inField.yMin(); j < inField.yMax(); ++j) {
      for(std::size_t i = inField.xMax(); i < inField.xSize(); ++i) {
        inField(i, j, k) = inField(i - xInterior, j, k);
      }
    }
  }
}

void apply_2nd_diffusion(Storage3D<double>& inField, Storage3D<double>& outField, double alpha,
    unsigned numIter, int x, int y, int z, int halo)
{
    for (std::size_t iter = 0; iter < numIter; ++iter)
    {
        updateHalo(inField);
        
        for (std::size_t k = 0; k < inField.zMax(); ++k)
        {
            for (std::size_t j = inField.yMin(); j < inField.yMax(); ++j)
            {
                for (std::size_t i = inField.xMin(); i < inField.xMax(); ++i)
                {
                    double lap = -4.0 * inField(i, j, k) + inField(i - 1, j, k) +
                        inField(i + 1, j, k) + inField(i, j - 1, k) + inField(i, j + 1, k);


                    //update field
                    if(iter == numIter - 1) 
                    {
                        outField(i, j, k) = inField(i, j, k) + alpha * lap;
                    }
                    else
                    {
                        inField(i, j, k) = inField(i, j, k) + alpha * lap;
                    }                    
                    
                }
            }
        }
    }
}

void apply_4th_diffusion(Storage3D<double>& inField, Storage3D<double>& outField, double alpha,
                     unsigned numIter, int x, int y, int z, int halo)
{

  Storage3D<double> tmp1Field(x, y, z, halo);

  for(std::size_t iter = 0; iter < numIter; ++iter)
  {
    updateHalo(inField);

    for(std::size_t k = 0; k < inField.zMax(); ++k)
    {
        
      // apply the first laplacian
      for(std::size_t j = inField.yMin() - 1; j < inField.yMax() + 1; ++j)
      {
        for(std::size_t i = inField.xMin() - 1; i < inField.xMax() + 1; ++i)
        {
          tmp1Field(i, j, 0) = -4.0 * inField(i, j, k) + inField(i - 1, j, k) +
                               inField(i + 1, j, k) + inField(i, j - 1, k) + inField(i, j + 1, k);
        }
      }

      // apply the second laplacian
      for(std::size_t j = inField.yMin(); j < inField.yMax(); ++j)
      {
        for(std::size_t i = inField.xMin(); i < inField.xMax(); ++i)
        {
          double laplap = -4.0 * tmp1Field(i, j, 0) + tmp1Field(i - 1, j, 0) +
                          tmp1Field(i + 1, j, 0) + tmp1Field(i, j - 1, 0) + tmp1Field(i, j + 1, 0);

          // and update the field
          if(iter == numIter - 1) {
            outField(i, j, k) = inField(i, j, k) - alpha * laplap;
          } else {
            inField(i, j, k) = inField(i, j, k) - alpha * laplap;
          } 
        }
      }
    }
  }
}



void apply_6th_diffusion(Storage3D<double>& inField, Storage3D<double>& outField, double alpha,
    unsigned numIter, int x, int y, int z, int halo)
{
    
    Storage3D<double> tmp1Field(x, y, z, halo);
    Storage3D<double> tmp2Field(x, y, z, halo);
    
    for (std::size_t iter = 0; iter < numIter; ++iter)
    {
        updateHalo(inField);
        
        for(std::size_t k = 0; k < inField.zMax(); ++k)
        {   
            //apply first laplacian
            for(std::size_t j = inField.yMin() - 2; j < inField.yMax() + 2; ++j)
            {
                for(std::size_t i = inField.xMin() - 2; i < inField.xMax() + 2; ++i)
                {
                    tmp1Field(i, j, 0) = -4.0 * inField(i, j, k) + inField(i - 1, j, k) +
                               inField(i + 1, j, k) + inField(i, j - 1, k) + inField(i, j + 1, k);
                    
                }
            }
            
            //second laplacian
            for(std::size_t j = inField.yMin() - 1; j < inField.yMax() + 1; ++j)
            {
                for(std::size_t i = inField.xMin() - 1; i < inField.xMax() + 1; ++i)
                {
                    tmp2Field(i, j, 0) = -4.0 * tmp1Field(i, j, 0) + tmp1Field(i - 1, j, 0) +
                               tmp1Field(i + 1, j, 0) + tmp1Field(i, j - 1, 0) + tmp1Field(i, j + 1, 0);
                }
            }

            //thrid laplacian
            for(std::size_t j = inField.yMin(); j < inField.yMax(); ++j)
            {
                for(std::size_t i = inField.xMin(); i < inField.xMax(); ++i)
                {
                    double laplaplap = -4.0 * tmp2Field(i, j, 0) + tmp2Field(i - 1, j, 0) + 
                        tmp2Field(i + 1, j, 0) + tmp2Field(i, j - 1, 0) + tmp2Field(i, j + 1, 0);
                    
                    //update field
                    if(iter == numIter - 1) 
                    {
                        outField(i, j, k) = inField(i, j, k) + alpha * laplaplap;
                    }
                    else
                    {
                        inField(i, j, k) = inField(i, j, k) + alpha * laplaplap;
                    }
                    
                }
            }
        }  
    }   
}

    
    
    
void Zalesak_filter(Storage3D<double>& phi_initial, Storage3D<double>& phi_diff, Storage3D<double>& output_field, double alpha, int n, double dx, double dy, double dt, int x, int y, int z, int numIter, int halo) {
        for (std::size_t iter = 0; iter < numIter; ++iter) {
            //phi_initial: infield without diffusion
            //phi_diff: diffusive flux of order (n-2)
            //n: order of diffusion equation
            
            updateHalo(phi_initial);
            //phi_diff: initialized, but without any further diffusion
            if (n == 4){
                for (std::size_t k=0; k < phi_diff.zMax(); ++k){
                    for (std::size_t i = phi_diff.xMin(); i < phi_diff.xMax(); ++i){
                        for (std::size_t j = phi_diff.yMin(); j < phi_diff.yMax(); ++j){
                            double lap = -4.0*phi_diff(i,j,k) + phi_diff(i-1,j,k) + phi_diff(i+1,j,k) + phi_diff(i,j-1,k) + phi_diff(i,j+1,k);
                            phi_diff(i,j,k) = phi_diff(i,j,k) + alpha*lap;
                        }
                    }
                }
            }
            if (n == 6){
                Storage3D<double> tmp1Field(x,y,z,halo);
                //apply first laplacian:
                for (std::size_t k=0; k < phi_diff.zMax(); ++k){
                    for (std::size_t i = phi_diff.xMin(); i < phi_diff.xMax(); ++i){
                        for (std::size_t j = phi_diff.yMin(); j < phi_diff.yMax(); ++j){
                            tmp1Field(i,j,0) = -4.0*phi_diff(i,j,k) + phi_diff(i-1,j,k) + phi_diff(i+1,j,k) + phi_diff(i,j-1,k) + phi_diff(i,j+1,k);
                        }
                    }
                }
                
                //apply second laplacian:
                for (std::size_t k=0; k < phi_diff.zMax(); ++k){
                    for (std::size_t i = phi_diff.xMin(); i < phi_diff.xMax(); ++i){
                        for (std::size_t j = phi_diff.yMin(); j < phi_diff.yMax(); ++j){
                            double laplap = -4.0*tmp1Field(i,j,0) + tmp1Field(i-1,j,k) + tmp1Field(i+1,j,k) + tmp1Field(i,j-1,k) + tmp1Field(i,j+1,k);
                            phi_diff(i,j,k) = phi_diff(i,j,k) + alpha*laplap;
                        }
                    }
                }
            }
            
            
            
            
            //definition of diffusive fluxes:
        updateHalo(phi_diff);
        Storage3D<double> Fx(x, y, z, halo);
        Storage3D<double> Fy(x, y, z, halo);
            
            //A: high order diffusive flux
        Storage3D<double> A_x(x, y, z, halo);
        Storage3D<double> A_y(x, y, z, halo);
            
            //FL: low order diffusive flux (second order)
        Storage3D<double> FL_x(x, y, z, halo);
        Storage3D<double> FL_y(x, y, z, halo);
            
            //AL
        Storage3D<double> AL_x(x, y, z, halo);
        Storage3D<double> AL_y(x, y, z, halo);
            
            //AHL = A - AL -->difference between high and low level
        Storage3D<double> AHL_x(x, y, z, halo);
        Storage3D<double> AHL_y(x, y, z, halo);
            
            //definition of phi_star: phi_star = phi_now - (AL[i+1]-AL[i])
        Storage3D<double> phi_star(x, y, z, halo);
            
            //definition of weights C:
        Storage3D<double> P_in(x, y, z, halo);
        Storage3D<double> P_out(x, y, z, halo);
            
            //phi_min/phi_max
            //updateHalo(phi_initial); -->?!?
        Storage3D<double> phi_min(x, y, z, halo);
        Storage3D<double> phi_max(x, y, z, halo);
            
            //R_plus/minus
            Storage3D<double> R_plus(x, y, z, halo);
        Storage3D<double> R_minus(x, y, z, halo);
            //changed:
        int epsilon = std::pow(10, -20);
        
            //C
        Storage3D<double> C_x(x, y, z, halo);
        Storage3D<double> C_y(x, y, z, halo);

        //updateHalo(phi_initial);
        
        for (std::size_t k = 0; k < phi_diff.zMax(); ++k) {
            for (std::size_t i = phi_diff.xMin(); i < phi_diff.xMax(); ++i) {
                for (std::size_t j = phi_diff.yMin(); j < phi_diff.yMax(); ++j) {
                    Fx(i,j,k) = std::pow(-1,n/2)*alpha/2/dx*(phi_diff(i+1,j,k)-phi_diff(i-1,j,k));
                    Fy(i, j, k) = std::pow(-1, n / 2) * alpha / 2 / dy * (phi_diff(i, j+1, k) - phi_diff(i, j-1, k));
                    
                    //A: high order diffusive fluxes:
                    A_x(i,j,k) = dt/dx*Fx(i,j,k);
                    A_y(i,j,k) = dt/dx*Fy(i,j,k);

                }
            }
        }        
        
        //A{i+1/2} = AL[i+1]
        updateHalo(A_x);
        updateHalo(A_y);
        
        for (std::size_t k = 0; k < phi_diff.zMax(); ++k) {
            for (std::size_t i = phi_diff.xMin(); i < phi_diff.xMax(); ++i) {
                for (std::size_t j = phi_diff.yMin(); j < phi_diff.yMax(); ++j) {
                    A_x(i, j, k) = 1 / 2 * (A_x(i, j, k) + A_x(i - 1, j, k));
                    A_y(i, j, k) = 1 / 2 * (A_y(i, j, k) + A_y(i - 1, j, k));
                    
                    //FL: low order diffusive flux (second order)
                    FL_x(i, j, k) = -alpha / 2 / dx * (phi_initial(i + 1, j, k) - phi_initial(i - 1, j, k));
                    FL_y(i, j, k) = -alpha / 2 / dy * (phi_initial(i, j + 1, k) - phi_initial(i, j - 1, k));
                    
                    //AL:
                    AL_x(i,j,k) = dt/dx*FL_x(i,j,k);
                    AL_y(i,j,k) = dt/dx*FL_y(i,j,k);
                }
            }
        }

        //AL{i+1/2} = AL[i+1]
        updateHalo(AL_x);
        updateHalo(AL_y);

        for (std::size_t k = 0; k < AL_x.zMax(); ++k) {
            for (std::size_t i = phi_diff.xMin(); i < phi_diff.xMax(); ++i) {
                for (std::size_t j = phi_diff.yMin(); j < phi_diff.yMax(); ++j) {
                    AL_x(i, j, k) = 1 / 2 * (AL_x(i, j, k) + AL_x(i - 1, j, k));
                    AL_y(i, j, k) = 1 / 2 * (AL_y(i, j, k) + AL_y(i -  1, j, k));
                    
                    //AHL = A - AL
                    AHL_x(i, j, k) = A_x(i, j, k) - AL_x(i, j, k);
                    AHL_y(i, j, k) = A_y(i, j, k) - AL_y(i, j, k);
                    
                    //phi_star
                    phi_star(i, j, k) = phi_initial(i, j, k) - (AL_x(i + 1, j, k) - AL_x(i, j, k)) - (AL_y(i, j + 1, k) - AL_y(i, j, k));
                    
                    //definition of weights: C --> P_in/P_out
                    P_in(i, j, k) = std::max(0., AHL_x(i, j, k)) - std::min(0., AHL_x(i + 1, j, k)) + std::max(0., AHL_y(i, j, k)) - std::min(0., AHL_y(i, j + 1, k));
                    P_out(i, j, k) = std::max(0., AHL_x(i + 1, j, k)) - std::min(0., AHL_x(i, j, k)) + std::max(0., AHL_y(i, j + 1, k)) - std::min(0., AHL_y(i, j, k));
                    
                    //phi_min/phi_max
                    phi_min(i,j,k) = std::min(phi_initial(i-1,j,k),std::min(phi_initial(i,j,k),std::min(phi_initial(i+1,j,k), std::min(phi_initial(i,j-1,k), phi_initial(i,j+1,k)))));
                    phi_max(i,j,k) = std::max(phi_initial(i-1,j,k),std::max(phi_initial(i,j,k),std::max(phi_initial(i+1,j,k), std::max(phi_initial(i,j-1,k), phi_initial(i,j+1,k)))));
                    
                    //R_plus/minus
                    R_plus(i, j, k) = (phi_max(i, j, k) - phi_star(i, j, k)) / (P_in(i, j, k) + epsilon);
                    R_minus(i, j, k) = (phi_min(i, j, k) - phi_star(i, j, k)) / (P_out(i, j, k) + epsilon);
                    
                }
            }
        }
        
        
            //C
        updateHalo(R_plus);
        updateHalo(R_minus);
        for (std::size_t k = 0; k < AL_x.zMax(); ++k) {
            for (std::size_t i = phi_diff.xMin(); i < phi_diff.xMax(); ++i) {
                for (std::size_t j = phi_diff.yMin(); j < phi_diff.yMax(); ++j) {
                    if (AHL_x(i, j, k) >= 0) {
                        C_x(i,j,k) = std::min(1., std::min(R_minus(i-1,j,k), R_plus(i,j,k)));
                    }
                    else {
                        C_x(i, j, k) = std::min(1., std::min(R_plus(i - 1, j, k), R_minus(i, j, k)));
                    }

                    if (AHL_y(i, j, k) >= 0) {
                        C_y(i, j, k) = std::min(1., std::min(R_plus(i, j, k), R_minus(i, j - 1, k)));
                    }
                    else {
                        C_y(i, j, k) = std::min(1., std::min(R_plus(i, j - 1, k), R_minus(i, j, k)));
                    }
                }
            }
        }
            
            
            //iteration
            if (iter == numIter-1) {
                for (std::size_t k = 0; k < AL_x.zMax(); ++k) {
                    for (std::size_t i = phi_diff.xMin(); i < phi_diff.xMax(); ++i) {
                        for (std::size_t j = phi_diff.yMin(); j < phi_diff.yMax(); ++j) {
                            output_field(i, j, k) = phi_star(i, j, k) - (C_x(i + 1, j, k) * AHL_x(i + 1, j, k) - C_x(i, j, k) * AHL_x(i, j, k)) - (C_y(i, j + 1, k) * AHL_y(i, j + 1, k) - C_y(i, j, k) * AHL_y(i, j, k));
                            //output_field(i,j,k) = R_plus(i,j,k);
                        }
                    }
                }
            }
            else {
                for (std::size_t k = 0; k < AL_x.zMax(); ++k) {
                    for (std::size_t i = phi_diff.xMin(); i < phi_diff.xMax(); ++i) {
                        for (std::size_t j = phi_diff.yMin(); j < phi_diff.yMax(); ++j) {
                            phi_initial(i, j, k) = phi_star(i, j, k) - (C_x(i + 1, j, k) * AHL_x(i + 1, j, k) - C_x(i, j, k) * AHL_x(i, j, k)) - (C_y(i, j + 1, k) * AHL_y(i, j + 1, k) - C_y(i, j, k) * AHL_y(i, j, k));
                        }
                    }
                }
            }
            
            
            

        }
    }
            

    
    
    
    

void reportTime(const Storage3D<double>& storage, int nIter, double diff)
{
  std::cout << "# ranks nx ny ny nz num_iter time\ndata = np.array( [ \\\n";
  int size;
#pragma omp parallel
  {
#pragma omp master
    { size = omp_get_num_threads(); }
  }
  std::cout << "[ " << size << ", " << storage.xMax() - storage.xMin() << ", "
            << storage.yMax() - storage.yMin() << ", " << storage.zMax() << ", " << nIter << ", "
            << diff << "],\n";
  std::cout << "] )" << std::endl;
}

} // namespace

int main(int argc, char const* argv[]) {
#ifdef CRAYPAT
  PAT_record(PAT_STATE_OFF);
#endif
  int x = atoi(argv[2]);        // string to integer
  int y = atoi(argv[4]);
  int z = atoi(argv[6]);
  int iter = atoi(argv[8]);
  int order = atoi(argv[10]);

  assert(x > 0 && y > 0 && z > 0 && iter > 0);
  assert(order==2 || order==4 || order==6);

  int nHalo = order / 2;
  double alpha_n   = 0;

    
  Storage3D<double> input(x, y, z, nHalo);
  input.initialize();
  Storage3D<double> output(x, y, z, nHalo);
  output.initialize();

  std::ofstream fout;
  fout.open("in_field.dat", std::ios::binary | std::ofstream::trunc);
  input.writeFile(fout);
  fout.close();

    
    
    //changed:
    double dx = 1;
    double dy = 1;
    double dt = 1;
    //change finished
    
    


#ifdef CRAYPAT          //performance analysis
  PAT_record(PAT_STATE_ON);
#endif
    
  auto start = std::chrono::steady_clock::now();
    
  if (order == 2)
    {
      alpha_n = 1/32.;
      //apply_2nd_diffusion(input, output, alpha_n, iter, x, y, z, nHalo);
      
      //changed:
      Storage3D<double> phi_diffusive(x, y, z, nHalo);
      phi_diffusive.initialize();
      
      Zalesak_filter(input, phi_diffusive, output, alpha_n, order, dx, dy, dt, x, y, z, iter, nHalo);
      
      //change finished
    }
  else if (order == 4)
    {
      alpha_n = 1/32.;
      //apply_4th_diffusion(input, output, alpha_n, iter, x, y, z, nHalo);
      
      //changed:
      Storage3D<double> phi_diffusive(x, y, z, nHalo);
      phi_diffusive.initialize();
      
      //apply_2nd_diffusion(input, phi_diffusive, 1/32, iter, x, y, z, nHalo);
      Zalesak_filter(input, phi_diffusive, output, alpha_n, order, dx, dy, dt, x, y, z, iter, nHalo);
      
      
      
      //change finished
      
      
      
      
      
    }
  else if (order == 6)
    {
      alpha_n = 1/256.;
      //apply_6th_diffusion(input, output, alpha_n, iter, x, y, z, nHalo);
      
      //changed:
      Storage3D<double> phi_diffusive(x,y,z,nHalo);
      phi_diffusive.initialize();
      
      //apply_4th_diffusion(input, phi_diffusive, alpha_n, iter, x, y, z, nHalo);
      Zalesak_filter(input, phi_diffusive, output, alpha_n, order, dx, dy, dt, x, y, z, iter, nHalo);
      
      //change finished
    }
    
  auto end = std::chrono::steady_clock::now();
    
#ifdef CRAYPAT
  PAT_record(PAT_STATE_OFF);
#endif
  updateHalo(output);
  fout.open("out_field.dat", std::ios::binary | std::ofstream::trunc);
  output.writeFile(fout);
  fout.close();

  auto diff = end - start;
  double timeDiff = std::chrono::duration<double, std::milli>(diff).count() / 1000.;
  reportTime(output, iter, timeDiff);

  return 0;
}