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

void apply_8th_diffusion(Storage3D<double>& inField, Storage3D<double>& outField, double alpha,
    unsigned numIter, int x, int y, int z, int halo)
{
    
    Storage3D<double> tmp1Field(x, y, z, halo);
    Storage3D<double> tmp2Field(x, y, z, halo);
    Storage3D<double> tmp3Field(x, y, z, halo);
    double laplaplaplap;
    
    for (std::size_t iter = 0; iter < numIter; ++iter)
    {
        updateHalo(inField);
        
        for(std::size_t k = 0; k < inField.zMax(); ++k)
        {   
            //apply first laplacian
            for(std::size_t j = inField.yMin() - 3; j < inField.yMax() + 3; ++j)
            {
                for(std::size_t i = inField.xMin() - 3; i < inField.xMax() + 3; ++i)
                {
                    tmp1Field(i, j, 0) = -4.0 * inField(i, j, k) + inField(i - 1, j, k) +
                               inField(i + 1, j, k) + inField(i, j - 1, k) + inField(i, j + 1, k);
                    
                }
            }
            
            //second laplacian
            for(std::size_t j = inField.yMin() - 2; j < inField.yMax() + 2; ++j)
            {
                for(std::size_t i = inField.xMin() - 2; i < inField.xMax() + 2; ++i)
                {
                    tmp2Field(i, j, 0) = -4.0 * tmp1Field(i, j, 0) + tmp1Field(i - 1, j, 0) +
                               tmp1Field(i + 1, j, 0) + tmp1Field(i, j - 1, 0) + tmp1Field(i, j + 1, 0);
                }
            }

            //thrid laplacian
            for(std::size_t j = inField.yMin()-1; j < inField.yMax()+1; ++j)
            {
                for(std::size_t i = inField.xMin()-1; i < inField.xMax()+1; ++i)
                {
                    tmp3Field(i,j,0) = -4.0 * tmp2Field(i, j, 0) + tmp2Field(i - 1, j, 0) + 
                        tmp2Field(i + 1, j, 0) + tmp2Field(i, j - 1, 0) + tmp2Field(i, j + 1, 0);
                    
                }
            }
            
            // 4th laplacian
            for(std::size_t j = inField.yMin(); j < inField.yMax(); ++j)
            {
                for(std::size_t i = inField.xMin(); i < inField.xMax(); ++i)
                {
                    laplaplaplap = -4.0 * tmp3Field(i, j, 0) + tmp3Field(i - 1, j, 0) + 
                        tmp3Field(i + 1, j, 0) + tmp3Field(i, j - 1, 0) + tmp3Field(i, j + 1, 0);
                    
                    //update field
                    if(iter == numIter - 1) 
                    {
                        outField(i, j, k) = inField(i, j, k) - alpha * laplaplaplap;
                    }
                    else
                    {
                        inField(i, j, k) = inField(i, j, k) - alpha * laplaplaplap;
                    }
                    
                }
            }
        
        }  
    }   
}


/**
 * Apply a simple filter to fourth order solutions, setting non downgradient fluxes to zero
 * 
 * @param[in] Storage3D<double> inField: initial 3D field to undergo diffusion
 * @param[in] Storage3D<double> outField: allocated 3D field to where output is written
 * @param[in] double alpha: smoothing parameter
 * @param[in] unsigned numIter: Number of tiemsteps
 * @param[in] int x, int y, int z: Number of gridpoints in x, y and z-directions
 * @param[in] int halo: Number of halopoints
 *
 * @param[out] Storage3D<double> outField: 4th order diffused, flux limited field
 */    
void simple_flux_limiter4(Storage3D<double>& inField, Storage3D<double>& outField, double alpha,
                     unsigned numIter, int x, int y, int z, int halo){
    
  // temporary field for storing first laplacian 
  Storage3D<double> tmp1Field(x, y, z, halo);
  
  // Loop over timesteps
  for(std::size_t iter = 0; iter < numIter; ++iter)
  {
    // update halo (assure .. at boundaries)
    updateHalo(inField);
    
    // Loop over k levels
    for(std::size_t k = 0; k < inField.zMax(); ++k)
    {
        
      // apply the first laplacian
      for(std::size_t j = inField.yMin() - 1; j < inField.yMax() + 1; ++j)
      {
        for(std::size_t i = inField.xMin() - 1; i < inField.xMax() + 1; ++i)
        {
          // grad(i, j, 0) = 
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
          
          // check whether higher order flux is downgradient
          int sign = (((-laplap)*tmp1Field(i, j, 0)) > 0) - (((-laplap)*tmp1Field(i, j, 0)) < 0);
          
          //  set F=0 if higher order flux is not downgradient
          double F;
          if(sign <= 0)
          {
              F = 0;
          }
          else
          {
              F = laplap;
          }  
          // Update the field
          if(iter == numIter - 1) {
            outField(i, j, k) = inField(i, j, k) - alpha * F;
          } else {
            inField(i, j, k) = inField(i, j, k) - alpha * F;
          } 
        }
      }
    }
  }
}

    
/**
 * Apply a simple filter to sixth order solutions, setting non downgradient fluxes to zero
 * 
 * @param[in] Storage3D<double> inField: initial 3D field to undergo diffusion
 * @param[in] Storage3D<double> outField: allocated 3D field to where output is written
 * @param[in] double alpha: smoothing parameter
 * @param[in] unsigned numIter: Number of tiemsteps
 * @param[in] int x, int y, int z: Number of gridpoints in x, y and z-directions
 * @param[in] int halo: Number of halopoints
 *
 * @param[out] Storage3D<double> outField: 6th order diffused, flux limited field
 */    
void simple_flux_limiter6(Storage3D<double>& inField, Storage3D<double>& outField, double alpha,
                     unsigned numIter, int x, int y, int z, int halo){

  // temporary fields for storing first and second laplacian 
  Storage3D<double> tmp1Field(x, y, z, halo);
  Storage3D<double> tmp2Field(x, y, z, halo);

  // Loop over timesteps
  for(std::size_t iter = 0; iter < numIter; ++iter)
  {
    updateHalo(inField);

    // Loop over k levels
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
              tmp2Field(i, j, 0) = -4.0 * tmp1Field(i, j, 0) + tmp1Field(i - 1, j, 0) +
                              tmp1Field(i + 1, j, 0) + tmp1Field(i, j - 1, 0) + tmp1Field(i, j + 1, 0);
            }
        }
        
        // apply thrid laplacian
        for(std::size_t j = inField.yMin(); j < inField.yMax(); ++j)
        {
            for(std::size_t i = inField.xMin(); i < inField.xMax(); ++i)
            {
                double laplaplap = -4.0 * tmp2Field(i, j, 0) + tmp2Field(i - 1, j, 0) + 
                    tmp2Field(i + 1, j, 0) + tmp2Field(i, j - 1, 0) + tmp2Field(i, j + 1, 0);


                // check sign to see if higher order flux is downgradient
                int sign = (((laplaplap)*tmp1Field(i, j, 0)) > 0) - (((laplaplap)*tmp1Field(i, j, 0)) < 0);
                
                //  set F=0 if higher order flux is not downgradient
                double F;
                if(sign <= 0)
                {
                  F = 0;
                }
                else
                {
                  F = laplaplap;
                }  
                // Update the field
                if(iter == numIter - 1) {
                    outField(i, j, k) = inField(i, j, k) + alpha * F;
                } 
                else {
                    inField(i, j, k) = inField(i, j, k) + alpha * F;
                } 
            }
          }
    }
  }
}

/**
 * Apply a simple filter to eight order solutions, setting non downgradient fluxes to zero
 * 
 * @param[in] Storage3D<double> inField: initial 3D field to undergo diffusion
 * @param[in] Storage3D<double> outField: allocated 3D field to where output is written
 * @param[in] double alpha: smoothing parameter
 * @param[in] unsigned numIter: Number of tiemsteps
 * @param[in] int x, int y, int z: Number of gridpoints in x, y and z-directions
 * @param[in] int halo: Number of halopoints
 *
 * @param[out] Storage3D<double> outField: 8th order diffused, flux limited field
 */  
void simple_flux_limiter8(Storage3D<double>& inField, Storage3D<double>& outField, double alpha,
    unsigned numIter, int x, int y, int z, int halo)
{
    
    Storage3D<double> tmp1Field(x, y, z, halo);
    Storage3D<double> tmp2Field(x, y, z, halo);
    Storage3D<double> tmp3Field(x, y, z, halo);
    double laplaplaplap;
    
    for (std::size_t iter = 0; iter < numIter; ++iter)
    {
        updateHalo(inField);
        
        for(std::size_t k = 0; k < inField.zMax(); ++k)
        {   
            //apply first laplacian
            for(std::size_t j = inField.yMin() - 3; j < inField.yMax() + 3; ++j)
            {
                for(std::size_t i = inField.xMin() - 3; i < inField.xMax() + 3; ++i)
                {
                    tmp1Field(i, j, 0) = -4.0 * inField(i, j, k) + inField(i - 1, j, k) +
                               inField(i + 1, j, k) + inField(i, j - 1, k) + inField(i, j + 1, k);
                    
                }
            }
            
            //second laplacian
            for(std::size_t j = inField.yMin() - 2; j < inField.yMax() + 2; ++j)
            {
                for(std::size_t i = inField.xMin() - 2; i < inField.xMax() + 2; ++i)
                {
                    tmp2Field(i, j, 0) = -4.0 * tmp1Field(i, j, 0) + tmp1Field(i - 1, j, 0) +
                               tmp1Field(i + 1, j, 0) + tmp1Field(i, j - 1, 0) + tmp1Field(i, j + 1, 0);
                }
            }

            //thrid laplacian
            for(std::size_t j = inField.yMin()-1; j < inField.yMax()+1; ++j)
            {
                for(std::size_t i = inField.xMin()-1; i < inField.xMax()+1; ++i)
                {
                    tmp3Field(i,j,0) = -4.0 * tmp2Field(i, j, 0) + tmp2Field(i - 1, j, 0) + 
                        tmp2Field(i + 1, j, 0) + tmp2Field(i, j - 1, 0) + tmp2Field(i, j + 1, 0);
                    
                }
            }
            
            // 4th laplacian
            for(std::size_t j = inField.yMin(); j < inField.yMax(); ++j)
            {
                for(std::size_t i = inField.xMin(); i < inField.xMax(); ++i)
                {
                    laplaplaplap = -4.0 * tmp3Field(i, j, 0) + tmp3Field(i - 1, j, 0) + 
                        tmp3Field(i + 1, j, 0) + tmp3Field(i, j - 1, 0) + tmp3Field(i, j + 1, 0);
                    
                    // check sign to see if higher order flux is downgradient
                    int sign = (((-laplaplaplap)*tmp1Field(i, j, 0)) > 0) - (((-laplaplaplap)*tmp1Field(i, j, 0)) < 0);

                    //  set F=0 if higher order flux is not downgradient
                    double F;
                    if(sign <= 0)
                    {
                      F = 0;
                    }
                    else
                    {
                      F = laplaplaplap;
                    }  
                    // Update the field
                    if(iter == numIter - 1) {
                        outField(i, j, k) = inField(i, j, k) - alpha * F;
                    } 
                    else {
                        inField(i, j, k) = inField(i, j, k) - alpha * F;
                    } 
                }
            }
        
        }  
    }   
}

void reportTime(const Storage3D<double>& storage, int nIter, double diff, double diff_filt)
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
            << diff << ", " << diff_filt <<"],\n";
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
  assert(order==2 || order==4 || order==6 || order==8);

  int nHalo = order / 2;
  double alpha_n   = 0;

    
  Storage3D<double> input(x, y, z, nHalo);
  input.initialize();
  Storage3D<double> output(x, y, z, nHalo);
  output.initialize();

  Storage3D<double> output_filt(x, y, z, nHalo);
  output_filt.initialize();
    
  std::ofstream fout;
  fout.open("in_field.dat", std::ios::binary | std::ofstream::trunc);
  input.writeFile(fout);
  fout.close();



#ifdef CRAYPAT          //performance analysis
  PAT_record(PAT_STATE_ON);
#endif
    
  auto start = std::chrono::steady_clock::now();
  auto start_filter = std::chrono::steady_clock::now();
  auto end = std::chrono::steady_clock::now(); 
  auto end_filter = std::chrono::steady_clock::now();       
    
  if (order == 2)
    {
      alpha_n = 1/32.;
      start = std::chrono::steady_clock::now();
      apply_2nd_diffusion(input, output, alpha_n, iter, x, y, z, nHalo);
      end = std::chrono::steady_clock::now();
    }
  else if (order == 4)
    {
      alpha_n = 1/32.;
    // take time for normal version
    start = std::chrono::steady_clock::now();
    simple_flux_limiter4(input, output_filt, alpha_n, iter, x, y, z, nHalo);
    end = std::chrono::steady_clock::now();
    
    // take time for flux limiter version
    start_filter = std::chrono::steady_clock::now();
    apply_4th_diffusion(input, output, alpha_n, iter, x, y, z, nHalo);
    end_filter = std::chrono::steady_clock::now();

  }
  else if (order == 6)
    {
      alpha_n = 1/256.;
      start = std::chrono::steady_clock::now();
      simple_flux_limiter6(input, output_filt, alpha_n, iter, x, y, z, nHalo);
      end = std::chrono::steady_clock::now();
      
      // take time for flux limiter version
      start_filter = std::chrono::steady_clock::now();
      apply_6th_diffusion(input, output, alpha_n, iter, x, y, z, nHalo);
      end_filter = std::chrono::steady_clock::now();
    }
  else if (order == 8)
    {
      alpha_n = 1/2048.;
      start = std::chrono::steady_clock::now();
      simple_flux_limiter8(input, output_filt, alpha_n, iter, x, y, z, nHalo);
      end = std::chrono::steady_clock::now();
      
      // take time for flux limiter version
      start_filter = std::chrono::steady_clock::now();
      apply_8th_diffusion(input, output, alpha_n, iter, x, y, z, nHalo);
      end_filter = std::chrono::steady_clock::now();
    }    
  
    
#ifdef CRAYPAT
  PAT_record(PAT_STATE_OFF);
#endif
  updateHalo(output);
  fout.open("out_field"  + std::to_string(order) + ".dat", std::ios::binary | std::ofstream::trunc);
  output.writeFile(fout);
  fout.close();

  // Save filtered output
  updateHalo(output_filt);
  fout.open("out_field_filtered"  + std::to_string(order) + ".dat", std::ios::binary | std::ofstream::trunc);
  output_filt.writeFile(fout);
  fout.close();

  auto diff = end - start;
  double timeDiff = std::chrono::duration<double, std::milli>(diff).count() / 1000.;
  auto diff_filter = end_filter - start_filter;
  double timeDiff_filter = std::chrono::duration<double, std::milli>(diff_filter).count() / 1000.;
  reportTime(output, iter, timeDiff, timeDiff_filter);
  std::cout << timeDiff << "," << timeDiff_filter << "\n" << std::endl;

  return 0;
}