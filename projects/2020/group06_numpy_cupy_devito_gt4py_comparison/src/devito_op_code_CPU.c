#define _POSIX_C_SOURCE 200809L
#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"

struct dataobj
{
  void *restrict data;
  int * size;
  int * npsize;
  int * dsize;
  int * hsize;
  int * hofs;
  int * oofs;
} ;

struct profiler
{
  double section0;
  double section1;
  double section2;
} ;

void bf0(const float dt, const float h_x, const float h_y, const float h_z, struct dataobj *restrict u_vec, const int i1x0_blk0_size, const int i1y0_blk0_size, const int i1z_ltkn, const int i1z_rtkn, const int t0, const int t1, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m);

int Kernel(const float dt, const float h_x, const float h_y, const float h_z, struct dataobj *restrict u_vec, const int i1x0_blk0_size, const int i1x_ltkn, const int i1x_rtkn, const int i1y0_blk0_size, const int i1y_ltkn, const int i1y_rtkn, const int i1z_ltkn, const int i1z_rtkn, const int time_M, const int time_m, struct profiler * timers, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m)
{
  double (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (double (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  for (int time = time_m, t0 = (time)%(2), t1 = (time + 1)%(2); time <= time_M; time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))
  {
    struct timeval start_section0, end_section0;
    gettimeofday(&start_section0, NULL);
    /* Begin section0 */
    bf0(dt,h_x,h_y,h_z,u_vec,i1x0_blk0_size,i1y0_blk0_size,i1z_ltkn,i1z_rtkn,t0,t1,-i1x_rtkn + x_M - (-i1x_ltkn - i1x_rtkn + x_M - x_m + 1)%(i1x0_blk0_size),i1x_ltkn + x_m,-i1y_rtkn + y_M - (-i1y_ltkn - i1y_rtkn + y_M - y_m + 1)%(i1y0_blk0_size),i1y_ltkn + y_m,z_M,z_m);
    bf0(dt,h_x,h_y,h_z,u_vec,i1x0_blk0_size,(-i1y_ltkn - i1y_rtkn + y_M - y_m + 1)%(i1y0_blk0_size),i1z_ltkn,i1z_rtkn,t0,t1,-i1x_rtkn + x_M - (-i1x_ltkn - i1x_rtkn + x_M - x_m + 1)%(i1x0_blk0_size),i1x_ltkn + x_m,-i1y_rtkn + y_M,-i1y_rtkn + y_M - (-i1y_ltkn - i1y_rtkn + y_M - y_m + 1)%(i1y0_blk0_size) + 1,z_M,z_m);
    bf0(dt,h_x,h_y,h_z,u_vec,(-i1x_ltkn - i1x_rtkn + x_M - x_m + 1)%(i1x0_blk0_size),i1y0_blk0_size,i1z_ltkn,i1z_rtkn,t0,t1,-i1x_rtkn + x_M,-i1x_rtkn + x_M - (-i1x_ltkn - i1x_rtkn + x_M - x_m + 1)%(i1x0_blk0_size) + 1,-i1y_rtkn + y_M - (-i1y_ltkn - i1y_rtkn + y_M - y_m + 1)%(i1y0_blk0_size),i1y_ltkn + y_m,z_M,z_m);
    bf0(dt,h_x,h_y,h_z,u_vec,(-i1x_ltkn - i1x_rtkn + x_M - x_m + 1)%(i1x0_blk0_size),(-i1y_ltkn - i1y_rtkn + y_M - y_m + 1)%(i1y0_blk0_size),i1z_ltkn,i1z_rtkn,t0,t1,-i1x_rtkn + x_M,-i1x_rtkn + x_M - (-i1x_ltkn - i1x_rtkn + x_M - x_m + 1)%(i1x0_blk0_size) + 1,-i1y_rtkn + y_M,-i1y_rtkn + y_M - (-i1y_ltkn - i1y_rtkn + y_M - y_m + 1)%(i1y0_blk0_size) + 1,z_M,z_m);
    /* End section0 */
    gettimeofday(&end_section0, NULL);
    timers->section0 += (double)(end_section0.tv_sec-start_section0.tv_sec)+(double)(end_section0.tv_usec-start_section0.tv_usec)/1000000;
    struct timeval start_section1, end_section1;
    gettimeofday(&start_section1, NULL);
    /* Begin section1 */
    for (int y = y_m; y <= y_M; y += 1)
    {
      #pragma omp simd aligned(u:32)
      for (int z = z_m; z <= z_M; z += 1)
      {
        u[t1][2][y + 2][z + 2] = 1.00000000000000;
        u[t1][33][y + 2][z + 2] = 1.00000000000000;
      }
    }
    /* End section1 */
    gettimeofday(&end_section1, NULL);
    timers->section1 += (double)(end_section1.tv_sec-start_section1.tv_sec)+(double)(end_section1.tv_usec-start_section1.tv_usec)/1000000;
    struct timeval start_section2, end_section2;
    gettimeofday(&start_section2, NULL);
    /* Begin section2 */
    for (int x = x_m; x <= x_M; x += 1)
    {
      #pragma omp simd aligned(u:32)
      for (int z = z_m; z <= z_M; z += 1)
      {
        u[t1][x + 2][33][z + 2] = 1.00000000000000;
        u[t1][x + 2][2][z + 2] = 1.00000000000000;
      }
      #pragma omp simd aligned(u:32)
      for (int y = y_m; y <= y_M; y += 1)
      {
        u[t1][x + 2][y + 2][2] = 1.00000000000000;
        u[t1][x + 2][y + 2][33] = 1.00000000000000;
      }
    }
    /* End section2 */
    gettimeofday(&end_section2, NULL);
    timers->section2 += (double)(end_section2.tv_sec-start_section2.tv_sec)+(double)(end_section2.tv_usec-start_section2.tv_usec)/1000000;
  }
  return 0;
}

void bf0(const float dt, const float h_x, const float h_y, const float h_z, struct dataobj *restrict u_vec, const int i1x0_blk0_size, const int i1y0_blk0_size, const int i1z_ltkn, const int i1z_rtkn, const int t0, const int t1, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m)
{
  double (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (double (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;

  for (int i1x0_blk0 = x_m; i1x0_blk0 <= x_M; i1x0_blk0 += i1x0_blk0_size)
  {
    for (int i1y0_blk0 = y_m; i1y0_blk0 <= y_M; i1y0_blk0 += i1y0_blk0_size)
    {
      for (int i1x = i1x0_blk0; i1x <= i1x0_blk0 + i1x0_blk0_size - 1; i1x += 1)
      {
        for (int i1y = i1y0_blk0; i1y <= i1y0_blk0 + i1y0_blk0_size - 1; i1y += 1)
        {
          #pragma omp simd aligned(u:32)
          for (int i1z = i1z_ltkn + z_m; i1z <= -i1z_rtkn + z_M; i1z += 1)
          {
            double r3 = -2.0*u[t0][i1x + 2][i1y + 2][i1z + 2];
            u[t1][i1x + 2][i1y + 2][i1z + 2] = dt*((r3 + 1.0*(u[t0][i1x + 2][i1y + 2][i1z + 1] + u[t0][i1x + 2][i1y + 2][i1z + 3]))/((h_z*h_z)) + (r3 + 1.0*(u[t0][i1x + 2][i1y + 1][i1z + 2] + u[t0][i1x + 2][i1y + 3][i1z + 2]))/((h_y*h_y)) + (r3 + 1.0*(u[t0][i1x + 1][i1y + 2][i1z + 2] + u[t0][i1x + 3][i1y + 2][i1z + 2]))/((h_x*h_x)) + u[t0][i1x + 2][i1y + 2][i1z + 2]/dt);
          }
        }
      }
    }
  }
}

