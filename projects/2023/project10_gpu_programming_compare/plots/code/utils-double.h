#include <cstring>
#include <cstdio>
#include <cstdlib>

#ifndef UTIL_H
#define UTIL_H

// 2D array access
#define A2D(x,y,w) ((x) + (y)*(w))

// 3D array access
#define A3D(x,y,z,w,h) ((x) + (y)*(w) + (z)*(w)*(h))

// Store grid dimensions
typedef struct grid_info_struct {
    unsigned int nx;
    unsigned int ny;
    unsigned int nz;
    unsigned int num_halo;
} grid_info;

void read_cmd_line_arguments(int argc, char *argv[], grid_info *grid,
			     unsigned int *num_iter, int *scan);

void write_field_to_file(double *field, grid_info *grid, const char *filename);

void write_timing_to_file(float msecs[], unsigned int samples, const char *filename);

void error(int yes, const char *msg);

#endif // UTIL_H
