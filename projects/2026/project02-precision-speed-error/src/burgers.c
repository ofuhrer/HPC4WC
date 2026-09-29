  // This is a C script for computing 2D Burgers equation on 3D field at user-defined 
// precision, with a default wavy initial condition and periodic boundary condition.
// It should be compiled with "gcc -DFLOATXX={16/32/64} -std=c17 -Wfatal-errors 
// burgers.c -o burgers.out -lm" and executed using "./burgers.out {nx} {nz} {output mode}".

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/stat.h>

// Select data type ////////////////////////////////////////////////////////////////////
#ifndef FLOATXX
    #define FLOATXX 32
    #warning "FLOATXX is not defined. Default to 32."
#endif

#if FLOATXX == 16
    typedef _Float16  _FloatXX;
    #define LITERALS(val) val##f16
    #define MATHPREC(fn)  fn##f      //the lowest math function preciseness
#elif FLOATXX == 32
    typedef _Float32  _FloatXX;
    #define LITERALS(val) val##f32
    #define MATHPREC(fn)  fn##f      //as for float
#elif FLOATXX == 64
    typedef _Float64  _FloatXX;
    #define LITERALS(val) val##f64
    #define MATHPREC(fn)  fn         //as for double
#else
    #error "FLOATXX must be either 16, 32, or 64."
#endif

// Define functions ////////////////////////////////////////////////////////////////////
typedef struct {
    int      nx;     //the computational domain
    int      ny;
    int      nz;
    int      nHalo;  //number of halo points (default to 1)
    int      nx_;    //with halo points
    int      ny_;
    _FloatXX dx;     //horizontal resolution
    _FloatXX dy;
    _FloatXX inv_dx;
    _FloatXX inv_dy;
    _FloatXX dt;     //time step (defined dynamically)
    double   t;      //current time
    _FloatXX *u;     //current u and v field
    _FloatXX *v;
    _FloatXX *u_old; //u and v field in the last time step
    _FloatXX *v_old;
} field;

void write_2Dfields(field *f,
                    int   k){ //index of the z level to write
    int   nx_                     = f->nx_;
    int   ny_                     = f->ny_;
    const _FloatXX (*u)[ny_][nx_] = (const void*)f->u;
    const _FloatXX (*v)[ny_][nx_] = (const void*)f->v;
    char  fname_u[100], fname_v[100], directory[100];

    
    
    snprintf(directory, sizeof(directory), 
             "./output/n%05d_f%03d/",
             f->nx, FLOATXX);
    snprintf(fname_u, sizeof(fname_u), 
             "./output/n%05d_f%03d/u_t%03d.bin",
             f->nx, FLOATXX, (int)(f->t*100));
    snprintf(fname_v, sizeof(fname_v), 
             "./output/n%05d_f%03d/v_t%03d.bin",
             f->nx, FLOATXX, (int)(f->t*100));

    mkdir("output", 0755);
    mkdir(directory, 0755);
    
    //1) writing as binary file
    FILE *file_u = fopen(fname_u, "wb");
    if (file_u == NULL) {
        printf("Failed to write to file!\n");
        abort();
    }
    FILE *file_v = fopen(fname_v, "wb");
    if (file_v == NULL) {
        printf("Failed to write to file!\n");
        abort();
    }
    for (int j = 0; j < f->ny; j++) {
        fwrite(&u[k][j + f->nHalo][f->nHalo], sizeof(_FloatXX), f->nx, file_u);
        fwrite(&v[k][j + f->nHalo][f->nHalo], sizeof(_FloatXX), f->nx, file_v);
    }
    
    fclose(file_u);
    fclose(file_v);
}

void update_halo(field *f){
    int      nx_            = f->nx_;
    int      ny_            = f->ny_;
    _FloatXX (*u)[ny_][nx_] = (void*)f->u;
    _FloatXX (*v)[ny_][nx_] = (void*)f->v;
    #pragma omp parallel for collapse(3) schedule(static)
    for (int k = 0; k < f->nz; k++) {
        for (int j = 0; j < f->nHalo; j++) {
            for (int i = f->nHalo; i < f->nx+f->nHalo; i++) {
                u[k][j               ][i] = u[k][j+f->ny   ][i]; //bottom without corners
                v[k][j               ][i] = v[k][j+f->ny   ][i];
                u[k][j+f->ny+f->nHalo][i] = u[k][j+f->nHalo][i]; //top without corners
                v[k][j+f->ny+f->nHalo][i] = v[k][j+f->nHalo][i];
            }
        }
    }
    #pragma omp parallel for collapse(3) schedule(static)
    for (int k = 0; k < f->nz; k++) {
        for (int j = 0; j < f->ny_; j++){
            for (int i = 0; i < f->nHalo; i++){
                u[k][j][i               ] = u[k][j][i+f->nx   ]; //left with corners
                v[k][j][i               ] = v[k][j][i+f->nx   ];
                u[k][j][i+f->nx+f->nHalo] = u[k][j][i+f->nHalo]; //right with corners
                v[k][j][i+f->nx+f->nHalo] = v[k][j][i+f->nHalo];
            }
        }
    }
}

void one_burgers_step(field   *f, const _FloatXX Re) {
    int            nx_                = f->nx_;
    int            ny_                = f->ny_;
    const _FloatXX (*u_old)[ny_][nx_] = (const void*)f->u_old;
    const _FloatXX (*v_old)[ny_][nx_] = (const void*)f->v_old;
    _FloatXX       (*u    )[ny_][nx_] = (void*      )f->u;
    _FloatXX       (*v    )[ny_][nx_] = (void*      )f->v;
    const _FloatXX inv_Re = 1.0 / Re;
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < f->nz; k++) {  // nz
        for (int j = f->nHalo; j < f->ny+f->nHalo; j++) {  // ny
            _FloatXX dudx, dudy, dvdx, dvdy, lapu, lapv;
            #pragma omp simd
            for (int i = f->nHalo; i < f->nx+f->nHalo; i++) {  // nx
                if (u_old[k][j][i]>0) {
                    dudx = (u_old[k][j][i]-u_old[k][j][i-1])*f->inv_dx;  // 2 flops
                    dvdx = (v_old[k][j][i]-v_old[k][j][i-1])*f->inv_dx;  // 2 flops
                }
                else {
                    dudx = (u_old[k][j][i+1]-u_old[k][j][i])*f->inv_dx;  // 2 flops
                    dvdx = (v_old[k][j][i+1]-v_old[k][j][i])*f->inv_dx;  // 2 flops
                }
                if (v_old[k][j][i]>0) {
                    dudy = (u_old[k][j][i]-u_old[k][j-1][i])*f->inv_dy;  // 2 flops
                    dvdy = (v_old[k][j][i]-v_old[k][j-1][i])*f->inv_dy;  // 2 flops
                }
                else {
                    dudy = (u_old[k][j+1][i]-u_old[k][j][i])*f->inv_dy;  // 2 flops
                    dvdy = (v_old[k][j+1][i]-v_old[k][j][i])*f->inv_dy;  // 2 flops
                }
                lapu       = (u_old[k][j][i+1]-2*u_old[k][j][i]+u_old[k][j][i-1])*f->inv_dx*f->inv_dx+
                             (u_old[k][j+1][i]-2*u_old[k][j][i]+u_old[k][j-1][i])*f->inv_dy*f->inv_dy;  // 11 flops
                lapv       = (v_old[k][j][i+1]-2*v_old[k][j][i]+v_old[k][j][i-1])*f->inv_dx*f->inv_dx+
                             (v_old[k][j+1][i]-2*v_old[k][j][i]+v_old[k][j-1][i])*f->inv_dy*f->inv_dy;  // 11 flops
                u[k][j][i] = u_old[k][j][i] + f->dt*
                             (-u_old[k][j][i]*dudx - v_old[k][j][i]*dudy + lapu * inv_Re);  // 7 flops
                v[k][j][i] = v_old[k][j][i] + f->dt*
                             (-u_old[k][j][i]*dvdx - v_old[k][j][i]*dvdy + lapv * inv_Re);  // 7 flops
            }
        }
    }
}

// Main program ////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]){
    //-------------------------------- Set domain size ---------------------------------
    int nx, ny, nz, doWrite;
    int invalid = (argc!=4);

    if (!invalid) { 
        nx      = strtol(argv[1], NULL, 10);
        ny      = nx;
        nz      = strtol(argv[2], NULL, 10);
        doWrite = strtol(argv[3], NULL, 10); // 0: no output, 1: output at beginning and end, 2: output at out_freq
        invalid = (nx<2 || nz<1 || doWrite>2);
    } 
    if (invalid) {
        printf(
            "Missing or invalid arguments. Running with default domain sizes and no output.\n"
            "Arguments required [nx] [nz] [output mode]\n"
            "output mode:\n"
            " - 0: no output\n"
            " - 1: only output initial and final fields\n"
            " - 2: output at fixed frequency\n"
        );
        nx      = 10;
        ny      = nx; 
        nz      = 10;
        doWrite = 0;
    }
    printf("nx = %d, ny = %d, nz = %d\n", nx, ny, nz);

    
    //--------------------------------- Initialization ---------------------------------
    //Create the `field` structure
    field f;
    f.nx              = nx;
    f.ny              = ny;
    f.nz              = nz;
    f.nHalo           = 1;
    f.nx_             = f.nx+2*f.nHalo;
    f.ny_             = f.ny+2*f.nHalo;
    int ny_           = f.ny_;
    int nx_           = f.nx_;
    f.dx              = LITERALS(1.)/nx; 
    f.dy              = LITERALS(1.)/ny; 
    f.inv_dx          = nx; 
    f.inv_dy          = ny; 
    f.t               = LITERALS(0.);
    size_t s          = sizeof(_FloatXX[nz][ny_][nx_]);
    f.u               = malloc(s);
    f.v               = malloc(s);
    f.u_old           = malloc(s);
    f.v_old           = malloc(s);
    //Everything in double to avoid misrepresentation errors
    double dx_d       = (double)f.dx;
    double dy_d       = (double)f.dy;
    double Re_d       = (double)LITERALS(10000.);
    double Vmax_d     = 1.0;
    double dt_cfl_d   = 0.5 / (Vmax_d/dx_d + Vmax_d/dy_d);
    double dt_diff_d  = 0.5 * Re_d / (1.0/(dx_d*dx_d) + 1.0/(dy_d*dy_d));
    _FloatXX pi       = LITERALS(3.14159265358979323846264338327950288);
    _FloatXX Re       = LITERALS(10000.); //Reynolds number
    double tmax       = doWrite == 2 ? 0.3 : 1.0;
    _FloatXX Vmax     = LITERALS(1.);     //expected maximum velocity throughout the simulation
    _FloatXX dt_cfl   = (_FloatXX)dt_cfl_d;
    _FloatXX dt_diff  = (_FloatXX)dt_diff_d;
    double dt_base  = (dt_cfl < dt_diff) ? dt_cfl : dt_diff;
    double out_freq = 0.01;               //frequency to output a 2D field
    double tout     = f.t+out_freq;
    _FloatXX *tmp     = f.u;              //temporary variable for swapping 
    int      out      = 0;
    
    //Create 3D variable length array (VLA) pointers for easy referencce to array elements
    _FloatXX (*u)[ny_][nx_] = (void*)f.u; 
    _FloatXX (*v)[ny_][nx_] = (void*)f.v;

    //Impose the initial condition (only function of x and y so same for all z levels)
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                u[k][j+f.nHalo][i+f.nHalo] = MATHPREC(sin)(i*f.dx*pi*2)*MATHPREC(cos)(j*f.dy*pi*2);
                v[k][j+f.nHalo][i+f.nHalo] = MATHPREC(cos)(i*f.dx*pi*2)*MATHPREC(sin)(j*f.dy*pi*2);
            }
        }
    }
    update_halo(&f);
    if (doWrite)
        write_2Dfields(&f, 0); //at the lowest z level
    
    //---------------------------------- Computation -----------------------------------
    double tic = omp_get_wtime();
    long int n_steps = 0;
    while (f.t < tmax) {
        out    = (f.t+dt_base)>=tout;             //check if need to output the 2D field
        double dt = out ? tout-f.t : dt_base;     //compute the next time step, in double
        f.t   += dt;
        f.dt   = (_FloatXX)dt;                    //cast down only for the physics kernel

        tmp  = f.u_old; f.u_old = f.u; f.u = tmp;
        tmp  = f.v_old; f.v_old = f.v; f.v = tmp;
        one_burgers_step(&f, Re);
        update_halo(&f);
        if (out) {
            if (doWrite == 2)
                write_2Dfields(&f, 0);
            tout += out_freq;
        }
        n_steps++;
    }
    double toc = omp_get_wtime();
    if (doWrite == 1)
        write_2Dfields(&f, 0);
    //----------------------------------- Finishing ------------------------------------

    free(f.u    );
    free(f.v    );
    free(f.u_old);
    free(f.v_old);
    printf("Time elapsed =  %.8f s\n", toc-tic);
    return 0;
}
