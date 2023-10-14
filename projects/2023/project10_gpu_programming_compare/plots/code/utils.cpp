#include "utils.h"



void read_cmd_line_arguments(int argc, char *argv[], grid_info *grid,
			     unsigned int *num_iter, int *scan) {
    int set_nx = false;
    int set_ny = false;
    int set_nz = false;
    int set_num_iter = false;

    *scan = false;

    unsigned int iarg = 1;
    while (iarg < argc) {
	if (!strcmp(argv[iarg], "--nx")) {
	    error(iarg + 1 >= argc, "Missing value for -nx argument");
	    error(argv[iarg + 1][0] == '-', "Missing value for -nx argument");
	    grid->nx = atoi(argv[iarg + 1]);
	    set_nx = true;

	    ++iarg;
	} else if (!strcmp(argv[iarg], "--ny")) {
	    error(iarg + 1 >= argc, "Missing value for -ny argument");
	    error(argv[iarg + 1][0] == '-', "Missing value for -ny argument");
	    grid->ny = atoi(argv[iarg + 1]);
	    set_ny = true;

	    ++iarg;
	} else if (!strcmp(argv[iarg], "--nz")) {
	    error(iarg + 1 >= argc, "Missing value for -nz argument");
	    error(argv[iarg + 1][0] == '-', "Missing value for -nz argument");
	    grid->nz = atoi(argv[iarg + 1]);
	    set_nz = true;

	    ++iarg;
	} else if (!strcmp(argv[iarg], "--num_iter")) {
	    error(iarg + 1 >= argc, "Missing value for -num_iter argument");
	    error(argv[iarg + 1][0] == '-', "Missing value for -num_iter argument");
	    *num_iter = atoi(argv[iarg + 1]);
	    set_num_iter = true;

	    ++iarg;
	} else if (!strcmp(argv[iarg], "--scan")) {
	    *scan = true;
	} else {
	    const char *msg = "Unknown command line argument encountered: ";
	    char str[sizeof(argv[iarg]) + strlen(msg)];
	    strcpy(str, msg);
	    strcat(str, argv[iarg]);
	    error(true, str);
	}

	++iarg;
    }

    if (!scan) {
	error(!set_nx, "You have to specify nx");
	error(!set_ny, "You have to specify ny");
    }

    error(!set_nz, "You have to specify nz");
    error(!set_num_iter, "You have to specify num_iter");

    if (!scan) {
	error(grid->nx > 1024*1024, "Please provide a reasonable value of nx");
	error(grid->nx > 1024*1024, "Please provide a reasonable value of ny");
    }

    error(grid->nz > 1024, "Please provide a reasonable value of nz");
    error(*num_iter < 1 || *num_iter > 1024*1024, "Please provide a reasonable value of num_iter");
}



void write_field_to_file(float *field, grid_info *grid, const char *filename) {
    unsigned int w = grid->nx;
    unsigned int h = grid->ny;
    unsigned int d = grid->nz;
    unsigned int nh = grid->num_halo;

    w += 2 * nh;
    h += 2 * nh;
    
    FILE *fptr;
    fptr = fopen(filename, "wb");

    unsigned int dim = 3;
    unsigned int bits = sizeof(*field) * 8;
    
    fwrite(&dim, sizeof(unsigned int), 1, fptr);
    fwrite(&bits, sizeof(unsigned int), 1, fptr);
    fwrite(&(grid->num_halo), sizeof(unsigned int), 1, fptr);
    fwrite(&(w), sizeof(unsigned int), 1, fptr);
    fwrite(&(h), sizeof(unsigned int), 1, fptr);
    fwrite(&(d), sizeof(unsigned int), 1, fptr);
    fwrite(field, sizeof(float), w * h * d, fptr);

    fclose(fptr);
}



void write_timing_to_file(float msecs[], unsigned int samples, const char *filename) {
    FILE *fptr;
    fptr = fopen(filename, "w");

    fprintf(fptr, "msecs\n");

    for (unsigned int i = 0; i < samples; ++i) {
	fprintf(fptr, "%f\n", msecs[i]);
    }

    fclose(fptr);
}



void error(int yes, const char *msg) {
    if (yes) {
	fprintf(stderr, "FATAL PROGRAM ERROR!\n");
	fprintf(stderr, "%s\n", msg);
	fprintf(stderr, "Execution aborted...\n");
	exit(-1);
    }
}
