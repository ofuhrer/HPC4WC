#ifndef WALLTIME
#define WALLTIME

#ifdef GETTIMEOFDAY
#include <sys/time.h> // For struct timeval, gettimeofday
#else
#include <time.h> // For struct timespec, clock_gettime, CLOCK_MONOTONIC
#endif
# include <stdio.h>

double wall_time ();

#endif