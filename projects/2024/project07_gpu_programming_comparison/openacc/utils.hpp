#pragma once

#include <iostream>


#ifdef _OPENACC
extern "C" {
    extern void acc_set_error_routine(void (*)(char *));
}
#endif


namespace {

// Translates a 3D index to a 1D/linear index.
static inline std::size_t index(std::size_t i, std::size_t j, std::size_t k,
                                std::size_t xsize, std::size_t ysize) {
    return i + j * xsize + k * xsize * ysize;
}


#ifdef _OPENACC
// Called by the OpenACC runtime in case of an error.
void error_routine(char *errmsg) {
    std::cerr << "ERROR: An OpenACC error has occurred.\n"
              << "Error message: \"" << errmsg << "\"\n";
    std::cerr << "================================================================================\n";
    exit(EXIT_FAILURE);
}
#endif

} // namespace


enum class Mode {
    kernels,
    parallel,
    invalid
};


namespace utils {

// Prints an explanation of the input syntax. Called when invalid input is detected.
void print_args_errmsg() {
    std::cerr << "================================================================================\n";
    std::cerr << "                             Welcome to stencil2d!\n";
    std::cerr << " nx  :: Amount of (interior) points in x-direction. Must be >0.\n";
    std::cerr << " ny  :: Amount of (interior) points in y-direction. Must be >0.\n";
    std::cerr << " nz  :: Amount of (interior) points in z-direction. Must be >0.\n";
    std::cerr << "bdry :: Boundary width. Must be >1.\n";
    std::cerr << "itrs :: Number of diffusive timesteps to perform. Must be >0.\n";
    std::cerr << "mode :: Computation mode. Must be \"kernels\" or \"parallel\".\n";
    std::cerr << "================================================================================\n";
    std::cerr << "Input syntax: ./main <nx> <ny> <nz> <bdry> <itrs> <mode>\n";
    std::cerr << "================================================================================\n";
}


// Translates an input string to an acceleration mode.
Mode mode_from_string(const char *s) {
    std::string mode(s);
    if(mode == "kernels") return Mode::kernels;
    if(mode == "parallel") return Mode::parallel;
    return Mode::invalid;
}


// Returns a brief description of a given acceleration mode.
std::string get_mode_desc(Mode mode) {
    switch(mode) {
        case Mode::kernels: return "OpenACC acceleration using only \"kernels\" pragmas.";
        case Mode::parallel: return "OpenACC acceleration using only \"parallel loop collapse\" pragmas.";
        default: __builtin_unreachable();
    }
}

} // namespace utils
