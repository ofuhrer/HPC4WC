#pragma once
#include <HPC4WC/field.h>

#include <string>

namespace HPC4WC {
/**
 * @brief Searches parameter-space for best parameters.
 * 
 * Examples (with executables/autotune_search.cpp):
 * 
 * EXECUTE NORMAL PROGRAMS
 * Windows:
 * `.\autotune_search.exe --executable diffusion2d --exe-args="nk=10"`
 * Linux:
 * `./autotune_search --executable diffusion2d --exe-args="nk=10"`
 * 
 * EXECUTE MPI PROGRAMS where the search goes over the parameter "n" (passed to mpiexec)
 * The final call be will be something like "`mpiexec -n X mpi_diffusion --openmp-num-threads=2`", which means, that the OpenMP num threads argument will be used inside diffusion2d.
 * Windows:
 * `.\autotune_search.exe --executable mpiexec --exe-args="mpi_diffusion --openmp-num-threads=2"`
 * Linux:
 * `./autotune_search --executable mpiexec --exe-args="mpi_diffusion --openmp-num-threads=2"`
 */
class AutoTuner {
public:
    /**
     * @brief Create an autotuner, based on the executable and additional arguments passed to it.
     * @param[in] exe_name
     * @param[in] exe_args
     */
    AutoTuner(const char* exe_name, const char* exe_args, Field::const_idx_t& iterations=1);

    /**
     * @brief Default deconstructor
     */
    ~AutoTuner() {}
    
    /**
     * @brief Add a variable to search.
     * @param[in] argument The argument to test.
     * @param[in] lower_bound The lower bound of the search for this variable.
     * @param[in] upper_bound The upper bound (inclusive) of the search for this variable.
     * @param[in] step_size Stepsize of the search, default 1.
     * @attention The upper bound is inclusive!
     */
    void add_range_argument(const char* argument, Field::const_idx_t& lower_bound, Field::const_idx_t& upper_bound, Field::const_idx_t& step_size = 1);

    /**
     * @brief Add a variable to the search based on a vector of available values.
     * @param[in] argument The argument to test.
     * @param[in] values A vector of values which the argument can take.
     */
    void add_range_argument(const char* argument, const std::vector<Field::idx_t>& values);
    /**
     * @brief Helper to add a bool variable.
     * 
     * Is the same as using add_range_argument with a lower bound of 0 and an upper bound of 1.
     * @param[in] argument The argument to test.
     */
    void add_bool_argument(const char* argument);

    /**
     * @brief Search over all parameters.
     * 
     * Executes the given executable for each possible combination of parameters. Prints the time for each and finally
     * report the best set of parameters (which was the fastest).
     */
    void search() const;

private:
    /**
     * @brief Open the executable with given arguments.
     * @param[in] arguments The arguments to pass to the executable.
     * @return The runtime in seconds.
     */
    double open_with_arguments(const std::string& arguments) const;
    const char* m_exe_name; ///< the executable to be run
    const char* m_exe_args; ///< the additional arguments passed to the executable
    unsigned m_max_wait_millseconds = 30000;  ///< max wait time on windows, 30s
    Field::const_idx_t m_iterations; ///< how many times a certain set of parameters should be tested.

    /**
     * Argument with corresponding possibilities of values (currently, only integer values are allowed)
     */
    std::vector<std::pair<const char*, std::vector<Field::idx_t>>> m_arguments;
};
}  // namespace HPC4WC