#pragma once

#include <HPC4WC/field.h>
#include <flags.h>

#include <array>
#include <fstream>
#include <string>
#include <vector>

namespace HPC4WC {

/**
 * @brief Simple Argument parser.
 * 
 * Wrapper around https://github.com/sailormoon/flags.
 * 
 * The default arguments are:
 * - help: Shows the help screen.
 * - block-i: Whether to block in i direction (boolean)
 * - block-j: Whether to block in j direction (boolean)
 * - openmp-num-threads: How many threads should act in the k-direction.
 * - blocking-size-i: How big a block should be, i direction.
 * - blocking-size-j: How big a block should be, j direction.
 */
class ArgsParser {
public:
    /**
     * @brief Initialize the argument parser with the number of arguments and all given arguments.
     * @param[in] argc Number of entries in argv
     * @param[in] argv Array of arguments.
     * @param[in] default_args If true, also adds the configuration parameters (block_i etc.), default true.
     */
    ArgsParser(int argc, char* argv[], bool default_args = true);

    /**
     * @brief Default deconstructor
     */
    ~ArgsParser() {}

    /**
     * @brief Add an optional bool argument to the parser.
     * 
     * The default value is the value the variable result currently has.
     * @param[inout] result In: Default value, out: new value if the argument has been specified.
     * @param[in] argument The argument to search for.
     * @param[in] Helper string for the help display.
     */
    void add_argument(bool& result, const char* argument, const char* help);

    /**
     * @brief Add an optional Field::idx_t argument to the parser.
     * 
     * The default value is the value the variable result currently has.
     * @param[inout] result In: Default value, out: new value if the argument has been specified.
     * @param[in] argument The argument to search for.
     * @param[in] Helper string for the help display.
     */
    void add_argument(Field::idx_t& result, const char* argument, const char* help);

    /**
     * @brief Add an optional Field::idx_t argument to the parser.
     * 
     * The default value is the value the variable result currently has.
     * @param[inout] result In: Default value, out: new value if the argument has been specified.
     * @param[in] argument The argument to search for.
     * @param[in] Helper string for the help display.
     */
    void add_argument(std::string& result, const char* argument, const char* help);

    /**
     * @brief Checker to see if an executable should show the help.
     * @return True, if the help data should be shown.
     */
    bool help() const;

    /**
     * @brief Write the help menu to the stream provided.
     * @param[inout] stream Where to write the help data.
     */
    void help(std::ostream& stream) const;

private:
    flags::args m_args;  ///< internal argument parser

    const char* m_argv_0;  ///< the name of the program, used inside help()

    std::vector<std::array<std::string, 3>> m_arguments;       ///< user defined arguments
    std::vector<std::array<std::string, 3>> m_config_entries;  ///< config arguments (block_i etc.)
};

}  // namespace HPC4WC