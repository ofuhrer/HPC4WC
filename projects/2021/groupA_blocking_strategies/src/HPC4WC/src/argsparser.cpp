#include <HPC4WC/argsparser.h>
#include <HPC4WC/config.h>

#ifdef _WIN32
#define PATH_CHAR "\\"
#else
#define PATH_CHAR "/"
#endif
#define PARSE_EXE_NAME(name) \
    std::string(name).substr(std::string(name).find_last_of(PATH_CHAR) + 1, std::string(name).size() - std::string(name).find_last_of(PATH_CHAR) - 1)

namespace HPC4WC {

ArgsParser::ArgsParser(int argc, char* argv[], bool default_args) : m_args(argc, argv), m_argv_0(argv[0]) {
    // Add config entries here.
    m_arguments.push_back({"help", "Shows this help.", "false"});

    if (default_args) {
        m_config_entries.push_back({"block-i", "Perform i-blocking.", Config::BLOCK_I ? "true" : "false"});
        m_config_entries.push_back({"block-j", "Perform j-blocking.", Config::BLOCK_J ? "true" : "false"});
        m_config_entries.push_back({"openmp-num-threads", "How many threads should be used for OpenMP.", std::to_string(Config::OPENMP_NUM_THREADS)});
        m_config_entries.push_back({"blocking-size-i", "How big the block should be, i direction.", std::to_string(Config::BLOCK_SIZE_I)});
        m_config_entries.push_back({"blocking-size-j", "How big the block should be, j direction.", std::to_string(Config::BLOCK_SIZE_J)});

        Config::BLOCK_I = m_args.get<bool>("block-i").value_or(Config::BLOCK_I);
        Config::BLOCK_J = m_args.get<bool>("block-j").value_or(Config::BLOCK_J);
        Config::OPENMP_NUM_THREADS = m_args.get<Field::idx_t>("openmp-num-threads").value_or(Config::OPENMP_NUM_THREADS);
        Config::BLOCK_SIZE_I = m_args.get<Field::idx_t>("blocking-size-i").value_or(Config::BLOCK_SIZE_I);
        Config::BLOCK_SIZE_J = m_args.get<Field::idx_t>("blocking-size-j").value_or(Config::BLOCK_SIZE_J);
    }
}

bool ArgsParser::help() const {
    return m_args.get<bool>("help", false);
}

void ArgsParser::help(std::ostream& stream) const {
    stream << PARSE_EXE_NAME(m_argv_0);
    for (auto& tr : m_arguments) {
        stream << " [--" << std::get<0>(tr) << "=" << std::get<2>(tr) << "]";
    }
    stream << std::endl;
    for (auto& tr : m_arguments) {
        stream << "  --" << std::get<0>(tr) << "  " << std::get<1>(tr) << " (default: " << std::get<2>(tr) << ")" << std::endl;
    }

    stream << std::endl;

    if (m_config_entries.size() > 0) {
        stream << "Further Configurations:" << std::endl;
        for (auto& tr : m_config_entries) {
            stream << "  --" << std::get<0>(tr) << "  " << std::get<1>(tr) << " (default: " << std::get<2>(tr) << ")" << std::endl;
        }
    }
}

void ArgsParser::add_argument(Field::idx_t& result, const char* argument, const char* help) {
    m_arguments.push_back({argument, help, std::to_string(result)});
    result = m_args.get<Field::idx_t>(argument).value_or(result);
}

void ArgsParser::add_argument(bool& result, const char* argument, const char* help) {
    m_arguments.push_back({argument, help, result ? "true" : "false"});
    result = m_args.get<Field::idx_t>(argument).value_or(result);
}

void ArgsParser::add_argument(std::string& result, const char* argument, const char* help) {
    m_arguments.push_back({argument, help, result});
    result = m_args.get<std::string>(argument).value_or(result);
}

}  // namespace HPC4WC