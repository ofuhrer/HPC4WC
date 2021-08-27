#include <HPC4WC/autotuner.h>
#include <HPC4WC/timer.h>

#include <iostream>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>

#include <array>
#include <memory>
#endif

// Sources for Windows:
// https://stackoverflow.com/questions/35050381/execute-binary-from-c-without-shell/35050382
// https://stackoverflow.com/questions/42402673/createprocess-and-capture-stdout
// https://docs.microsoft.com/en-us/windows/win32/procthread/creating-a-child-process-with-redirected-input-and-output?redirectedfrom=MSDN
// https://stackoverflow.com/questions/50759946/waiting-on-a-handle-in-windows-thread

// Sources for Linux:
// http://www.microhowto.info/howto/capture_the_output_of_a_child_process_in_c.html
// https://stackoverflow.com/questions/478898/how-do-i-execute-a-command-and-get-the-output-of-the-command-within-c-using-po

#define BUFSIZE 1024
std::string CURRENT_OUTPUT;  ///< temporary variable for the std. output of the program executed.

#ifdef _WIN32
HANDLE m_hChildStd_OUT_Rd = NULL;
HANDLE m_hChildStd_OUT_Wr = NULL;
HANDLE m_hreadDataFromExtProgram = NULL;

DWORD __stdcall readDataFromExtProgram(void* argh) {
    DWORD dwRead;
    CHAR chBuf[BUFSIZE];
    BOOL bSuccess = FALSE;

    for (;;) {
        bSuccess = ReadFile(m_hChildStd_OUT_Rd, chBuf, BUFSIZE, &dwRead, NULL);
        if (!bSuccess || dwRead == 0)
            continue;

        CURRENT_OUTPUT = CURRENT_OUTPUT + chBuf;

        if (!bSuccess)
            break;
    }
    return 0;
}
HRESULT run_win_program(std::string externalProgram, std::string arguments, unsigned max_wait_millseconds) {
    STARTUPINFO si;
    PROCESS_INFORMATION pi;
    SECURITY_ATTRIBUTES saAttr;

    ZeroMemory(&saAttr, sizeof(saAttr));
    saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle = TRUE;
    saAttr.lpSecurityDescriptor = NULL;

    // Create a pipe for the child process's STDOUT.
    if (!CreatePipe(&m_hChildStd_OUT_Rd, &m_hChildStd_OUT_Wr, &saAttr, 0)) {
        // log error
        return HRESULT_FROM_WIN32(GetLastError());
    }

    // Ensure the read handle to the pipe for STDOUT is not inherited.
    if (!SetHandleInformation(m_hChildStd_OUT_Rd, HANDLE_FLAG_INHERIT, 0)) {
        // log error
        return HRESULT_FROM_WIN32(GetLastError());
    }

    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.hStdError = m_hChildStd_OUT_Wr;
    si.hStdOutput = m_hChildStd_OUT_Wr;
    si.dwFlags |= STARTF_USESTDHANDLES;

    ZeroMemory(&pi, sizeof(pi));

    std::string commandLine = externalProgram + " " + arguments;

    // Start the child process.
    if (!CreateProcessA(NULL,                         // No module name (use command line)
                        (TCHAR*)commandLine.c_str(),  // Command line
                        NULL,                         // Process handle not inheritable
                        NULL,                         // Thread handle not inheritable
                        TRUE,                         // Set handle inheritance
                        0,                            // No creation flags
                        NULL,                         // Use parent's environment block
                        NULL,                         // Use parent's starting directory
                        &si,                          // Pointer to STARTUPINFO structure
                        &pi)                          // Pointer to PROCESS_INFORMATION structure
    ) {
        return HRESULT_FROM_WIN32(GetLastError());
    } else {
        m_hreadDataFromExtProgram = CreateThread(0, 0, readDataFromExtProgram, NULL, 0, NULL);
    }
    // try to wait on pi
    WaitForSingleObject(pi.hProcess, max_wait_millseconds);

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    return S_OK;
}
#else
int run_posix_program(std::string externalProgram, std::string arguments) {
    std::array<char, BUFSIZE> buffer;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(std::string("./" + externalProgram + " " + arguments).c_str(), "r"), pclose);
    if (!pipe)
        return -1;
    try {
        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            CURRENT_OUTPUT = CURRENT_OUTPUT + buffer.data();
        }
    } catch (...) {
        return -1;
    }
    return 0;
}
#endif
namespace HPC4WC {

AutoTuner::AutoTuner(const char* exe_name, const char* exe_args, Field::const_idx_t& iterations) : m_exe_name(exe_name), m_exe_args(exe_args), m_iterations(iterations) {}

double AutoTuner::open_with_arguments(const std::string& arguments) const {
    CURRENT_OUTPUT = "";
    Timer timer;
#ifdef _WIN32
    HRESULT res = run_win_program(m_exe_name, arguments + " " + m_exe_args, m_max_wait_millseconds);
#else
    int res = run_posix_program(m_exe_name, arguments + " " + m_exe_args);
#endif
    if (res != 0) {
        throw std::runtime_error("Something did not work.");
    }
    return timer.timeElapsed();
}

void AutoTuner::add_bool_argument(const char* argument) {
    m_arguments.push_back({argument, {0, 1}});
}

void AutoTuner::add_range_argument(const char* argument, Field::const_idx_t& lower_bound, Field::const_idx_t& upper_bound, Field::const_idx_t& step_size) {
    m_arguments.push_back({argument, {}});
    for (Field::idx_t i = lower_bound; i <= upper_bound; i += step_size) {
        m_arguments.back().second.push_back(i);
    }
}

void AutoTuner::add_range_argument(const char* argument, const std::vector<Field::idx_t>& values) {
    m_arguments.push_back({argument, {}});
    for (const auto& val : values) {
        m_arguments.back().second.push_back(val);
    }
}

void AutoTuner::search() const {
    // source: https://gist.github.com/Yengas/9010715
    // search every permutation of m_arguments

    Field::idx_t search_space_size = 1;
    for (auto& arg : m_arguments) {
        search_space_size *= (Field::idx_t) arg.second.size();
    }
    std::cout << "Performing search over " << search_space_size << " different possibilities." << std::endl;

    std::vector<size_t> counters(m_arguments.size(), 0);

    std::string best_params;
    double best_time = 100000;

    for (size_t i = 0; i < search_space_size; i++) {
        std::string param = "";
        for (size_t p = 0; p < m_arguments.size(); p++) {
            param = param + "-" + m_arguments[p].first + " " + std::to_string(m_arguments[p].second[counters[p]]) + " ";
        }

        std::cout << param << std::endl;
        double total_time = 0;
        for (Field::idx_t t = 0; t < m_iterations; t++) {
            double time = open_with_arguments(param);
            total_time += time;
        }
        std::cout << "on average took " << total_time / m_iterations << " seconds" << std::endl;
        if (total_time / m_iterations < best_time) {
            best_time = total_time / m_iterations;
            best_params = param;
        }

        // do use a signed loop variable here because otherwise we have an integer underflow and then vector access error.
        for (long index = (long)counters.size() - 1; index >= 0; index--) {
            if (counters[index] + 1 < m_arguments[index].second.size()) {
                counters[index]++;
                break;
            }
            counters[index] = 0;
        }
    }

    std::cout << "==================" << std::endl;
    std::cout << "Search complete." << std::endl;
    std::cout << "Best parameters (" << best_time << "s): " << std::endl << best_params << std::endl;
}
}  // namespace HPC4WC