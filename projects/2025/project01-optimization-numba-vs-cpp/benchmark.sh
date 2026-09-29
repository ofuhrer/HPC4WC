#!/bin/bash

# Default values for arguments
nx="1500"
max_iter="10"
repetitions="1"
measure_scaling="false" # Controls scaling: 'false', 'strong', 'weak'
omp_max_threads="" # Renamed: Max OMP threads for scaling or fixed value
use_srun="false"     # Feature flag to enable/disable srun

# Global variables to store detailed CPU topology
num_sockets=""
physical_cores_per_socket=""
total_logical_cpus=""
thread_per_core=""

# Function to display help message
display_help() {
    echo "Usage: $0 [NX] [MAX_ITER] [REPETITIONS] [MEASURE_SCALING] [OMP_MAX_THREADS] [USE_SRUN]"
    echo ""
    echo "This script runs C++ and Python programs located in specific directories."
    echo "It can optionally measure strong or weak scaling across different OMP_NUM_THREADS values."
    echo "It can also use srun for parallel execution if specific conditions are met."
    echo ""
    echo "Arguments:"
    echo "  NX                     (Optional) The initial value for nx and ny. Default is $nx."
    echo "                         If MEASURE_SCALING is 'weak', this NX will be the base for 1 thread."
    echo "  MAX_ITER               (Optional) The maximum number of iterations. Default is $max_iter."
    echo "  REPETITIONS            (Optional) The number of times to run each program. Default is $repetitions."
    echo "  MEASURE_SCALING        (Optional) Set to 'false', 'strong', or 'weak' to control OMP_NUM_THREADS scaling."
    echo "                         - 'false': OMP_NUM_THREADS will not be explicitly set by the script,"
    echo "                           UNLESS OMP_MAX_THREADS is provided (sets a fixed value)."
    echo "                         - 'strong': Enables strong scaling. OMP_NUM_THREADS uses powers of two (1, 2, 4,...)"
    echo "                           up to the specified or auto-detected max. Problem size (NX, NY) remains fixed."
    echo "                         - 'weak': Enables weak scaling. OMP_NUM_THREADS uses powers of two (1, 2, 4,...)"
    echo "                           up to the specified or auto-detected max. Problem size (NX, NY) scales up by"
    echo "                           a factor of sqrt(number_of_threads), rounded to the nearest 100. For example,"
    echo "                           if initial NX=1000, for 2 threads, NX becomes approx. 1400; for 4 threads, NX becomes 2000."
    echo "  OMP_MAX_THREADS        (Optional) Numeric value for OMP_NUM_THREADS configuration."
    echo "                         - If MEASURE_SCALING is 'strong' or 'weak': This sets the maximum OMP_NUM_THREADS"
    echo "                           value to use (e.g., 16). If not provided, the script will auto-detect the total"
    echo "                           number of logical CPU cores on the system."
    echo "                         - If MEASURE_SCALING is 'false': This sets a fixed OMP_NUM_THREADS value for all runs (e.g., 8)."
    echo "                                                         If system has e.g. 16 logical cores, and USE_SRUN is true,"
    echo "                                                         'srun -n 2' might be used to fully utilize cores."
    echo "  USE_SRUN               (Optional) Set to 'true' to enable the usage of 'srun' for launching processes."
    echo "                         If 'false' (default), only a single instance of the program will run,"
    echo "                         regardless of OMP_NUM_THREADS or total CPU cores, and MPI will not be used."
    echo ""
    echo "Example:"
    echo "  $0                                       # Runs with default values (nx=1500, max_iter=10, repetitions=1)"
    echo "  $0 2000                                  # Sets nx=2000, others default"
    echo "  $0 2000 50                               # Sets nx=2000, max_iter=50, repetitions=1"
    echo "  $0 2000 50 3                             # Sets nx=2000, max_iter=50, repetitions=3"
    echo "  $0 2000 50 1 strong                      # Enables strong scaling, auto-detects max threads (total logical CPU cores)"
    echo "  $0 2000 50 1 strong 16                   # Enables strong scaling, sets max threads for scaling to 16"
    echo "  $0 1000 50 1 weak                        # Enables weak scaling, initial NX=1000, auto-detects max threads"
    echo "  $0 2000 50 1 false 8                     # Disables scaling, sets OMP_NUM_THREADS to 8 per process."
    echo "                                                 # If system has e.g. 16 cores, 'srun -n 2' might be used."
    echo "  $0 2000 50 1 false 8 true                # Same as above, but explicitly enables srun."
    echo "  $0 2000 50 1 false "" false              # Disables scaling, no specific OMP_NUM_THREADS, disables srun (default)."
    echo "  $0 --help                                # Displays this help message"
    exit 0
}

# Check for --help flag
if [[ "$1" == "--help" ]]; then
    display_help
fi

# Parse arguments
if [ -n "$1" ]; then
    nx="$1"
fi

if [ -n "$2" ]; then
    max_iter="$2"
fi

if [ -n "$3" ]; then
    repetitions="$3"
fi

# Check if the 4th argument (MEASURE_SCALING) is provided and is 'strong', 'weak', or 'false'
if [ -n "$4" ]; then
    case "$4" in
        "strong"|"weak")
            echo "MEASURE_SCALING is ENABLED ($4)"
            measure_scaling="$4"
            ;;
        "false")
            echo "MEASURE_SCALING is DISABLED"
            measure_scaling="false"
            ;;
        *)
            echo "Warning: Invalid MEASURE_SCALING value '$4'. Must be 'false', 'strong', or 'weak'. Defaulting to 'false'."
            measure_scaling="false"
            ;;
    esac
fi

# Check if the 5th argument (OMP_MAX_THREADS) is provided
if [ -n "$5" ]; then
    if [[ "$5" =~ ^[0-9]+$ ]] && [ "$5" -gt 0 ]; then
        omp_max_threads="$5"
    else
        echo "Warning: Invalid OMP_MAX_THREADS '$5'. Must be a positive integer. Ignoring explicit value."
    fi
fi

# Check if the 6th argument (USE_SRUN) is provided and is 'true'
if [ -n "$6" ] && [ "$6" == "true" ]; then
    use_srun="true"
fi

# print hardware info (cpu, memory speed and amount, gpu)
echo ""
echo "Hardware Information:"
if command -v lscpu &> /dev/null; then
    echo "  CPU Information:"
    lscpu | grep "Model name:" # Get CPU model name
    lscpu
fi

if command -v free &> /dev/null; then
    echo "  Memory Information:"
    free -h
fi

if command -v lspci &> /dev/null; then
    echo "  GPU Information:"
    lspci | grep -i nvidia
fi
echo ""
echo ""
echo ./CPU/cpp/include/macros.hpp
cat ./CPU/cpp/include/macros.hpp
echo ""
echo ""

# Detect detailed CPU topology
if command -v lscpu &> /dev/null; then
    num_sockets=$(lscpu | grep "Socket(s):" | awk '{print $NF}')
    physical_cores_per_socket=$(lscpu | grep "Core(s) per socket:" | awk '{print $NF}')
    # Get total logical CPUs (threads) directly from "CPU(s):" line, which is more reliable
    total_logical_cpus=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
    thread_per_core=$(lscpu | grep "Thread(s) per core:" | awk '{print $NF}')
elif command -v nproc &> /dev/null; then
    # Fallback to nproc if lscpu is not available
    total_logical_cpus=$(nproc)
    num_sockets="1" # Assume single socket
    physical_cores_per_socket=$(nproc) # Assume all are physical cores per socket
    thread_per_core="1" # Assume no hyperthreading
fi


# Fallback/Default values if previous detection fails or values are empty/zero
if [ -z "$num_sockets" ] || [ "$num_sockets" -eq 0 ]; then num_sockets="1"; fi
if [ -z "$physical_cores_per_socket" ] || [ "$physical_cores_per_socket" -eq 0 ]; then physical_cores_per_socket="8"; fi # Default physical cores per socket
if [ -z "$total_logical_cpus" ] || [ "$total_logical_cpus" -eq 0 ]; then total_logical_cpus=$((num_sockets * physical_cores_per_socket * thread_per_core)); fi # Default total logical CPUs
if [ -z "$thread_per_core" ] || [ "$thread_per_core" -eq 0 ]; then thread_per_core="1"; fi

echo "Hardware CPU Topology:"
echo "  Sockets: $num_sockets"
echo "  Physical Cores per Socket: $physical_cores_per_socket"
echo "  Threads per Core: $thread_per_core"
echo "  Total Logical CPUs (Threads) on system: $total_logical_cpus"
echo ""

# Determine the final OMP thread configuration for the run
omp_config_val_for_run=""
if [ "$measure_scaling" == "strong" ] || [ "$measure_scaling" == "weak" ]; then
    if [ -n "$omp_max_threads" ]; then
        # Use user-specified max threads for scaling, capped by total logical CPUs
        omp_config_val_for_run=$(( omp_max_threads > total_logical_cpus ? total_logical_cpus : omp_max_threads ))
    else
        # If scaling is enabled and no explicit max is given, scale up to all logical CPUs
        omp_config_val_for_run="$total_logical_cpus"
    fi
else
    # If scaling is disabled, use fixed value or nothing
    omp_config_val_for_run="$omp_max_threads"
    # Cap fixed OMP_NUM_THREADS to total logical CPUs to prevent oversubscription
    if [ -n "$omp_config_val_for_run" ] && [ "$omp_config_val_for_run" -gt "$total_logical_cpus" ]; then
        echo "  Warning: Provided OMP_NUM_THREADS ($omp_config_val_for_run) is greater than total detected logical CPUs ($total_logical_cpus)."
        echo "  It will be capped to $total_logical_cpus for execution to avoid oversubscription issues."
        omp_config_val_for_run="$total_logical_cpus"
    fi
fi

echo "Starting to run programs with arguments: initial nx=ny=$nx, max_iter=$max_iter, repetitions=$repetitions"
if [ "$measure_scaling" == "strong" ] || [ "$measure_scaling" == "weak" ]; then
    echo "MEASURE_SCALING is ENABLED ($measure_scaling) up to $omp_config_val_for_run threads."
    if [ "$use_srun" == "true" ]; then
        echo "  srun/MPI is ENABLED. The script will attempt to use 'srun' to launch multiple instances for full core utilization"
        echo "  when OMP_NUM_THREADS values evenly divide the total logical CPUs. ⚠️ Requires MPI-enabled programs."
    else
        echo "  srun/MPI is DISABLED. Only a single instance will be launched regardless of core utilization."
    fi
elif [ -n "$omp_config_val_for_run" ]; then
    echo "MEASURE_SCALING is DISABLED. Programs will run with OMP_NUM_THREADS=$omp_config_val_for_run per process."
    if [ "$use_srun" == "true" ]; then
        if [ "$omp_config_val_for_run" -lt "$total_logical_cpus" ] && [ "$(( total_logical_cpus % omp_config_val_for_run ))" -eq 0 ]; then
            echo "  srun/MPI is ENABLED. Since OMP_NUM_THREADS ($omp_config_val_for_run) is less than total logical CPUs ($total_logical_cpus) and divides it evenly,"
            echo "  the script will attempt to use 'srun' to launch multiple instances to fully utilize cores."
            echo "  ⚠️ IMPORTANT: This requires the C++ and Python programs to be MPI-enabled for correct behavior."
        elif [ "$omp_config_val_for_run" -gt "$total_logical_cpus" ]; then # This block is now mostly redundant due to capping above, but good for clarity.
            echo "  Warning: Provided OMP_NUM_THREADS ($omp_config_val_for_run) is greater than total detected logical CPUs ($total_logical_cpus)."
            echo "  It has been capped to $total_logical_cpus for execution to avoid oversubscription issues."
        fi
    else
        echo "  srun/MPI is DISABLED. Only a single instance will be launched."
    fi
else
    echo "MEASURE_SCALING is DISABLED. OMP_NUM_THREADS will NOT be explicitly set by the script."
    if [ "$use_srun" == "true" ]; then
        echo "  srun/MPI is ENABLED."
    else
        echo "  srun/MPI is DISABLED. Only a single instance will be launched."
    fi
fi
echo ""

# Function to run a program with OMP_NUM_THREADS and potentially MPI configuration
run_program_with_scaling() {
    local program_path="$1"
    local program_type="$2"
    local initial_nx="$3"
    local current_max_iter="$4"
    local current_repetitions="$5"
    local measure_scaling_type="$6"
    local specific_omp_val_for_logic="$7" # This is the max threads to consider for scaling or fixed value
    local system_num_sockets="$8"
    local system_physical_cores_per_socket="$9"
    local system_total_logical_cpus="${10}"
    local srun_enabled_flag="${11}"

    if [ "$measure_scaling_type" == "strong" ] || [ "$measure_scaling_type" == "weak" ]; then
        local max_threads_to_test=$(( specific_omp_val_for_logic > system_total_logical_cpus ? system_total_logical_cpus : specific_omp_val_for_logic ))
        if [ "$max_threads_to_test" -eq 0 ]; then max_threads_to_test=1; fi # Ensure max_threads_to_test is at least 1

        local omp_thread_sequence=()
        local i=1

        # Phase 1: Powers of 2 up to the maximum threads to test
        while [ "$i" -le "$max_threads_to_test" ]; do
            if [[ ! " ${omp_thread_sequence[*]} " =~ " ${i} " ]]; then # Avoid duplicates
                omp_thread_sequence+=($i)
            fi
            if [ "$i" -eq "$max_threads_to_test" ]; then break; fi
            # Safely multiply by 2, checking for overflow if it gets too large (unlikely in this context)
            if [ $(( i * 2 )) -le "$max_threads_to_test" ]; then
                i=$(( i * 2 ))
            else
                # If next power of 2 exceeds max, just jump to max_threads_to_test if not already added
                if [[ ! " ${omp_thread_sequence[*]} " =~ " ${max_threads_to_test} " ]]; then
                    omp_thread_sequence+=("$max_threads_to_test")
                fi
                break
            fi
        done

        # Ensure single-socket physical core count is included if it's within test range and not already present
        if [[ ! " ${omp_thread_sequence[*]} " =~ " ${system_physical_cores_per_socket} " ]] && [ "$system_physical_cores_per_socket" -le "$max_threads_to_test" ]; then
            omp_thread_sequence+=("$system_physical_cores_per_socket")
        fi

        # Ensure 2x socket physical core count is included if applicable, within test range, and not already present
        if [ "$system_num_sockets" -ge 2 ]; then
            local two_socket_physical_cores=$(( system_physical_cores_per_socket * 2 ))
            if [[ ! " ${omp_thread_sequence[*]} " =~ " ${two_socket_physical_cores} " ]] && [ "$two_socket_physical_cores" -le "$max_threads_to_test" ]; then
                omp_thread_sequence+=("$two_socket_physical_cores")
            fi
        fi

        # Ensure total logical CPUs count is included if it's within test range and not already present
        if [[ ! " ${omp_thread_sequence[*]} " =~ " ${system_total_logical_cpus} " ]] && [ "$system_total_logical_cpus" -le "$max_threads_to_test" ]; then
            omp_thread_sequence+=("$system_total_logical_cpus")
        fi

        # Remove duplicates and sort numerically to ensure correct order
        IFS=$'\n' omp_thread_sequence=($(sort -n -u <<<"${omp_thread_sequence[*]}"))
        unset IFS

        echo "Info: Using OMP thread sequence: ${omp_thread_sequence[*]}"

        for omp_thread_current in "${omp_thread_sequence[@]}"; do
            if [ "$omp_thread_current" -eq 0 ]; then continue; fi # Skip 0 threads

            local current_nx_scaled="$initial_nx"
            if [ "$measure_scaling_type" == "weak" ]; then
                # Weak scaling: Problem size (NX, NY) scales up by factor of sqrt(number_of_threads)
                local float_nx=$(echo "scale=5; $initial_nx * sqrt($omp_thread_current)" | bc -l)
                current_nx_scaled=$(printf "%.0f\n" "$float_nx")
                # Round to nearest 100
                current_nx_scaled=$(( (current_nx_scaled + 50) / 100 * 100 ))
                if [ "$current_nx_scaled" -eq 0 ]; then current_nx_scaled=100; fi # Ensure NX is not 0 for small initial_nx
            fi

            local num_mpi_processes_for_current_run=1
            local mpi_prefix=""

            if [ "$srun_enabled_flag" == "true" ]; then
                # Calculate number of MPI processes based on total logical CPUs and OMP threads per process
                if [ "$omp_thread_current" -gt 0 ]; then
                    num_mpi_processes_for_current_run=$(( system_total_logical_cpus / omp_thread_current ))
                    if [ "$num_mpi_processes_for_current_run" -eq 0 ]; then num_mpi_processes_for_current_run=1; fi # Ensure at least 1 MPI process

                    if [ "$num_mpi_processes_for_current_run" -gt 1 ]; then
                        mpi_prefix="srun -n $num_mpi_processes_for_current_run"
                        # Use --ntasks-per-node to spread processes if OMP_NUM_THREADS fits within a single socket
                        if [ "$omp_thread_current" -le "$system_physical_cores_per_socket" ] && [ "$system_num_sockets" -gt 1 ]; then
                            local tasks_per_node=$(( system_physical_cores_per_socket / omp_thread_current ))
                            if [ "$tasks_per_node" -eq 0 ]; then tasks_per_node=1; fi # At least 1 task per node
                            mpi_prefix="$mpi_prefix --ntasks-per-node=$tasks_per_node"
                        fi
                    fi
                    if [ "$(( system_total_logical_cpus % omp_thread_current ))" -ne 0 ]; then
                        echo "  Warning: Total logical CPUs ($system_total_logical_cpus) not perfectly divisible by OMP_NUM_THREADS ($omp_thread_current)."
                        echo "  Using $num_mpi_processes_for_current_run MPI processes, resulting in some unused logical CPUs."
                    fi
                fi
            fi

            for (( rep=1; rep<=$current_repetitions; rep++ )); do
                echo "Running $program_type program (repetition $rep of $current_repetitions) with OMP_NUM_THREADS=$omp_thread_current, MPI Processes=$num_mpi_processes_for_current_run, NX=$current_nx_scaled:"
                local full_command_parts=()

                if [ -n "$mpi_prefix" ]; then
                    full_command_parts+=("$mpi_prefix")
                    full_command_parts+=("env")
                fi

                # Set OMP_PROC_BIND and OMP_PLACES for optimal thread affinity
                full_command_parts+=("OMP_PROC_BIND=close")
                full_command_parts+=("OMP_PLACES=cores")

                full_command_parts+=("OMP_NUM_THREADS=$omp_thread_current")
                full_command_parts+=("NUMBA_NUM_THREADS=$omp_thread_current") # For Numba in Python

                if [ "$program_type" == "cpp" ]; then
                    full_command_parts+=("\"$program_path\"")
                elif [ "$program_type" == "python" ]; then
                    full_command_parts+=("python")
                    full_command_parts+=("\"$program_path\"")
                fi

                full_command_parts+=("\"$current_nx_scaled\"")
                full_command_parts+=("\"$current_nx_scaled\"")
                full_command_parts+=("\"$current_max_iter\"")

                local final_command=$(printf "%s " "${full_command_parts[@]}")
                echo "  Command: $final_command"
                eval "$final_command"
                echo ""
            done
        done
    else
        # Case 2: Fixed OMP_NUM_THREADS, potentially with srun for full core utilization
        local num_mpi_processes=1
        local mpi_prefix=""
        local effective_omp_threads=""

        if [ -n "$specific_omp_val_for_logic" ]; then
            effective_omp_threads=$specific_omp_val_for_logic # Already capped to total_logical_cpus above
            if [ "$effective_omp_threads" -eq 0 ]; then effective_omp_threads=1; fi

            if [ "$srun_enabled_flag" == "true" ]; then
                num_mpi_processes=$(( system_total_logical_cpus / effective_omp_threads ))
                if [ "$num_mpi_processes" -eq 0 ]; then num_mpi_processes=1; fi # Ensure at least 1 MPI process

                if [ "$num_mpi_processes" -gt 1 ]; then
                    mpi_prefix="srun -n $num_mpi_processes"
                    # Use --ntasks-per-node to spread processes if OMP_NUM_THREADS fits within a single socket
                    if [ "$effective_omp_threads" -le "$system_physical_cores_per_socket" ] && [ "$system_num_sockets" -gt 1 ]; then
                        local tasks_per_node=$(( system_physical_cores_per_socket / effective_omp_threads ))
                        if [ "$tasks_per_node" -eq 0 ]; then tasks_per_node=1; fi
                        mpi_prefix="$mpi_prefix --ntasks-per-node=$tasks_per_node"
                    fi
                fi

                if [ "$(( system_total_logical_cpus % effective_omp_threads ))" -ne 0 ]; then
                    echo "  Warning: Total logical CPUs ($system_total_logical_cpus) not perfectly divisible by OMP_NUM_THREADS ($effective_omp_threads)."
                    echo "  Using $num_mpi_processes MPI processes, resulting in some unused logical CPUs."
                fi
            fi
        fi

        for (( rep=1; rep<=$current_repetitions; rep++ )); do
            echo "Running $program_type program (repetition $rep of $current_repetitions) with OMP_NUM_THREADS=$effective_omp_threads and MPI Processes=$num_mpi_processes, NX=$initial_nx:"
            local full_command_parts=()

            if [ -n "$mpi_prefix" ]; then
                full_command_parts+=("$mpi_prefix")
                full_command_parts+=("env")
            fi

            # Set OMP_PROC_BIND and OMP_PLACES
            full_command_parts+=("OMP_PROC_BIND=close")
            full_command_parts+=("OMP_PLACES=cores")

            if [ -n "$effective_omp_threads" ]; then
                full_command_parts+=("OMP_NUM_THREADS=$effective_omp_threads")
                full_command_parts+=("NUMBA_NUM_THREADS=$effective_omp_threads")
            fi

            if [ "$program_type" == "cpp" ]; then
                full_command_parts+=("\"$program_path\"")
            elif [ "$program_type" == "python" ]; then
                full_command_parts+=("python")
                full_command_parts+=("\"$program_path\"")
            fi

            full_command_parts+=("\"$initial_nx\"")
            full_command_parts+=("\"$initial_nx\"")
            full_command_parts+=("\"$current_max_iter\"")

            local final_command=$(printf "%s " "${full_command_parts[@]}")
            echo "  Command: $final_command"
            eval "$final_command"
            echo ""
        done
    fi
}

echo "---------------------------------"
echo "Searching for and running C++ programs in ./CPU/cpp/build..."

if [ -d "./CPU/cpp/build" ]; then
    echo "  Attempting to detect C++ compiler information from build artifacts..."
    compiler_name="N/A"
    compiler_version="N/A"
    compiler_flags="N/A"

    if [ -f "./CPU/cpp/build/CMakeCache.txt" ]; then
        compiler_path=$(grep -m 1 "CMAKE_CXX_COMPILER:" ./CPU/cpp/build/CMakeCache.txt | cut -d '=' -f 2 | xargs)
        if [ -n "$compiler_path" ]; then
            compiler_name=$(basename "$compiler_path")
            echo "    Detected C++ Compiler Path: $compiler_path"
            if command -v "$compiler_path" &> /dev/null; then
                version_output=$("$compiler_path" --version 2>&1 | head -n 1)
                compiler_version=$(echo "$version_output" | grep -oP '(?<=version )[^ ]+' || echo "$version_output" | head -n 1)
            fi
        fi

        flags_line=$(grep -m 1 "CMAKE_CXX_FLAGS:STRING=" ./CPU/cpp/build/CMakeCache.txt)
        if [ -n "$flags_line" ]; then
            compiler_flags=$(echo "$flags_line" | cut -d '=' -f 2- | xargs)
        fi
        echo "    Compiler Name: $compiler_name"
        echo "    Compiler Version: $compiler_version"
        echo "    Compiler Flags: $compiler_flags"
    else
        echo "  Warning: CMakeCache.txt not found in ./CPU/cpp/build. Cannot precisely determine compiler and flags."
        echo "  This script assumes C++ programs are built using CMake. If not, compiler info might be missing."
    fi
    echo ""

    mapfile -t cpp_programs < <(find ./CPU/cpp/build -not -path "./CPU/cpp/build/CMakeFiles/*" -type f -executable)
    for program in "${cpp_programs[@]}"; do
        echo "--- Processing C++ program: $program ---"
        # Pass new CPU topology variables to the function
        run_program_with_scaling "$program" "cpp" "$nx" "$max_iter" "$repetitions" "$measure_scaling" "$omp_config_val_for_run" "$num_sockets" "$physical_cores_per_socket" "$total_logical_cpus" "$use_srun"
        echo "---------------------------------"
        echo ""
    done
else
    echo "C++ build directory ./CPU/cpp/build not found. Skipping C++ programs."
    echo "---------------------------------"
fi

echo "Searching for and running Python programs in ./CPU/numba..."
if [ -d "./CPU/numba" ]; then
    mapfile -t python_programs < <(find ./CPU/numba -type f -name "*.py")
    for program in "${python_programs[@]}"; do
        echo "--- Processing Python program: $program ---"
        # Pass new CPU topology variables to the function
        run_program_with_scaling "$program" "python" "$nx" "$max_iter" "$repetitions" "$measure_scaling" "$omp_config_val_for_run" "$num_sockets" "$physical_cores_per_socket" "$total_logical_cpus" "$use_srun"
        echo "---------------------------------"
        echo ""
    done
else
    echo "Python directory ./CPU/numba not found. Skipping Python programs."
    echo "---------------------------------"
fi

echo "All specified programs have been attempted."
