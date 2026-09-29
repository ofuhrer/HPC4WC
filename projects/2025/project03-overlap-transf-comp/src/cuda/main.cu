#include <iostream>

#include "arccos_cuda.cuh"

int main(int argc, char** argv) {

    // Check command line arguments
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << "<num_arccos_calls> <size> <num_streams> <num_repetitions> <validation_period>" << std::endl;
        return 1;
    }

    // Parse command line arguments
    int num_arccos_calls = std::atoi(argv[1]);
    int size = std::atoi(argv[2]);
    int num_streams = std::atoi(argv[3]);
    int num_repetitions = std::atoi(argv[4]);
    int validation_period = std::atoi(argv[5]);

    // Validate command line arguments
    if (size <= 0 || num_streams <= 0 || num_repetitions <= 0) {
        std::cerr << "Size, number of streams and number of repetitions must be positive integers." << std::endl;
        return 1;
    }

    if (validation_period < 0 || validation_period > num_repetitions) {
        std::cerr << "validation period must be between 0 and the number of repetitions." << std::endl;
        return 1;
    }

    if (size % num_streams != 0) {
        std::cerr << "Size (" << size << ") must be divisible by number of streams (" << num_streams << ")." << std::endl;
        return 1;
    }

    // Declare data and result arrays and streams
    int size_per_stream = size / num_streams;
    
    size_t bytes = size_per_stream * sizeof(fType);

    fType* h_data[num_streams], *h_result[num_streams], *h_reference[num_streams];
    fType* d_data[num_streams];
    cudaStream_t streams[num_streams];

    // Initialize all data and streams
    if (init_data(h_data, h_result, h_reference, d_data, bytes, streams, num_arccos_calls, num_streams, size_per_stream)) {
        std::cerr << "Error initializing data." << std::endl;
        return 1;
    }

    // Run the arccos computation multiple times and measure the duration
    std::chrono::duration<double> avg_duration(0.0);
    int success = 0;
    for (int i = 0; i < num_repetitions; ++i) {
        
        // Run the arccos computation
        std::chrono::duration<double> duration;
        success = run_arccos(num_arccos_calls, size_per_stream, num_streams, duration, h_data, h_result, d_data, streams);

        // If validation period is set, verify results every validation_period repetitions
        if (success == 0 && validation_period > 0 && i % validation_period == 0) {
            success = validate_result(h_reference, h_result, size_per_stream, num_streams);
        }

        // Check the result
        if (success == 0) {
            avg_duration += duration;
        } else {
            std::cerr << "There were errors in the results or there was a runtime error." << std::endl;
            break;
        }
    }

    // Cleanup all allocated memory and destroy all streams
    cleanup(h_data, h_result, h_reference, d_data, streams, num_streams);

    // Compute the average duration
    avg_duration /= num_repetitions;

    // Clean output in out stream
    if (success == 0) {
        std::cout << "### " << num_arccos_calls << " " << size << " " << num_streams << " " << avg_duration.count() << std::endl;
    }

    return success;
}