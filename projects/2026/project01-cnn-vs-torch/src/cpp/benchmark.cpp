#include "include/timing.h"
#include "include/cnn.h"
#include "include/utils.h"

#include <cstring>
#include <cctype>

#ifndef NUM_RUNS
#define NUM_RUNS 10
#endif
#ifndef NUM_WARMUP_RUNS
#define NUM_WARMUP_RUNS 2
#endif
#ifndef BATCH_SIZE
#define BATCH_SIZE 1
#endif

static volatile double benchmark_global_sink = 0.0;

typedef struct {
    const char *name;
    cnn_func function;
} benchmark_impl_t;

typedef struct {
    double cycles_median;
    double seconds_median;
    bool has_data;
} bench_result_t;

// ----------------------------------------------------------------------------
// Runs a warm-cache phase for a single implementation: warmup runs followed
// by NUM_RUNS measured runs, reporting median cycles/seconds.
// ----------------------------------------------------------------------------
static bench_result_t benchmark_cnn(cnn_func f, CNNContext &ctx,
                                    pmu_ctx_t &pmu, int num_runs, int num_warmup) {
    bench_result_t result = {0.0, 0.0, false};

    std::cout << Color::GREEN << "--- Performance Benchmarking (" << num_runs << " Runs) ---"
               << Color::RESET << "\n";

    std::vector<double> cycles_arr((size_t)num_runs);
    std::vector<double> sec_arr((size_t)num_runs);

    std::cout << "Running " << num_warmup << " warmup run" << (num_warmup == 1 ? "" : "s") << "...\n";
    for (int w = 0; w < num_warmup; ++w) {
        f(ctx);
        benchmark_global_sink += checksum_tensor(ctx.final_logits);
    }

    std::cout << "Warmup complete. Starting " << num_runs << " benchmark run"
              << (num_runs == 1 ? "" : "s") << "...\n";

    for (int i = 0; i < num_runs; ++i) {

        timer_context_t timer_ctx;
        pmu_cycles_start(&pmu);
        start_timer(&timer_ctx);

        f(ctx);

        stop_timer(&timer_ctx);
        uint64_t cycles = pmu_cycles_stop(&pmu);

        sec_arr[i] = get_elapsed_os_sec(&timer_ctx);
        cycles_arr[i] = (double)cycles;

        benchmark_global_sink += checksum_tensor(ctx.final_logits);
    }

    result.has_data = true;
    result.cycles_median = compute_median(cycles_arr.data(), num_runs);
    result.seconds_median = compute_median(sec_arr.data(), num_runs);

    std::cout << Color::GREEN << "=== Results ===" << Color::RESET << "\n";
    std::cout << "  Median cycles : " << result.cycles_median << "\n";
    std::cout << "  Median time   : " << result.seconds_median
              << " s (" << result.seconds_median * 1000.0 << " ms)\n\n";

    return result;
}


// Fix function naming for verification mode
static std::string slugify(const std::string& name) {
    std::string out;
    bool last_underscore = false;
    for (unsigned char c : name) {
        char lc = static_cast<char>(std::tolower(c));
        if (std::isalnum((unsigned char)lc)) {
            out += lc;
            last_underscore = false;
        } else if (!last_underscore) {
            out += '_';
            last_underscore = true;
        }
    }
    while (!out.empty() && out.front() == '_') out.erase(out.begin());
    while (!out.empty() && out.back()  == '_') out.pop_back();
    return out;
}

int main(int argc, char** argv) {
    bool verify_mode = false;
    bool profile_mode = false;
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "verify")
            verify_mode = true;
        else if (arg == "profile")
            profile_mode = true;
        else if (arg == "-v" || arg == "--verbose" || arg == "verbose")
            verbose = true;
    }

    int batch_size  = verify_mode ? 10000 : BATCH_SIZE;
    int num_classes = 10;

    int num_runs = NUM_RUNS;
    int num_warmup = NUM_WARMUP_RUNS;

    pmu_ctx_t pmu;
    if (pmu_cycles_init(&pmu) != 0) {
        std::cerr << Color::RED << "PMU init failed — cycle counts will be invalid.\n" << Color::RESET;
    }

    std::cout << Color::BOLD_CYAN << "Batch size: " << batch_size << Color::RESET << "\n";
    std::cout << Color::CYAN << "Memory allocation is starting..." << Color::RESET << "\n";

    // 1. INPUT
    Tensor input_batch;
    try {
        load_mnist_images("../../data/MNIST/raw/t10k-images-idx3-ubyte", input_batch, batch_size);
        std::cout << Color::GREEN << "MNIST dataset successfully loaded." << Color::RESET << "\n";
    } catch (const std::exception& e) {
        std::cerr << Color::RED << e.what() << Color::RESET << "\n";
        return 1;
    }

#ifdef PRINT_ASCII
    print_ascii_image(input_batch, 0);
#endif

    // 2. WEIGHTS
    std::cout << Color::BOLD_CYAN << "\nLoading real weights..." << Color::RESET << "\n";
    const std::string w_path = "../python/weights_cpp/";

    Tensor conv1_weight = {std::vector<float>(32 * 1 * 3 * 3), 32,  1,  3,   3};
    Tensor conv1_bias   = {std::vector<float>(32), 32,  1,  1,   1};
    load_binary_weights(w_path + "model_0_weight.bin", conv1_weight.data);
    load_binary_weights(w_path + "model_0_bias.bin", conv1_bias.data);

    Tensor conv2_weight = {std::vector<float>(64 * 32 * 3 * 3), 64, 32,  3,   3};
    Tensor conv2_bias   = {std::vector<float>(64), 64,  1,  1,   1};
    load_binary_weights(w_path + "model_3_weight.bin", conv2_weight.data);
    load_binary_weights(w_path + "model_3_bias.bin", conv2_bias.data);

    Tensor conv3_weight = {std::vector<float>(128 * 64 * 3 * 3), 128, 64,  3,   3};
    Tensor conv3_bias   = {std::vector<float>(128), 128,  1,  1,   1};
    load_binary_weights(w_path + "model_6_weight.bin", conv3_weight.data);
    load_binary_weights(w_path + "model_6_bias.bin", conv3_bias.data);

    Tensor fc_weight    = {std::vector<float>(num_classes * 128),        1,  1, num_classes, 128};
    Tensor fc_bias      = {std::vector<float>(num_classes), num_classes, 1,  1,   1};
    load_binary_weights(w_path + "model_10_weight.bin", fc_weight.data);
    load_binary_weights(w_path + "model_10_bias.bin", fc_bias.data);

    // 3. INTERMEDIATE ACTIVATIONS
    Tensor conv1_out   = {std::vector<float>(batch_size * 32  * 26 * 26), batch_size,  32, 26, 26};
    Tensor pool1_out   = {std::vector<float>(batch_size * 32  * 13 * 13), batch_size,  32, 13, 13};
    Tensor conv2_out   = {std::vector<float>(batch_size * 64  * 11 * 11), batch_size,  64, 11, 11};
    Tensor pool2_out   = {std::vector<float>(batch_size * 64  *  5 *  5), batch_size,  64,  5,  5};
    Tensor conv3_out   = {std::vector<float>(batch_size * 128 *  3 *  3), batch_size, 128,  3,  3};
    Tensor avgpool_out = {std::vector<float>(batch_size * 128),           batch_size, 128,  1,  1};
    Tensor final_logits= {std::vector<float>(batch_size * num_classes),   batch_size,   1,  1, num_classes};

    CNNContext ctx = {
        input_batch,
        conv1_weight, conv1_bias, conv1_out, pool1_out,
        conv2_weight, conv2_bias, conv2_out, pool2_out,
        conv3_weight, conv3_bias, conv3_out, avgpool_out,
        fc_weight,    fc_bias,    final_logits
    };


    // ------------------------------------------------------------------
    // Registered implementations (see CNN_IMPLEMENTATIONS in cnn.h)
    // ------------------------------------------------------------------
        #define CNN_BENCH_ENTRY(function, name) {name, function},
    const benchmark_impl_t implementations[] = {
        CNN_IMPLEMENTATIONS(CNN_BENCH_ENTRY)
    };
    #undef CNN_BENCH_ENTRY

    int num_impls = sizeof(implementations) / sizeof(implementations[0]);

    if (verify_mode) {
        std::string banner = "  VERIFYING ALL REGISTERED IMPLEMENTATIONS  ";
        std::string border(banner.size(), '=');
        std::cout << "\n" << Color::BOLD_YELLOW << border << "\n" << banner << "\n"
                  << border << Color::RESET << "\n\n";

        const std::string out_dir = "../python/weights_cpp/";
        std::ofstream manifest(out_dir + "cpp_verify_manifest.json");
        if (!manifest.is_open()) {
            std::cerr << Color::RED << "Could not open " << out_dir
                      << "cpp_verify_manifest.json for writing.\n" << Color::RESET;
            pmu_cycles_close(&pmu);
            return 1;
        }

        manifest << "[\n";
        for (int i = 0; i < num_impls; ++i) {
            std::cout << Color::CYAN << "-> Running " << implementations[i].name
                      << " for verification..." << Color::RESET << "\n";

            implementations[i].function(ctx);

            std::string filename = "cpp_logits_" + slugify(implementations[i].name) + ".bin";
            save_binary(out_dir + filename, final_logits.data);

            manifest << "  {\"name\": \"" << implementations[i].name << "\", "
                     << "\"file\": \"" << filename << "\"}"
                     << (i + 1 < num_impls ? ",\n" : "\n");

            std::cout << Color::GREEN << "   Wrote " << filename << Color::RESET << "\n";
        }
        manifest << "]\n";

        std::cout << Color::BOLD_GREEN << "Wrote logits for " << input_batch.batches
                  << " images across " << num_impls << " implementation(s)."
                  << Color::RESET << "\n";

        pmu_cycles_close(&pmu);
        return 0;
    }

    if (profile_mode) {
        std::cout << Color::BOLD_CYAN << "\n--- Per-Layer Profiling (" << num_runs << " Runs) ---"
                  << Color::RESET << "\n";

        struct LayerStat { const char *name; std::vector<double> cycles; };
        LayerStat layers[] = {
            {"conv1",   std::vector<double>(num_runs)},
            {"relu1",   std::vector<double>(num_runs)},
            {"pool1",   std::vector<double>(num_runs)},
            {"conv2",   std::vector<double>(num_runs)},
            {"relu2",   std::vector<double>(num_runs)},
            {"pool2",   std::vector<double>(num_runs)},
            {"conv3",   std::vector<double>(num_runs)},
            {"relu3",   std::vector<double>(num_runs)},
            {"avgpool", std::vector<double>(num_runs)},
            {"fc",      std::vector<double>(num_runs)},
        };
        constexpr int num_layers = sizeof(layers) / sizeof(layers[0]);

        for (int w = 0; w < num_warmup; ++w) cnn_baseline(ctx);

        for (int i = 0; i < num_runs; ++i) {
            uint64_t c;
            int li = 0;

            pmu_cycles_start(&pmu); conv2d_forward(ctx.input_batch, ctx.conv1_weight, ctx.conv1_bias, ctx.conv1_out, 1, 0);
            c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;

            pmu_cycles_start(&pmu); relu_forward(ctx.conv1_out);
            c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;

            pmu_cycles_start(&pmu); maxpool2d_forward(ctx.conv1_out, ctx.pool1_out, 2, 2);
            c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;

            pmu_cycles_start(&pmu); conv2d_forward(ctx.pool1_out, ctx.conv2_weight, ctx.conv2_bias, ctx.conv2_out, 1, 0);
            c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;

            pmu_cycles_start(&pmu); relu_forward(ctx.conv2_out);
            c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;

            pmu_cycles_start(&pmu); maxpool2d_forward(ctx.conv2_out, ctx.pool2_out, 2, 2);
            c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;

            pmu_cycles_start(&pmu); conv2d_forward(ctx.pool2_out, ctx.conv3_weight, ctx.conv3_bias, ctx.conv3_out, 1, 0);
            c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;

            pmu_cycles_start(&pmu); relu_forward(ctx.conv3_out);
            c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;

            pmu_cycles_start(&pmu); adaptive_avgpool2d_forward(ctx.conv3_out, ctx.avgpool_out);
            c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;

            pmu_cycles_start(&pmu); linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
            c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;

            benchmark_global_sink += checksum_tensor(ctx.final_logits);
        }

        std::vector<double> medians(num_layers);
        double total = 0.0;
        for (int k = 0; k < num_layers; ++k) {
            medians[k] = compute_median(layers[k].cycles.data(), num_runs);
            total += medians[k];
        }

        std::cout << Color::GREEN << "=== Per-Layer Breakdown (median cycles) ===" << Color::RESET << "\n";
        for (int k = 0; k < num_layers; ++k) {
            std::cout << "  " << Color::BOLD << layers[k].name << Color::RESET
                      << std::string(9 - strlen(layers[k].name), ' ')
                      << medians[k] << " cycles  (" << (100.0 * medians[k] / total) << "%)\n";
        }
        std::cout << "  total: " << total << " cycles\n\n";

        pmu_cycles_close(&pmu);
        return 0;
    }

    std::vector<bench_result_t> results((size_t)num_impls);

    const char *bench_filter = std::getenv("CNN_BENCH_FILTER");
    int num_selected = 0;
    for (int i = 0; i < num_impls; ++i)
        if (implementation_matches_filter(implementations[i].name, bench_filter)) num_selected++;

    if (num_selected == 0) {
        std::cerr << "Error: CNN_BENCH_FILTER matched no implementations: "
                  << (bench_filter ? bench_filter : "") << "\n";
        pmu_cycles_close(&pmu);
        return 1;
    }
    if (bench_filter && *bench_filter) {
        std::cout << Color::BOLD_CYAN << "Benchmark filter: " << bench_filter
                  << " (" << num_selected << " of " << num_impls << " implementations)"
                  << Color::RESET << "\n\n";
    }

    std::cout << Color::BOLD_CYAN << "Runs: " << num_runs << " measured, "
              << num_warmup << " warmup\n\n" << Color::RESET;

    // ------------------------------------------------------------------
    // Execution loop: warm cache only for every implementation
    // ------------------------------------------------------------------
    for (int i = 0; i < num_impls; ++i) {
        if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;

        std::cout << Color::CYAN << "==================================================\n"
                  << " -> Benchmarking " << implementations[i].name << " (WARM CACHE)\n"
                  << "==================================================" << Color::RESET << "\n";
        results[i] = benchmark_cnn(implementations[i].function, ctx, pmu, num_runs, num_warmup);
        std::cout << "\n\n";
    }

    pmu_cycles_close(&pmu);

    if (verbose) {
        std::cout << Color::BOLD_YELLOW << "Raw Logits (Computational Verification):" << Color::RESET << "\n";
        for (int i = 0; i < num_classes; ++i)
            std::cout << "Class " << i << ": " << final_logits.data[i] << "\n";
        std::cout << "\n";
    }

    // ------------------------------------------------------------------
    // Final Summary Table
    // ------------------------------------------------------------------
    std::cout << Color::BOLD_YELLOW << "=== Final Performance Summary ===" << Color::RESET << "\n\n";
    printf("%-42s | %-12s | %-20s | %-14s\n",
           "Implementation", "vs Base", "Cycles", "Seconds");
    printf("-------------------------------------------------------------------------------------------------\n");

    int baseline_idx = -1;
    int best_idx = -1;
    double baseline_cycles = 0.0;
    double best_cycles = 0.0;

    for (int i = 0; i < num_impls; ++i) {
        if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
        if (!results[i].has_data) continue;
        if (std::strcmp(implementations[i].name, "Baseline Nested-Loop") == 0) {
            baseline_idx = i;
            baseline_cycles = results[i].cycles_median;
        }
        if (best_idx < 0 || results[i].cycles_median < best_cycles) {
            best_idx = i;
            best_cycles = results[i].cycles_median;
        }
    }

    for (int i = 0; i < num_impls; ++i) {
        if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;

        char speedup_text[32] = "";
        if (baseline_idx >= 0 && results[i].has_data && results[i].cycles_median > 0.0) {
            snprintf(speedup_text, sizeof(speedup_text), "%.2fx",
                     baseline_cycles / results[i].cycles_median);
        }

        char cyc[24], sec[24];
        if (results[i].has_data) {
            snprintf(cyc, sizeof(cyc), "%.2f", results[i].cycles_median);
            snprintf(sec, sizeof(sec), "%.6f", results[i].seconds_median);
        } else {
            snprintf(cyc, sizeof(cyc), "-");
            snprintf(sec, sizeof(sec), "-");
        }

        bool is_best = (i == best_idx);
        if (is_best) std::cout << Color::GREEN;
        printf("%-42s | %-12s | %-20s | %-14s",
               implementations[i].name, speedup_text, cyc, sec);
        if (is_best) std::cout << Color::RESET;
        std::cout << "\n";
    }

    std::cout << "\nBenchmark sink: " << benchmark_global_sink << "\n\n";

    // Persisted for benchmark.py's C++ vs Python comparison table.
    // Writes every benchmarked implementation (not just the baseline) as a
    // JSON array, so benchmark.py can compare Python against all of them.
    // Filename is batch-size-specific so that concurrently-running jobs for
    // different batch sizes don't race on the same output file.
    std::ofstream timing_file("../python/weights_cpp/cpp_timing_batch" +
                               std::to_string(batch_size) + ".json");
    if (timing_file.is_open()) {
        timing_file << "[\n";
        bool first = true;
        for (int i = 0; i < num_impls; ++i) {
            if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
            if (!results[i].has_data) continue;
            if (!first) timing_file << ",\n";
            first = false;
            timing_file << "  {\"name\": \"" << implementations[i].name << "\", "
                        << "\"median_time_sec\": " << results[i].seconds_median << ", "
                        << "\"median_cycles\": " << results[i].cycles_median << "}";
        }
        timing_file << "\n]\n";
    }

    return 0;
}
