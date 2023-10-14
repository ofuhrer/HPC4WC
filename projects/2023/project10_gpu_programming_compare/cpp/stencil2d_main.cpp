
#include "popl.hpp"
#include "stencil2d_common.h"
#include <iostream>
#include <string.h>
#include <vector>

void parseArgs(
    int argc, char** argv, int& x, int& y, int& z, int& iter, int& b,
    bool& host, bool& gpu, bool& scan, bool& noshared, bool& occ, int& runs,
    std::string& filename)
{
    popl::OptionParser opts("Parameters");
    opts.add<popl::Value<int>>("x", "", "number of grid points in x dimension", x, &x);
    opts.add<popl::Value<int>>("y", "", "number of grid points in y dimension", y, &y);
    opts.add<popl::Value<int>>("z", "", "number of grid points in z dimension", z, &z);
    opts.add<popl::Value<int>>("i", "", "number of iterations", iter, &iter);
    opts.add<popl::Value<int>>("b", "", "block size where bx=by=b (shared memory: 4/_12_/28, otherwise:8/_16_/32)", b, &b);
    opts.add<popl::Switch>("h", "host", "run computation on host", &host);
    opts.add<popl::Switch>("g", "gpu", "run computation on gpu", &gpu);
    opts.add<popl::Switch>("s", "scan", "run scan over different configurations", &scan);
    opts.add<popl::Switch>("n", "noshared", "disable shared memory on GPU and do direct calculation", &noshared);
    opts.add<popl::Switch>("o", "occupancy", "show occupancy analysis", &occ);
    opts.add<popl::Value<int>>("r", "runs", "measure best time over multiple kernel invocations", runs, &runs);
    opts.add<popl::Value<std::string>>("f", "file", "file where results are written to", filename, &filename);
    try {
        opts.parse(argc, argv);
    } catch (const std::exception& e) {
        std::cout << "Invalid argument: " << e.what() << "\n";
        std::cout << opts << "\n";
        exit(EXIT_FAILURE);
    }
    if (!opts.unknown_options().empty()) {
        for (const auto& opt : opts.unknown_options())
            std::cout << "Unknown argument: " << opt << "\n";
        std::cout << opts << "\n";
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv)
{
    // Default parameters
    int x = 120, y = 120, z = 64, iter = 1024, b = 0, runs = 1;
    bool host = false, gpu = false, scan = false, noshared = false, occ = false;
    std::string filename;

    // Parse command line arguments
    parseArgs(argc, argv, x, y, z, iter, b, host, gpu, scan, noshared, occ, runs, filename);

    // Set optimal block size if not specified
    if (gpu && b == 0)
        b = noshared ? 16 : 12;

    // Print all parameters
    if (!scan) {
        printf("arguments: x=%d y=%d z=%d, iter=%d, b=%d, host=%d, gpu=%d, scan=%d, noshared=%d, occ=%d, runs=%d.\n",
            x, y, z, iter, b, host, gpu, scan, noshared, occ, runs);
    }

    // Run on host
    if (host) {
        printf("running on host ...");
        int tmp_z = 1;
        stencil2d_host(x, y, tmp_z, iter, filename.c_str());
        printf(" finished.\n");
    }

    // Run occupancy analysis
    if (occ) {
        int tmp_iter = 0;
        int tmp_runs = 1;
        float time_ms = stencil2d_cuda(y, x, z, tmp_iter, b, noshared, occ, tmp_runs, "");
    }

    // Run on GPU
    if (gpu) {
        printf("running on gpu ...");
        bool tmp_occ = false;
        float time_ms = stencil2d_cuda(x, y, z, iter, b, noshared, tmp_occ, runs, filename.c_str());
        printf(" finished.\n");
        printf("time: %.3f ms\n", time_ms);
    }

    // Run scan on GPU
    if (scan) {
        printf("pd.DataFrame(columns=['x','y','z','iter','time'],data=[\\\n");
        std::vector<int> values;
        if (!noshared) {
            if (b == 4 || b == 12)
                values = { 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 276, 288, 300 };
            if (b == 28)
                values = { 28, 56, 84, 112, 140, 168, 196, 224, 252, 280, 308 };
        } else {
            if (b == 8 || b == 16)
                values = { 12, 28, 44, 60, 76, 92, 108, 124, 140, 156, 172, 188, 204, 220, 236, 252, 268, 284, 300 };
            if (b == 32)
                values = { 28, 60, 92, 124, 156, 188, 220, 252, 284 };
        }
        for (int tmp_x : values) {
            for (int tmp_y : values) {
                bool tmp_occ = false;
                float time_ms = stencil2d_cuda(tmp_x, tmp_y, z, iter, b, noshared, tmp_occ, runs, "");
                printf("[%3d, %3d, %2d, %3d, %6.2f],\\\n", tmp_x, tmp_y, z, iter, time_ms);
                fflush(stdout);
            }
        }
        printf("])\n");
    }
}
