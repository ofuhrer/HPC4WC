// Author: Philip Ploner
// Date: 2024-08-24
// This is an adapted version of stencil_1node.cpp, which works on n nodes using MPI parallelization


#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <thread>
#include <chrono>
#include <sstream>
#ifdef CRAYPAT
#include "pat_api.h"
#endif
#include "utils.h"
#include "partitioner.cpp"


namespace {

// Halo udate using MPI. The message passing is done only using std::vectors
// Before sending, the Storage3D<double> fields are therefore decomposed into std::vectors and reassembled into Storage3D<double> elements after receiving
void updateHalo(Storage3D<double> &inField, Partitioner &p) {

  const int xInterior = inField.xMax() - inField.xMin();
  const int yInterior = inField.yMax() - inField.yMin();
  MPI_Request sendRequests[4], recvRequests[4];
  MPI_Status statuses[4];

  // allocate recv buffer for all edges (corners in left and right buffer)
  std::vector<double> b_rcvbuf(inField.zMax() * inField.haloSize() * xInterior);
  std::vector<double> t_rcvbuf(inField.zMax() * inField.haloSize() * xInterior);
  std::vector<double> l_rcvbuf(inField.zMax() * inField.haloSize() * inField.ySize());
  std::vector<double> r_rcvbuf(inField.zMax() * inField.haloSize() * inField.ySize());

  // allocate send buffer for all edges (corners in left and right buffer)
  std::vector<double> b_sndbuf(inField.zMax() * inField.haloSize() * xInterior);
  std::vector<double> t_sndbuf(inField.zMax() * inField.haloSize() * xInterior);
  std::vector<double> l_sndbuf(inField.zMax() * inField.haloSize() * inField.ySize());
  std::vector<double> r_sndbuf(inField.zMax() * inField.haloSize() * inField.ySize());

  // fill send buffers
  for (std::size_t k = 0; k < inField.zMax(); ++k) {
    for (std::size_t j = 0; j < inField.haloSize(); ++j) {
      for (std::size_t i = 0; i < xInterior; ++i) {
        b_sndbuf[i + j * xInterior + k * xInterior * inField.haloSize()] = inField(i + inField.xMin(), j + inField.yMin(), k);
        t_sndbuf[i + j * xInterior + k * xInterior * inField.haloSize()] = inField(i + inField.xMin(), j + yInterior, k);
        }
    }
  }

  for (std::size_t k = 0; k < inField.zMax(); ++k) {
    for (std::size_t j = 0; j < inField.ySize(); ++j) {
      for (std::size_t i = 0; i < inField.haloSize(); ++i) {
        l_sndbuf[i + j * inField.haloSize() + k * inField.haloSize() * inField.ySize()] = inField(i + inField.xMin(), j, k);
        r_sndbuf[i + j * inField.haloSize() + k * inField.haloSize() * inField.ySize()] = inField(i + xInterior, j, k);
      }
    }
  }           

  // give non-blocking send and receive commands
  MPI_Isend(t_sndbuf.data(), t_sndbuf.size(), MPI_DOUBLE, p.top(), 0, MPI_COMM_WORLD, &sendRequests[0]);
  MPI_Isend(b_sndbuf.data(), b_sndbuf.size(), MPI_DOUBLE, p.bottom(), 1, MPI_COMM_WORLD, &sendRequests[1]);
  MPI_Isend(l_sndbuf.data(), l_sndbuf.size(), MPI_DOUBLE, p.left(), 2, MPI_COMM_WORLD, &sendRequests[2]);
  MPI_Isend(r_sndbuf.data(), r_sndbuf.size(), MPI_DOUBLE, p.right(), 3, MPI_COMM_WORLD, &sendRequests[3]);

  MPI_Irecv(t_rcvbuf.data(), t_rcvbuf.size(), MPI_DOUBLE, p.top(), 1, MPI_COMM_WORLD, &recvRequests[0]);
  MPI_Irecv(b_rcvbuf.data(), b_rcvbuf.size(), MPI_DOUBLE, p.bottom(), 0, MPI_COMM_WORLD, &recvRequests[1]);
  MPI_Irecv(l_rcvbuf.data(), l_rcvbuf.size(), MPI_DOUBLE, p.left(), 3, MPI_COMM_WORLD, &recvRequests[2]);
  MPI_Irecv(r_rcvbuf.data(), r_rcvbuf.size(), MPI_DOUBLE, p.right(), 2, MPI_COMM_WORLD, &recvRequests[3]);

  // wait for all non-blocking communication to finish
  MPI_Waitall(4, sendRequests, statuses);
  MPI_Waitall(4, recvRequests, statuses);

  // Update the halos with received data
  // bottom and top edge (without corners)
  for (std::size_t k = 0; k < inField.zMax(); ++k) {
    for (std::size_t j = 0; j < inField.yMin(); ++j) {
      for (std::size_t i = 0; i < xInterior; ++i) {
        inField(i + inField.xMin(), j, k) = b_rcvbuf[i + j * xInterior + k * xInterior * inField.haloSize()];
        inField(i + inField.xMin(), j + inField.yMax(), k) = t_rcvbuf[i + j * xInterior + k * xInterior * inField.haloSize()];
      }
    }
  }

  // left and right edge (with corners)
  for (std::size_t k = 0; k < inField.zMax(); ++k) {
    for (std::size_t j = 0; j < inField.ySize(); ++j) {
      for (std::size_t i = 0; i < inField.xMin(); ++i) {
        inField(i, j, k) = l_rcvbuf[i + j * inField.haloSize() + k * inField.haloSize() * inField.ySize()];
        inField(i + inField.xMax(), j, k) = r_rcvbuf[i + j * inField.haloSize() + k * inField.haloSize() * inField.ySize()];
      }
    }
  }
}

// Diffusion operation on a given field. Includes time measurement of updateHalo().
// diffusion requires a halo number > 1 to work
void apply_diffusion(Storage3D<double> &inField, Storage3D<double> &outField,
                     double alpha, unsigned numIter, int x, int y, int z,
                     int halo, Partitioner &p, std::vector<double> &halo_update_perf) {

  using Duration = std::chrono::duration<double, std::milli>;
  
  Storage3D<double> tmp1Field(x, y, z, halo);

  for (std::size_t iter = 0; iter < numIter; ++iter) {

    //track performance of updateHalo on the current node via time measurement
    //store the time taken for the updateHalo() call in each iteration in the halo_update_perf vector
    auto start = std::chrono::steady_clock::now();
    updateHalo(inField, p);
    auto end = std::chrono::steady_clock::now();
    double diff = Duration(end-start).count() / 1000.;
    halo_update_perf[iter] = diff;


    for (std::size_t k = 0; k < inField.zMax(); ++k) {

      // apply the initial laplacian
      for (std::size_t j = inField.yMin() - 1; j < inField.yMax() + 1; ++j) {
        for (std::size_t i = inField.xMin() - 1; i < inField.xMax() + 1; ++i) {
          tmp1Field(i, j, 0) = -4.0 * inField(i, j, k) + inField(i - 1, j, k) +
                               inField(i + 1, j, k) + inField(i, j - 1, k) +
                               inField(i, j + 1, k);
        }
      }

      // apply the second laplacian
      for (std::size_t j = inField.yMin(); j < inField.yMax(); ++j) {
        for (std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
          double laplap = -4.0 * tmp1Field(i, j, 0) + tmp1Field(i - 1, j, 0) +
                          tmp1Field(i + 1, j, 0) + tmp1Field(i, j - 1, 0) +
                          tmp1Field(i, j + 1, 0);

          // and update the field
          if (iter == numIter - 1) {
            outField(i, j, k) = inField(i, j, k) - alpha * laplap;
          } else {
            inField(i, j, k) = inField(i, j, k) - alpha * laplap;
          }
        }
      }
    }
  }
}

} // namespace

int main(int argc, char *argv[]) {

  using Duration = std::chrono::duration<double, std::milli>;

  //initialize MPI communication
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  //read parameters given in srun command
#ifdef CRAYPAT
  PAT_record(PAT_STATE_OFF);
#endif
  int x = atoi(argv[2]);
  int y = atoi(argv[4]);
  int z = atoi(argv[6]);
  int nHalo = atoi(argv[8]);
  int iter = atoi(argv[10]);

  std::vector<int> dims = {z, y, x};

  //check for valid parameters
  assert(x > 0 && y > 0 && z > 0 && nHalo > 1 && iter > 0);

  std::unique_ptr<Storage3D<double>> input;
  std::unique_ptr<Storage3D<double>> output;

  if (rank == 0) {
    // initialize field with 1s in the central part and 0s on the outside (analoguous to the fields used in our exercises)
    input = std::make_unique<Storage3D<double>>(x, y, z, nHalo);
    input->initialize();
    output = std::make_unique<Storage3D<double>>(x, y, z, nHalo);
    output->initialize();

    // save initialized input field data for debugging purposes
    std::ofstream fout;
    fout.open("in_field.dat", std::ios::binary | std::ofstream::trunc);
    input->writeFile(fout);
    fout.close();
  }

  double alpha = 1. / 32.;

  // initialize an object of the partitioner class used for scattering and gathering the fields among MPI ranks
  Partitioner p(MPI_COMM_WORLD, dims, nHalo);

  // scatter the input field from the root over all MPI ranks
  MPI_Barrier(MPI_COMM_WORLD);
  Storage3D<double> part_input = p.scatter(*input);
  MPI_Barrier(MPI_COMM_WORLD);
  Storage3D<double> part_output = part_input;
    
  if (rank == 0) {
      std::cout << "Scattering successful." << std::endl;
  }

  // save dimensions of scattered field on current rank  
  int part_x = part_input.xMax() - part_input.xMin();
  int part_y = part_input.yMax() - part_input.yMin();
  int part_z = part_input.zMax();

  // initialize array for time measurements
  std::vector<double> halo_update_perf(iter + 1, 0.0);

  // apply the diffusion on each rank and fill array of updateHalo() time measurements
  apply_diffusion(part_input, part_output, alpha, iter, part_x, part_y, part_z, nHalo, p, halo_update_perf);

  // perform final halo update after last diffusion and fill corresponding last time measurement array entry
  auto start = std::chrono::steady_clock::now();
  updateHalo(part_output, p);
  auto end = std::chrono::steady_clock::now();
  double diff = Duration(end-start).count() / 1000.;
  halo_update_perf.back() = diff;


  // gather the output field from all MPI ranks on the root
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << "Diffusion operation successful." << std::endl;
  }
  output = p.gather(part_output);
  MPI_Barrier(MPI_COMM_WORLD);

  // gather the peformance vectors filled on each MPI rank and store all of them in one vector on the root
  std::vector<double> all_halo_update_perf;
  if (rank == 0) {
    all_halo_update_perf.resize(size * (iter + 1));
  }
  MPI_Gather(halo_update_perf.data(), iter + 1, MPI_DOUBLE, all_halo_update_perf.data(), iter + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  // check for the slowest updateHalo() time over all ranks in each iteration, and fill a vector containing only these highest time values per iteration
  // these will then be used in the data analysis for performance benchmarking
  std::vector<double> halo_update_slowest_perf;
  if (rank == 0) {
    halo_update_slowest_perf.resize(iter + 1);
    for (int i = 0; i <= iter; ++i) {
      double max_perf = 0.0;
      for (int j = 0; j < size; ++j) {
        max_perf = std::max(max_perf, all_halo_update_perf[j * (iter + 1) + i]);
      }
      halo_update_slowest_perf[i] = max_perf;
    }
  }

  if (rank == 0) {
    std::cout << "Gathering successful." << std::endl;
  }

  // Save files containing the output fields (for veryfication of correct data) and the performance data
  if (rank == 0) {
    std::ostringstream filename;
    filename << "output_folder/out_fields/out_field_nNodes_" << size << "_nHalo_" << nHalo << ".dat";
    std::ofstream fout;
    fout.open(filename.str(), std::ios::binary | std::ofstream::trunc);
    output->writeFile(fout);
    fout.close();

    std::ostringstream perf_filename;
    perf_filename << "output_folder/halo_update_perfs/halo_update_perf_nNodes_" << size << "_nHalo_" << nHalo << ".csv";
    std::ofstream perf_out;
    perf_out.open(perf_filename.str());
    for (const auto& perf : halo_update_slowest_perf) {
      perf_out << perf << "\n";
    }
    perf_out.close();
  }

  return 0;
}


