// This script was used for various testing and debugging purposes of the partitioner.cpp file, in particular for printing small fields at various steps of the scattering and gathering processes
// It was changed a lot while testing and debugging and is only provided for completeness and not optimized for ease of use


#include <iostream>
#include <vector>
#include <mpi.h>
#include <cassert>
#include <fstream>
#include <omp.h>
#include "utils.h"
#include "partitioner.cpp"




int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nx = 16;
    int ny = 16;
    int nz = 5;

    int num_halo = 2;

    // vector with dimensions of the grid
    std::vector<int> dims = {nz, ny, nx};

    // create a 3d field with 4x4x4 coordinates
    Storage3D<double> input(nx, ny, nz, num_halo);
    input.initialize();
    std::cout << " input data size: " << input.data_size() << std::endl;
    
    // print initial field
    if (rank == 0) {
        std::cout << "Input field" << std::endl;
        for (int k = 0; k < input.zMax(); k++) {
            std::cout << "z = " << k << std::endl;
            for (int i = 0; i < input.xSize(); i++) {
                for (int j = 0; j < input.ySize(); j++) {
                    std::cout << input(i,j,k) << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    
    // initialize an object of the partitioner class
    Partitioner p(MPI_COMM_WORLD, dims, num_halo);
    std::vector<int> global_shape = p.global_shape();
    
    // print "Partitioner initialized"
    if (rank == 0) {
        std::cout << "Number of ranks: " << p.num_ranks() << std::endl;
        std::cout << "Global shape: "<< global_shape[0] << " " << global_shape[1] << " " <<global_shape[2] << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 1) {
    std::cout << "Number of ranks: " << p.num_ranks() << std::endl;
    std::cout << "Global shape: "<< global_shape[0] << " " << global_shape[1] << " " <<global_shape[2] << std::endl;
    }

    Storage3D<double> part_field = p.scatter(input);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "Scattering successful." << std::endl;
    }


    // print scattered fields on first 3 ranks. Sometimes overlaps in the output happen, inspite of the MPI_Barrier calls. In that case, adding wait commands for a different amount of seconds on each node helps for a clearer output.
    if (rank == 0) {
        std::cout << "rank: " << rank << std::endl;
        for (int k = 0; k < part_field.zMax(); k++) {
            std::cout << "z = " << k << std::endl;
            for (int i = 0; i < part_field.xSize(); i++) {
                for (int j = 0; j < part_field.ySize(); j++) {
                    std::cout << part_field(i,j,k) << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 1) {
        std::cout << "rank: " << rank << std::endl;
        for (int k = 0; k < part_field.zMax(); k++) {
            std::cout << "z = " << k << std::endl;
            for (int i = 0; i < part_field.xSize(); i++) {
                for (int j = 0; j < part_field.ySize(); j++) {
                    std::cout << part_field(i,j,k) << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 2) {
        std::cout << "rank: " << rank << std::endl;
        for (int k = 0; k < part_field.zMax(); k++) {
            std::cout << "z = " << k << std::endl;
            for (int i = 0; i < part_field.xSize(); i++) {
                for (int j = 0; j < part_field.ySize(); j++) {
                    std::cout << part_field(i,j,k) << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
   /*
    if (rank == 0) {
        std::cout << "rank: " << rank << std::endl;
        std::cout << "left: " << p.left() << std::endl;
        std::cout << "right: " << p.right() << std::endl;
        std::cout << "top: " << p.top() << std::endl;
        std::cout << "bottom: " << p.bottom() << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1) {
        std::cout << "rank: " << rank << std::endl;
        std::cout << "left: " << p.left() << std::endl;
        std::cout << "right: " << p.right() << std::endl;
        std::cout << "top: " << p.top() << std::endl;
        std::cout << "bottom: " << p.bottom() << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 2) {
        std::cout << "rank: " << rank << std::endl;
        std::cout << "left: " << p.left() << std::endl;
        std::cout << "right: " << p.right() << std::endl;
        std::cout << "top: " << p.top() << std::endl;
        std::cout << "bottom: " << p.bottom() << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 3) {
        std::cout << "rank: " << rank << std::endl;
        std::cout << "left: " << p.left() << std::endl;
        std::cout << "right: " << p.right() << std::endl;
        std::cout << "top: " << p.top() << std::endl;
        std::cout << "bottom: " << p.bottom() << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 4) {
        std::cout << "rank: " << rank << std::endl;
        std::cout << "left: " << p.left() << std::endl;
        std::cout << "right: " << p.right() << std::endl;
        std::cout << "top: " << p.top() << std::endl;
        std::cout << "bottom: " << p.bottom() << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 5) {
        std::cout << "rank: " << rank << std::endl;
        std::cout << "left: " << p.left() << std::endl;
        std::cout << "right: " << p.right() << std::endl;
        std::cout << "top: " << p.top() << std::endl;
        std::cout << "bottom: " << p.bottom() << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 6) {
        std::cout << "rank: " << rank << std::endl;
        std::cout << "left: " << p.left() << std::endl;
        std::cout << "right: " << p.right() << std::endl;
        std::cout << "top: " << p.top() << std::endl;
        std::cout << "bottom: " << p.bottom() << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 7) {
        std::cout << "rank: " << rank << std::endl;
        std::cout << "left: " << p.left() << std::endl;
        std::cout << "right: " << p.right() << std::endl;
        std::cout << "top: " << p.top() << std::endl;
        std::cout << "bottom: " << p.bottom() << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 8) {
        std::cout << "rank: " << rank << std::endl;
        std::cout << "left: " << p.left() << std::endl;
        std::cout << "right: " << p.right() << std::endl;
        std::cout << "top: " << p.top() << std::endl;
        std::cout << "bottom: " << p.bottom() << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 9) {
        std::cout << "rank: " << rank << std::endl;
        std::cout << "left: " << p.left() << std::endl;
        std::cout << "right: " << p.right() << std::endl;
        std::cout << "top: " << p.top() << std::endl;
        std::cout << "bottom: " << p.bottom() << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 10) {
        std::cout << "rank: " << rank << std::endl;
        std::cout << "left: " << p.left() << std::endl;
        std::cout << "right: " << p.right() << std::endl;
        std::cout << "top: " << p.top() << std::endl;
        std::cout << "bottom: " << p.bottom() << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 11) {
        std::cout << "rank: " << rank << std::endl;
        std::cout << "left: " << p.left() << std::endl;
        std::cout << "right: " << p.right() << std::endl;
        std::cout << "top: " << p.top() << std::endl;
        std::cout << "bottom: " << p.bottom() << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
*/

    
    // gather the fields again
    std::unique_ptr<Storage3D<double>> outpoint;
    
    if (rank == 0) {
        outpoint = std::make_unique<Storage3D<double>>(p.global_shape()[2] - 2*p.num_halo(), p.global_shape()[1] - 2*p.num_halo(), p.global_shape()[0], p.num_halo(), 0.0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    outpoint = p.gather(part_field);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0){
        std::cout << "Gathering successful " << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    Storage3D<double> output;
    
    
    // print gathered field
    if (rank == 0) {
        output = *outpoint;
        std::cout << "rank: " << rank << std::endl;
        for (int k = 0; k < output.zMax(); k++) {
            std::cout << "z = " << k << std::endl;
            for (int i = 0; i < output.xSize(); i++) {
                for (int j = 0; j < output.ySize(); j++) {
                    std::cout << output(i,j,k) << " ";
                }
                std::cout << std::endl;
            }
        }
        std::cout << "Printing successful" << std::endl;
        std::ofstream fout;
        fout.open("input.dat", std::ios::binary | std::ofstream::trunc);
        input.writeFile(fout);
        fout.close();
        fout.open("out_field.dat", std::ios::binary | std::ofstream::trunc);
        output.writeFile(fout);
        fout.close();    
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    

    

        
    MPI_Finalize();
    std::cout << "MPI closed." << std::endl;
    return 0;
}
