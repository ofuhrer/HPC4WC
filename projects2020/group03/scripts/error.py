#!/usr/bin/env python
import os
import numpy as np

def read_field_from_file(filename, num_halo=None):
    (rank, nbits, num_halo, nx, ny, nz) = np.fromfile(filename, dtype=np.int32, count=6)
    offset=(3 + rank) * 32 // nbits
    data = np.fromfile(filename, dtype=np.float32 if nbits == 32 else np.float64, \
                       count=nz * ny * nx + offset)
    if rank == 3:
        return np.reshape(data[offset:], (nz, ny, nx))
    else:
        return np.reshape(data[offset:], (ny, nx))

def main():
    compilers = ['gnu', 'intel', 'cray', 'pgi']
    versions = ['orig', 'mpi', 'openmp', 'openmp_target', 'openmp_split', 'openacc', 'openacc_split']

    here = os.path.dirname(os.path.abspath(__file__))
    baseline_path = here + '/../build/cray/src/orig'
    in_field_base = read_field_from_file(baseline_path + '/in_field.dat')
    out_field_base = read_field_from_file(baseline_path + '/out_field.dat')
    for compiler in compilers:
        for version in versions:
            path = f'{here}/../build/{compiler}/src/{version}'
            try:
                in_field = read_field_from_file(path + '/in_field.dat')
                out_field = read_field_from_file(path + '/out_field.dat')
                assert np.all(in_field == in_field_base)
                assert np.all(np.abs(out_field - out_field_base) < 1e-4)
                print(compiler, version,  np.amax(np.abs(out_field - out_field_base)))
            except FileNotFoundError:
                pass

if __name__ == '__main__':
    main()

# vim: set filetype=python expandtab tabstop=4 softtabstop=4 shiftwidth=4 :
