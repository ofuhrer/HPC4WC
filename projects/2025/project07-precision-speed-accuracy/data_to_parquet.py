import sys
import numpy as np
import duckdb as ddb
import pyarrow as pa
import pyarrow.parquet as pq
import math
import os

# command line call signature: python data_to_parquet.py --nx 100 --ny 100 --nz 1 --nums 100 --tf timestep_double.txt --prc double
if __name__ == "__main__":

    nx = int(sys.argv[2])
    ny = int(sys.argv[4])
    nz = int(sys.argv[6])
    nms = int(sys.argv[8])
    tf = str(sys.argv[10])

    # os.chdir('..')
    prc = str(sys.argv[12])

    if prc == 'longdouble':
        dtype = np.float64()
        pa_dtype = pa.float64()
    elif prc == 'double':
        dtype = np.float64()
        pa_dtype = pa.float64()
    elif prc == 'single':
        dtype = np.float32()
        pa_dtype = pa.float32()
    else:
        dtype = np.float16()
        pa_dtype = pa.float16()

    n_halo = 3

    header_size = 6 * 4  # 6 int32 fields × 4 bytes each (to skip metadata)
    field_size = (nx + 2 * n_halo) * (ny + 2 * n_halo) * nz * np.dtype(dtype).itemsize
    phys_field_size = (nx + 2 * n_halo) * (ny + 2 * n_halo) * nz

    dt = np.genfromtxt(tf, dtype=dtype)

    output_files = [f"output_data/u_out_field_nx_{nx}_ny_{ny}_nz_{nz}_{prc}.dat",
                    f"output_data/v_out_field_nx_{nx}_ny_{ny}_nz_{nz}_{prc}.dat"]

    cats = ['vel', 't']
    x = 1.0 / (nx-1)

    schema = pa.schema([pa.field(cat, pa_dtype) for cat in cats])

    for o_file in output_files:

        o_files = o_file.split('.')

        no_file = o_files[0] + '.parquet'

        writer = pq.ParquetWriter(no_file, schema)

        with open(o_file, "rb") as f:

            for i in range(nms):

                data = np.fromfile(o_file, dtype=dtype, count= phys_field_size, sep="", offset = header_size + i * (header_size + field_size)).reshape((ny + 2 * n_halo), (nx + 2 * n_halo))

                inner = data[n_halo:n_halo+ny, n_halo:n_halo+nx].astype(dtype)
                ts = i * dt * np.ones(shape=(ny, nx), dtype=dtype)

                # print([inner.ravel()])

                batch = pa.record_batch(data=[inner.ravel(), ts.ravel()], schema=schema)
                writer.write_batch(batch)
        writer.close()

        # os.remove(o_file)