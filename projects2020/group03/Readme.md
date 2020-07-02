# Overlapping computation on the CPU and the GPU

## Description

> Implement a version of `stencil2d.F90` that offload part of the work onto the GPU and keeps parts on the CPU. How does domain decomposition affect this? Extend the stencil to 3d and decide on domain decomposition in this case. If time permits try any dense linear algebra method. test if we can get an advantage of doing a part of the work on the CPU while the GPU kernel is running. (Literatur [1])

## Compiling
```bash
	source scripts/daint.sh
	scripts/build.sh [ Debug | Release | RelWithDebInfo ]
```

## Running
```bash
	sourc scripts/daint.sh
	sbatch scripts/run.sh
```

## Postprocessing
```bash
	scripts/table.sh path/to/slurm.out
```

[1] https://reader.elsevier.com/reader/sd/pii/S0045782511000235?token=D6719774F918FDC2DBA99AF121BCA0F1F371703BACD747EDFFDC2771E43E920816D31843CEAF2AF698CD81FC86624D65
