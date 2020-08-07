#!/bin/bash

#This script launches all the jobs for our comparison: the mpi base version, mpiomp version, and compare script
#See notes below for running mpiomp versions with single or multiple nodes

#OPTIONAL clean before starting 
#If you want to keep the job run files then comment this out
make clean 

#General commands to run on daint
module unload PrgEnv-gnu
module load PrgEnv-cray
make VERSION=mpiomp-otf
make VERSION=mpiomp

#Runs both programs regardless of how many nodes your requested with login to jupyter
#Because you may want to run on a different number of nodes, then you must submit a sbatch
#Submits jobs to queue

#Make changes to number of nodes, ranks, openMP threads, etc. in sbatch scripts

#Run mpi only version
#sbatch -C gpu run_mpi.sh

#Run hybrid version
#sbatch -C gpu run_hybrid.sh

#Run hybrid and then on the fly version
sbatch -C gpu run_hybrid.sh
sbatch -C gpu run_otf.sh

#Checks jobs for specific user (-u USERNAME)
squeue -u course40

#### NOTES for Slurm ####
# nodes = --nodes 
# mpi ranks = --ntasks
# mpi ranks split over nodes, so: --ntasks-per-node = --ntasks / --nodes
# openMP threads = export OMP_NUM_THREADS
