#!/bin/sh
#BSUB -q mpi
#BSUB -W 8:00
#BSUB -o /usr/users/scarste/lfs_out/out.%J
#BSUB -a openmpi
#BSUB -n 41
#BSUB -R scratch

source ~/.bashrc
module purge
module load intel/compiler
module load intel/mkl
module load openmpi/gcc
module load python/site-modules

export OMPI_MCA_btl_openib_ib_timeout=28

config_file=~/projects/ensemble_hic/scripts/proteins/config.cfg

mpirun python -u ~/projects/ensemble_hic/scripts/run_simulation.py $config_file
