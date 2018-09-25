#!/bin/sh
#BSUB -q mpi
#BSUB -W wall_time_PH
#BSUB -o /usr/users/scarste/lfs_out/out.%J
#BSUB -a openmpi
#BSUB -n n_replicas_PH
#BSUB -R scratch
#BSUB -R same[model]
#BSUB -R span[ptile='!']

source ~/.bashrc
module purge
module load intel/compiler
module load intel/mkl
module load openmpi/gcc
module load python/site-modules

export OMPI_MCA_btl_openib_ib_timeout=28

config_file=~/projects/ensemble_hic/scripts/nora2012/config_PH

date
mpirun.lsf python -u ~/projects/ensemble_hic/scripts/run_simulation_prior.py $config_file
date
