#!/bin/sh
#SBATCH -t 1-00:00:00
#SBATCH -e /baycells/home/carstens/slurm_out2/slurm-%j.err
#SBATCH -e /baycells/home/carstens/slurm_out2/slurm-%j.out
#SBATCH -n n_replicas_PH

source ~/.bashrc
# module load openmpi

export OMPI_MCA_btl_openib_ib_timeout=28

config_file=config_PH

date
mpirun python -u ~/projects/ensemble_hic/scripts/continue_simulation/continue_simulation.py $config_file
date
