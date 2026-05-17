#!/bin/bash -l
#SBATCH --job-name="quatrex-example-cnt"
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --account=lp16
#SBATCH --time=00:30:00
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=1
#SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

date


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=0
export OMP_PROC_BIND=true
export OMP_PLACES=cores

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

export CUPY_CACHE_DIR=${SCRATCH}/.cupy/kernel_cache
export NUMBA_CACHE_DIR=${SCRATCH}/.numba/kernel_cache

# qttools profiling options.
export PROFILE_LEVEL=off

source ~/load_env.sh 
source ~/load_env.sh 

srun quatrex run /capstor/scratch/cscs/luckeller/quatrex/examples/w90/carbon-nanotube/gw/quatrex_config.toml


