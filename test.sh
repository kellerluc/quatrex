#!/bin/bash -l
#SBATCH --job-name="quatrex-example-cnt"
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --account=lp16
##SBATCH --account=g186
##SBATCH --reservation=g186
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=1
#SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
##SBATCH --uenv=prgenv-gnu/26.5:v2
##SBATCH --view=modules


# set -e -u

# module list
date


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=0
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# export NCCL_NET='AWS Libfabric'
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

export CUPY_CACHE_DIR=${SCRATCH}/.cupy/kernel_cache
export NUMBA_CACHE_DIR=${SCRATCH}/.numba/kernel_cache

# qttools profiling options.
export PROFILE_LEVEL=off


# export ALLTOALL_COMM_TYPE=host_mpi
# export OTHER_COMM_TYPE=host_mpi

conda activate quatrex-py13

srun quatrex run_example --name carbon-nanotube