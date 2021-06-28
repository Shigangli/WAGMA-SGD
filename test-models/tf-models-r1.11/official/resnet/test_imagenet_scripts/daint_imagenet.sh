#!/bin/bash -l
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --partition=debug
#SBATCH --time=00:25:00
#SBATCH --output=localsgd.txt

module load daint-gpu
module load cudatoolkit

module unload PrgEnv-cray
module load PrgEnv-gnu

which python
source ~/.bashrc
export SLURM_NTASKS_PER_NODE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HDF5_USE_FILE_LOCKING=FALSE

export PYTHONPATH=$PYTHONPATH:/path/to/WAGMA-SGD/WAGMA-SGD-modules/deep500/:/path/to/WAGMA-SGD/test-models/tf-models-r1.11/




HOROVOD_FLAG="-solo"

DISABLE_WARMUP=

TESTNAME="test-imagenet/"


cd ..
# ImageNet FB run
srun -n $SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE -c $SLURM_CPUS_PER_TASK python imagenet_main_distributed.py -dd /project/g34/imagenet -md /path/to/imgnetmodel-$TESTNAME -ed /path/to/imgnetexport-$TESTNAME $HOROVOD_FLAG -ebe 10 -rs 50 -rv 1 -bs 128 $DISABLE_WARMUP

