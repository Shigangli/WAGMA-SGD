#!/bin/bash -l
#SBATCH --nodes=64
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --partition=normal
#SBATCH --time=03:40:00
#SBATCH --output=64nodes-group4-90epoch.txt

module load daint-gpu
module load cudatoolkit

module unload PrgEnv-cray
module load PrgEnv-gnu

which python
source ~/.bashrc
export SLURM_NTASKS_PER_NODE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HDF5_USE_FILE_LOCKING=FALSE

export PYTHONPATH=$PYTHONPATH:/path/to/WAGMA-SGD/wagmaSGD-modules/deep500/:/path/to/WAGMA-SGD/test-models/tf-models-r1.11/




#HOROVOD_FLAG=
#HOROVOD_FLAG="-hvd"
HOROVOD_FLAG="-solo"

DISABLE_WARMUP=
#DISABLE_WARMUP="-dwu"

TESTNAME="test-WAGMA-SGD-group4/"





# ImageNet FB run
srun -n $SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE -c $SLURM_CPUS_PER_TASK python imagenet_main_wagma_sgd.py -dd /project/g34/imagenet -md /scratch/imgnetmodel-$TESTNAME -ed /scratch/imgnetexport-$TESTNAME $HOROVOD_FLAG -ebe 10 -rs 50 -rv 1 -bs 128 $DISABLE_WARMUP
#done

