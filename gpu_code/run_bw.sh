#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=00:01:00
#SBATCH --mem=1000mb
#SBATCH -J test
#SBATCH --partition=dev_gpu_4		# dev_gpu4, gpu_4, or gpu_8
#SBATCH --gres=gpu:1
#SBATCH --signal=B:USR2@60		# send SIGNAL to the code to wrap up 60 seconds before killing it

#module load devel/cuda/12.0
#module load compiler/gnu/12.1
module load compiler/gnu/10.2
module load devel/cuda/11.4

#without 'exec' here, the signal is not passed to our executable /=
exec ./main