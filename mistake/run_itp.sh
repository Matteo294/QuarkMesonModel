#!/bin/bash
##PBS -l walltime=1:00:00
#PBS -l nodes=1:ppn=1:gpus=1:gshort
#PBS -q gshort
module load cuda/11.4
module load gcc/10.2

mydev=`cat $PBS_GPUFILE | sed s/.*-gpu// `
export CUDA_VISIBLE_DEVICES=$mydev

cd QuarkMesonModel/mistake
exec ./out input.toml
