#!/bin/bash

module --force purge
module load Stages/2023 
module load GCC/11.3.0 
module load OpenMPI/4.1.4
module load PyTorch/1.12.0-CUDA-11.7
module load PyTorch-Geometric/2.1.0-PyTorch-1.12.0-CUDA-11.7
module load tensorboard/2.11.2

tensorboard --logdir /p/home/jusers/kasravi1/juwels/shared/projects/atmorep/logs/profile_par_rank0_dv5uhcmm/profile --port 3005
