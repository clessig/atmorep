#!/bin/bash
#SBATCH -p develbooster
#SBATCH -A deepacf
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
##SBATCH --mem=0
#SBATCH --job-name atmorep-profiling
#SBATCH --output=./results/slurm/slurm-%j-%x.out

# Print current datetime:
echo "START" $(date +"%Y-%m-%d %H:%M:%S")

# Print node list:
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
echo "SLURM_NODELIST=$SLURM_NODELIST"

# Note: the following srun commands assume that pyxis plugin is installed on a SLURM cluster.
# https://github.com/NVIDIA/pyxis


# Set number of threads to use for parallel regions:
export OMP_NUM_THREADS=1

# Set MLPerf variables:
export DATESTAMP=$(date +"%y%m%d%H%M%S%N")
export EXP_ID=1

# Run the command:
# Note: MASTER_ADDR and MASTER_PORT variables are set automatically by pyxis.

module purge
module try-load GCC Apptainer-Tools
module load GCC CUDA NCCL

##export NCCL_DEBUG=INFO 
export NCCL_SOCKET_IFNAME=ib0 
export GLOO_SOCKET_IFNAME=ib0 


export sif_path="/p/scratch/atmlaml/apptainer/images/atmorep_amd64.sif"



export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi
MASTER_PORT=25678

srun --export=ALL env -u CUDA_VISIBLE_DEVICES \
    apptainer exec \
    --nv \
    --bind .:/workspace/atmorep\
    "$sif_path" \
    bash -c "\
        export PYTHONPATH='/workspace/atmorep'; \
        python -u /workspace/atmorep/atmorep/core/train.py"
