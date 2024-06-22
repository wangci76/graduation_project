#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p v100
#SBATCH --qos=dcgpu
#SBATCH --gres=gpu:4
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --job-name=nice
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2731833770@qq.com

#singularity exec --nv sandbox sh train_exp.sh
singularity exec --nv sandbox python demo.py
