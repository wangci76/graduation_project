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

singularity shell --nv -w sandbox/ -c 'cd /lab/tangb_lab/30011373/zmj/DatasetExpansion/GIF_SD/CIFAR/data/SwinIR; python main_test_swinir.py --task real_sr --scale 4 --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth --folder_lq testsets/RealSRSet+5images/anger_facial_expression --tile
'



