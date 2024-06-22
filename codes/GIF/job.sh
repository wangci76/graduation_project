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
#SBATCH --mail-user=12012024@mail.sustech.edu.cn

nvidia-smi

singularity exec --nv -w sdbox/ bash -c "export CUDA_VISIBLE_DEVICES=0,1,2,3; CUDA_VISIBLE_DEVICES=0 python3  dataset_expansion_stable_diffusion_CLIP_batch_optimization_final.py -a  CLIP-VIT-L14   -d cifar100 --checkpoint checkpoint/cifar100/test  --data_dir data/CIFAR_10000  --data_save_dir data/CIFAR_10000_expansion/cifar100_stable_diffusion_scale50_strength0.9_CLIP_optimization_up0.8_batch_5x  --ckpt model/stable_diffusion_v1-4.ckpt  --train-batch 1 --test-batch 1   --expanded_number_per_sample 5 --expanded_batch_size 2 --scale 50 --strength 0.25 --constraint_value 0.8 --total_split 4 --split 0; CUDA_VISIBLE_DEVICES=1 python3  dataset_expansion_stable_diffusion_CLIP_batch_optimization_final.py -a  CLIP-VIT-L14   -d cifar100 --checkpoint checkpoint/cifar100/test  --data_dir data/CIFAR_10000  --data_save_dir data/CIFAR_10000_expansion/cifar100_stable_diffusion_scale50_strength0.9_CLIP_optimization_up0.8_batch_5x  --ckpt model/stable_diffusion_v1-4.ckpt  --train-batch 1 --test-batch 1   --expanded_number_per_sample 5 --expanded_batch_size 2 --scale 50 --strength 0.25 --constraint_value 0.8 --total_split 4 --split 1; CUDA_VISIBLE_DEVICES=2 python3  dataset_expansion_stable_diffusion_CLIP_batch_optimization_final.py -a  CLIP-VIT-L14   -d cifar100 --checkpoint checkpoint/cifar100/test  --data_dir data/CIFAR_10000  --data_save_dir data/CIFAR_10000_expansion/cifar100_stable_diffusion_scale50_strength0.9_CLIP_optimization_up0.8_batch_5x  --ckpt model/stable_diffusion_v1-4.ckpt  --train-batch 1 --test-batch 1   --expanded_number_per_sample 5 --expanded_batch_size 2 --scale 50 --strength 0.25 --constraint_value 0.8 --total_split 4 --split 2; CUDA_VISIBLE_DEVICES=3 python3  dataset_expansion_stable_diffusion_CLIP_batch_optimization_final.py -a  CLIP-VIT-L14   -d cifar100 --checkpoint checkpoint/cifar100/test  --data_dir data/CIFAR_10000  --data_save_dir data/CIFAR_10000_expansion/cifar100_stable_diffusion_scale50_strength0.9_CLIP_optimization_up0.8_batch_5x  --ckpt model/stable_diffusion_v1-4.ckpt  --train-batch 1 --test-batch 1   --expanded_number_per_sample 5 --expanded_batch_size 2 --scale 50 --strength 0.25 --constraint_value 0.8 --total_split 4 --split 3"