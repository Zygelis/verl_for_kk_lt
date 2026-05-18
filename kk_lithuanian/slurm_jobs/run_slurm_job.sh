#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:2             # Change to more GPUs if needed
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00            # Adjust time limit as you expect
#SBATCH -n1
#SBATCH --job-name=qwen3_1.7B_dapo
#SBATCH --output=qwen3_1.7B_dapo_%j.out

cd /scratch/lustre/home/zygi9184/verl
source venv/bin/activate

python3 --version
which python3
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

# Run your training (e.g., your shell script)
chmod +x kk_lithuanian/training_scripts/dapo/dapo_qwen3_1.7B.sh
bash kk_lithuanian/training_scripts/dapo/dapo_qwen3_1.7B.sh