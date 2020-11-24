#!/bin/bash
#SBATCH --time=02:00:00    #Time reuired for job to complete
#SBATCH --mem=16g          #CPU ememory
#SBATCH -N 1               # Number of Nodes
#SBATCH -n 4               # Number of CPUs
#SBATCH -o job_%j.out      # Output file
#SBATCH -e job_%j.err      # Error File
#SBATCH -A gk-condo        # Account fixed please don't change
#SBATCH -p gpu             # Partition Name which is gpu
#SBATCH --gres=gpu:1       # Number of GPUs
#SBATCH --constraint=titanrtx        # To run it on the Specific GPU e.g titanrtx 
module load tensorflow/1.14.0_gpu_py36 
module load cuda/10.0.130 cudnn/7.4 python/3.6.6_test
module load Xvfb
python3 -u test_serial_train.py 
