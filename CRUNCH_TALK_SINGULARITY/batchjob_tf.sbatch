#!/bin/bash

#SBATCH -J My_Cool_Science
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH -o tf_sing_job_%j.o
#SBATCH -e tf_sing_job_%j.e

# Print key runtime properties for records
echo Master process running on `hostname`
echo Directory is `pwd`
echo Starting execution at `date`
echo Current PATH is $PATH

export SINGULARITY_BINDPATH="/gpfs/scratch,/gpfs/data"

CONTAINER=/gpfs/data/gk/crunch_package/CRUNCH_TALK_SINGULARITY/crunch_tf2.simg
SCRIPT=/gpfs/data/gk/crunch_package/CRUNCH_TALK_SINGULARITY/pinn_laplace_TF2.py


# Run The Job Through Singularity
singularity exec --nv $CONTAINER python $SCRIPT 
