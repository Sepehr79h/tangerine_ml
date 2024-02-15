#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=a100:4
#SBATCH --ntasks=1
#SBATCH --time=12:0:0
#SBATCH --account=def-shuruiz
#SBATCH --mem=0
#SBATCH --output=/home/sepehrh/projects/def-shuruiz/sepehrh/tangerine_ml/slurm-%j.out

module load gcc/9.3.0 arrow/8 python/3.8

VENV_PATH="/home/sepehrh/projects/def-shuruiz/sepehrh/venv"
source "$VENV_PATH/bin/activate"
python fine-tune.py
