#!/bin/bash -x
#SBATCH --account=hai_pathology
# budget account where contingent is taken from
#SBATCH --nodes=1
#SBATCH --job-name=luk_fusion
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=out.txt
#SBATCH --error=err.txt
#SBATCH --partition=booster
#SBATCH --time=01:00:00
#SBATCH --mem=32GB

# *** start of job script **


source $HOME/luknarova/bin/activate
module load Python/3.11.3

python3 code/train.py

deactivate

