#!/bin/bash 
#SBATCH --account=hai_pathology
# budget account where contingent is taken from
#SBATCH --nodes=1
#SBATCH --job-name=luk_fusion
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=output.out
#SBATCH --error=output.err
#SBATCH --partition=booster
#SBATCH --time=05:00:00


# *** start of job script **


source $HOME/luknarova/bin/activate
module load Python/3.11.3

python3 src/train.py


