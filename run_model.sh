#!/bin/bash 
#SBATCH --account=hai_pathology
# budget account where contingent is taken from
#SBATCH --nodes=2
#SBATCH --job-name=luk_fusion
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=output.out
#SBATCH --error=output.err
#SBATCH --partition=booster
#SBATCH --time=05:00:00


# *** start of job script **

module load Python/3.11.3
source /p/project1/hai_pathology/luknarova/venvs/luknarova2/bin/activate

python3 src/train.py


