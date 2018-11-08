#!/bin/bash
#SBATCH --account=su62
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=512
#SBATCH --cpus-per-task=1
# SBATCH --mail-user=juan.nunez-iglesias@monash.edu
# SBATCH --mail-type=FAIL
echo $DIR
/home/jnun0003/.conda/envs/mic/bin/python ./convert-tifs.py $DIR
