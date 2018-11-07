#!/bin/bash
#SBATCH --job-name=illum-stitch
#SBATCH --account=su62
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1024
#SBATCH --cpus-per-task=8
# SBATCH --mail-user=juan.nunez-iglesias@monash.edu
# SBATCH --mail-type=END
# SBATCH --mail-type=FAIL
/home/jnun0003/.conda/envs/mic/bin/python ./single-process-vi.py
