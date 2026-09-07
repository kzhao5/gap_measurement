#!/bin/bash
#SBATCH --job-name=s0c-cpu
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/home/kzhao2/nobackup/autodelete/gap_measurement/logs/cpu_analysis_%j.out
# Login node kills heavy CPU jobs (Arbiter); run analyses under Slurm.
source /home/kzhao2/gap_measurement/slurm/env.sh
set -x
python analysis/w1prime_delta.py --dataset full --boot 500 --boot-max-trajs 4000
python analysis/s0_closure.py --arch moe --boot 300
python analysis/s0_closure.py --arch dense --boot 300
echo CPU_ANALYSIS_ALL_DONE
