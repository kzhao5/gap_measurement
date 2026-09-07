#!/bin/bash
#SBATCH --job-name=contam
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --exclude=dw-2-4
#SBATCH --output=/home/kzhao2/nobackup/autodelete/gap_measurement/logs/contam_%j.out

# (Z,B) contamination model verification: paired fresh/stale scoring with
# lg0-e3 checkpoint versions (v28/v57/v86), then the full A-J analysis.
source /home/kzhao2/gap_measurement/slurm/env.sh
LOGD=/home/kzhao2/nobackup/autodelete/gap_measurement/logs
set -e

for g in 0 1 2 3; do
  ( set -e
    CUDA_VISIBLE_DEVICES=$g python src/paired_staleness.py --stage gen --shard $g
    for M in v57 v86 v28; do
      CUDA_VISIBLE_DEVICES=$g python src/paired_staleness.py --stage pre --shard $g --model $M
    done
    CUDA_VISIBLE_DEVICES=$g python src/paired_staleness.py --stage hf --shard $g
  ) > $LOGD/paired_$g.log 2>&1 &
done
wait
for g in 0 1 2 3; do
  ls /home/kzhao2/nobackup/autodelete/gap_measurement/paired/hf_v86_s$g.parquet \
    || { echo "shard $g incomplete"; tail -5 $LOGD/paired_$g.log; exit 1; }
done

python analysis/contamination_verify.py --dump recipe_sigmatis
echo CONTAM_ALL_DONE
