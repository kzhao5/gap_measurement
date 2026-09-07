#!/bin/bash
#SBATCH --job-name=gradcoh
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=06:00:00
#SBATCH --exclude=dw-2-4
#SBATCH --output=/home/kzhao2/nobackup/autodelete/gap_measurement/logs/gradcoh_%j.out

# Gradient coherence matrix: re-harvest all 8 shards with h_proj added
# (2 shards per GPU, sequential), then the CPU coherence analysis.
source /home/kzhao2/gap_measurement/slurm/env.sh
LOGD=/home/kzhao2/nobackup/autodelete/gap_measurement/logs
set -e

CUDA_VISIBLE_DEVICES=0 python src/harvest_grad.py --shard 0 --smoke
echo "SMOKE OK"

for g in 0 1 2 3; do
  ( CUDA_VISIBLE_DEVICES=$g python src/harvest_grad.py --shard $g \
      > $LOGD/hv2_$g.log 2>&1 && \
    CUDA_VISIBLE_DEVICES=$g python src/harvest_grad.py --shard $((g+4)) \
      > $LOGD/hv2_$((g+4)).log 2>&1 ) &
done
wait
for s in 0 1 2 3 4 5 6 7; do
  ls /home/kzhao2/nobackup/autodelete/gap_measurement/harvest/moe/harv_shard00$s.parquet
done

python analysis/grad_coherence.py --shards 0,1,2,3,4,5,6,7
echo GRADCOH_ALL_DONE
