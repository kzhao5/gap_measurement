#!/bin/bash
#SBATCH --job-name=tokenfp
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --exclude=dw-2-4
#SBATCH --output=/home/kzhao2/nobackup/autodelete/gap_measurement/logs/tokenfp_%j.out

# Token-level MSE fixed point: GPU smoke -> 4-way gradient-proxy harvest
# (h_norm + 128-dim direction projection) -> CPU iteration + shape readout.
source /home/kzhao2/gap_measurement/slurm/env.sh
LOGD=/home/kzhao2/nobackup/autodelete/gap_measurement/logs
set -e

CUDA_VISIBLE_DEVICES=0 python src/harvest_grad.py --shard 0 --smoke
echo "SMOKE OK"

for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python src/harvest_grad.py --shard $g \
    > $LOGD/harv_$g.log 2>&1 &
done
wait
for g in 0 1 2 3; do
  ls /home/kzhao2/nobackup/autodelete/gap_measurement/harvest/moe/harv_shard00$g.parquet
done

python analysis/token_fixedpoint.py --shards 0,1,2,3 --boot 500
echo TOKENFP_ALL_DONE
