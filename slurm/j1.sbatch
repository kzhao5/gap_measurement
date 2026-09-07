#!/bin/bash
#SBATCH --job-name=j1coupling
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --exclude=dw-2-4
#SBATCH --output=/home/kzhao2/nobackup/autodelete/gap_measurement/logs/j1_%j.out

# (J1) trajectory coupling law, independent batch #3:
# disjoint prompts (skip 250 per shard) + offset RNG streams, then
# training-side recompute and the J1 analysis (beta1/beta2/lambda_eff).
source /home/kzhao2/gap_measurement/slurm/env.sh
LOGD=/home/kzhao2/nobackup/autodelete/gap_measurement/logs

for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python src/gen_w1p.py --shard $g --num-shards 4 \
    --tag-base 910 --skip-prompts 250 --seed-offset 25 \
    > $LOGD/j1_gen_$g.log 2>&1 &
done
wait
for g in 0 1 2 3; do
  ls /home/kzhao2/nobackup/autodelete/gap_measurement/gen/moe/trajs_shard91$g.parquet \
    || { echo "gen shard $g missing"; exit 1; }
done

for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python src/recompute_train.py --arch moe \
    --shard $((910+g)) --precision bf16 --entropy --tag w1p \
    > $LOGD/j1_rec_$g.log 2>&1 &
done
wait

python analysis/j1_coupling.py --dataset fresh2 --boot 500
