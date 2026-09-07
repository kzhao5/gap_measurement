#!/bin/bash
#SBATCH --job-name=w1prime
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --exclude=dw-2-4
#SBATCH --output=/home/kzhao2/nobackup/autodelete/gap_measurement/logs/w1prime_%j.out

# (W1') within-trajectory coupling, fresh replication batch:
# 4 GPUs = 4 parallel gen shards (vLLM, decode-time logprobs) ->
# 4 parallel training-side recomputes (bf16, --entropy skips routing) ->
# CPU analysis (Delta(phi) buckets, bootstrap, controls).
source /home/kzhao2/gap_measurement/slurm/env.sh
LOGD=/home/kzhao2/nobackup/autodelete/gap_measurement/logs

for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python src/gen_w1p.py --shard $g --num-shards 4 \
    > $LOGD/w1p_gen_$g.log 2>&1 &
done
wait
for g in 0 1 2 3; do
  ls /home/kzhao2/nobackup/autodelete/gap_measurement/gen/moe/trajs_shard90$g.parquet \
    || { echo "gen shard $g missing"; exit 1; }
done

for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python src/recompute_train.py --arch moe \
    --shard $((900+g)) --precision bf16 --entropy --tag w1p \
    > $LOGD/w1p_rec_$g.log 2>&1 &
done
wait

python analysis/w1prime_delta.py --dataset fresh --boot 500
