#!/bin/bash
#SBATCH --job-name=kt-measure
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=08:00:00
#SBATCH --exclude=dw-2-4
#SBATCH --output=/home/kzhao2/nobackup/autodelete/areal_rl/logs/measure_%j.out
# Static (k_t,p_t) measurement for a new arch: gen 8 shards (4-way parallel) + bf16 rescore.
source /home/kzhao2/gap_measurement/rl/env.sh
cd /home/kzhao2/gap_measurement
source .venv/bin/activate
set -x
for round in 0 1; do
  for g in 0 1 2 3; do
    s=$((round*4+g))
    CUDA_VISIBLE_DEVICES=$g OPENSSL_CONF=/dev/null python src/gen_vllm.py --arch $ARCH --shard $s --num-shards 8 &
  done; wait
done
for round in 0 1; do
  for g in 0 1 2 3; do
    s=$((round*4+g))
    CUDA_VISIBLE_DEVICES=$g OPENSSL_CONF=/dev/null python src/recompute_train.py --arch $ARCH --shard $s --precision bf16 &
  done; wait
done
echo MEASURE_DONE_$ARCH
