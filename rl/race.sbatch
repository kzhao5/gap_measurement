#!/bin/bash
#SBATCH --job-name=kt-race
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=700G
#SBATCH --time=08:00:00
#SBATCH --exclude=dw-2-4
#SBATCH --output=/home/%u/nobackup/autodelete/areal_rl/logs/race_%j.out

source $HOME/gap_measurement/rl/env.sh
LOCKBASE=$RLROOT/locks
mkdir -p "$LOCKBASE"
LOCK=$LOCKBASE/$KT_ARM
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "[race] $KT_ARM already claimed, exiting ($SLURM_JOB_ID on $SLURM_JOB_PARTITION)"
  exit 0
fi
echo "$SLURM_JOB_ID $SLURM_JOB_PARTITION $(hostname)" > "$LOCK/winner"
sleep 5
for j in $(cat $LOCKBASE/$KT_ARM.siblings 2>/dev/null); do
  [ "$j" != "$SLURM_JOB_ID" ] && scancel "$j" 2>/dev/null
done
echo "[race] $KT_ARM winner: job $SLURM_JOB_ID partition $SLURM_JOB_PARTITION node $(hostname)"

cd "$AREAL"
source .venv/bin/activate
python -c "import torch, transformers, datasets, sglang; import areal.trainer.rl_trainer" || true

export KT_SIGMA_TIS=1 KT_SIGMA_C1=1.0 KT_SIGMA_B=2.3
case $KT_ARM in
  affine)  export KT_SIGMA_A=1e9 KT_SIGMA_FLOOR=0    KT_SIGMA_ETA=0.12 ;;
  lowb001) export KT_SIGMA_A=1e9 KT_SIGMA_FLOOR=5e-3 KT_SIGMA_ETA=0 KT_SIGMA_LOWER_ABS=6.91 ;;
  lowb01)  export KT_SIGMA_A=1e9 KT_SIGMA_FLOOR=5e-3 KT_SIGMA_ETA=0 KT_SIGMA_LOWER_ABS=2.303 ;;
esac
export KT_DUMP_DIR="$RLROOT/ktdump/v3_$KT_ARM"
export KT_DUMP_EVERY=10

python examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_icepop.yaml \
    experiment_name=kt-v3race \
    trial_name=$KT_ARM-e3 \
    seed=1 \
    scheduler.type=local \
    cluster.fileroot="$RLROOT/experiments" \
    cluster.name_resolve.nfs_record_root="$RLROOT/name_resolve" \
    actor.path=Qwen/Qwen1.5-MoE-A2.7B-Chat \
    actor.optimizer.lr=3e-6 \
    actor.optimizer.weight_decay=0.01 \
    ++actor.eps_clip=0.2 \
    ++actor.eps_clip_higher=0.28 \
    actor.rejection_sampling.action=clamp \
    actor.rejection_sampling.lower=1e-6 \
    actor.rejection_sampling.upper=1e6 \
    actor.mb_spec.max_tokens_per_mb=4096 \
    ref.mb_spec.max_tokens_per_mb=4096 \
    ++train_dataset.setup_timeout=900 \
    ++valid_dataset.setup_timeout=900 \
    ++rollout.setup_timeout=900 \
    total_train_epochs=3

# winner submits its own TP4 eval the moment training ends
CK=$RLROOT/experiments/checkpoints/kzhao2/kt-v3race/$KT_ARM-e3/default
E=$(ls "$CK" 2>/dev/null | tail -1)
[ -n "$E" ] && sbatch --export=ALL,EVAL_PATH="$CK/$E",EVAL_TAG="v3_$KT_ARM" $HOME/gap_measurement/rl/eval_tp.sbatch
