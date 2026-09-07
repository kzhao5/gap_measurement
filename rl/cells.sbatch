#!/bin/bash
#SBATCH --job-name=kt-cells
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=700G
#SBATCH --time=08:00:00
#SBATCH --exclude=dw-2-4
#SBATCH --output=/home/kzhao2/nobackup/autodelete/areal_rl/logs/seeds_%j.out

# Multi-seed significance: METHOD x SEED via --export, exact e3-recipe flags.
source /home/kzhao2/gap_measurement/rl/env.sh
cd "$AREAL"
source .venv/bin/activate
python -c "import torch, transformers, datasets, sglang; import areal.trainer.rl_trainer" || true

OVR=""

case $CELL in
  dsv2) MODELPATH=deepseek-ai/DeepSeek-V2-Lite-Chat
        MOVR="++actor.optimizer_dtype=bfloat16 ++sglang.attention_backend=triton" ;;
  q30b) MODELPATH=Qwen/Qwen3-30B-A3B
        MOVR="++actor.optimizer_dtype=bfloat16 actor.mb_spec.max_tokens_per_mb=2048 ref.mb_spec.max_tokens_per_mb=2048 sglang.mem_fraction_static=0.7 rollout.backend=sglang:d2p1t2 ++sglang.disable_custom_all_reduce=true" ;;
  *) echo "unknown CELL"; exit 1 ;;
esac
[ "$SMOKE" = "1" ] && MOVR="$MOVR total_train_epochs=1"

case $METHOD in
  nocorr) OVR="actor.use_decoupled_loss=false" ;;
  fullis) OVR="actor.rejection_sampling.action=clamp actor.rejection_sampling.lower=1e-6 actor.rejection_sampling.upper=1e6" ;;
  tis)    OVR="actor.rejection_sampling.action=clamp actor.rejection_sampling.lower=1e-6 actor.rejection_sampling.upper=2.0" ;;
  icepop) OVR="" ;;
  kpop)   OVR="actor.rejection_sampling.metric=binary_kl actor.rejection_sampling.upper=2.0 ~actor.rejection_sampling.lower" ;;
  kpopfix) OVR="actor.rejection_sampling.metric=binary_kl actor.rejection_sampling.upper=2.0 ~actor.rejection_sampling.lower"
          export KT_KPOP_FIX=1 ;;
  seqtis) OVR="actor.rejection_sampling.action=clamp actor.rejection_sampling.level=sequence +actor.rejection_sampling.agg=sum actor.rejection_sampling.lower=1e-6 actor.rejection_sampling.upper=2.0"
          export KT_SEQ_PROD=1 ;;
  seqmis) OVR="actor.rejection_sampling.action=mask actor.rejection_sampling.level=sequence +actor.rejection_sampling.agg=sum actor.rejection_sampling.lower=0.5 actor.rejection_sampling.upper=2.0"
          export KT_SEQ_PROD=1 ;;
  gspo)   OVR="~actor.rejection_sampling +actor.importance_sampling_level=sequence ++actor.eps_clip=3e-4 ++actor.eps_clip_higher=4e-4" ;;
  fp16)   OVR="actor.use_decoupled_loss=false actor.dtype=float16"
          export KT_FP16_LOSS_SCALE=1024 ;;
  ours)   OVR="actor.rejection_sampling.action=clamp actor.rejection_sampling.lower=1e-6 actor.rejection_sampling.upper=1e6"
          export KT_SIGMA_TIS=1 KT_SIGMA_C1=1.0 KT_SIGMA_B=${LAMP:-2.3} KT_SIGMA_A=1e9 KT_SIGMA_FLOOR=5e-3 ;;
esac
export KT_DUMP_DIR="$RLROOT/ktdump/${CELL}_${METHOD}${TAG}_s${SEED}"
export KT_DUMP_EVERY=20

python examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_icepop.yaml \
    experiment_name=kt-${CELL} \
    trial_name=${METHOD}${TAG}-s${SEED} \
    seed=$SEED \
    scheduler.type=local \
    cluster.fileroot="$RLROOT/experiments" \
    cluster.name_resolve.nfs_record_root="$RLROOT/name_resolve" \
    actor.path=$MODELPATH \
    actor.optimizer.lr=3e-6 \
    actor.optimizer.weight_decay=0.01 \
    ++actor.eps_clip=0.2 \
    ++actor.eps_clip_higher=0.28 \
    actor.mb_spec.max_tokens_per_mb=4096 \
    ref.mb_spec.max_tokens_per_mb=4096 \
    ++train_dataset.setup_timeout=900 \
    ++valid_dataset.setup_timeout=900 \
    ++rollout.setup_timeout=900 \
    total_train_epochs=3 \
    $MOVR $OVR $EXTRA

CK=$RLROOT/experiments/checkpoints/kzhao2/kt-${CELL}/${METHOD}${TAG}-s${SEED}/default
E=$(ls "$CK" 2>/dev/null | grep "^epoch" | tail -1)
[ -n "$E" ] && sbatch --export=ALL,EVAL_PATH="$CK/$E",EVAL_TAG="suite_${CELL}_${METHOD}${TAG}_s${SEED}" /home/kzhao2/gap_measurement/rl/eval_suite.sbatch
