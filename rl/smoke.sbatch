#!/bin/bash
#SBATCH --job-name=areal-smoke
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1500G
#SBATCH --time=04:00:00
#SBATCH --exclude=dw-2-4
#SBATCH --output=/home/kzhao2/nobackup/autodelete/areal_rl/logs/smoke_%j.out

source /home/kzhao2/gap_measurement/rl/env.sh
cd "$AREAL"

# Activate the venv properly: AReaL spawns subprocesses (sglang server,
# workers) via PATH python; bare .venv/bin/python leaks system python
# into children ("init_sys_streams" fatal at io.py).
source .venv/bin/activate

# Warm the NFS page cache: worker spawn has a hardcoded 30s readiness
# timeout (controller._WORKERS_READY_TIMEOUT) and a cold import off NFS
# takes ~80s. One throwaway import makes every spawned worker fast.
python -c "import torch, transformers, datasets, sglang; \
import areal.trainer.rl_trainer" || true

# Single-arm smoke: official IcePop config, our measured MoE model,
# 1 epoch cap -- goal is "trains some steps, logs written", not convergence.
python examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_icepop.yaml \
    experiment_name=kt-smoke-icepop \
    trial_name=moe-a27b-t5 \
    scheduler.type=local \
    cluster.fileroot="$RLROOT/experiments" \
    cluster.name_resolve.nfs_record_root="$RLROOT/name_resolve" \
    actor.path=Qwen/Qwen1.5-MoE-A2.7B-Chat \
    +actor.attn_impl=sdpa \
    +sglang.disable_cuda_graph=true \
    actor.mb_spec.max_tokens_per_mb=4096 \
    ref.mb_spec.max_tokens_per_mb=4096 \
    total_train_epochs=1
# attn_impl note: cluster is A100 (sm80) -- FA3/FA4 unsupported there, and
# the locked flash_attn_4 beta ships no FA2 symbols, so sdpa is the only
# valid trainer backend without compiling real flash-attn.
