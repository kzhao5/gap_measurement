# Environment for AReaL RL runs on the dw cluster (FIPS gotchas included)
export OPENSSL_CONF=/dev/null                 # FIPS openssl crashes python ssl
# sglang JIT kernels (tvm-ffi) need nvcc; compute nodes have no
# /usr/local/cuda -- CUDA comes from the module system.
module load cuda/12.8.1 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc 2>/dev/null)")")}"
# Triton JIT compiles a CPython extension at runtime; neither login nor
# compute nodes ship python3.11-devel. Use uv-managed CPython's headers.
# Triton/inductor JIT compiles a CPython extension at runtime (vLLM on H200 hits this);
# compute nodes have no python3-devel, so point gcc at the venv interpreter's own headers.
_PYINC=$($HOME/AReaL/.venv/bin/python -c "import sysconfig;print(sysconfig.get_paths()['include'])" 2>/dev/null)
[ -n "$_PYINC" ] && export C_INCLUDE_PATH=$_PYINC${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}
export HF_HOME=$HOME/nobackup/autodelete/hf
export HF_HUB_CACHE=$HOME/nobackup/autodelete/hf
export HF_HUB_OFFLINE=1                       # compute nodes have no internet
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export UV_CACHE_DIR=$HOME/nobackup/autodelete/uv_cache
AREAL=$HOME/AReaL
RLROOT=$HOME/nobackup/autodelete/areal_rl
mkdir -p "$RLROOT/experiments" "$RLROOT/name_resolve" "$RLROOT/logs"
export AREAL_ALLOW_DEFAULT_ADMIN_KEY=1  # trusted Slurm-internal network
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NOTE: opencv-python-headless must stay uninstalled in AReaL venv (FIPS libcrypto crash); re-remove after any vllm reinstall
