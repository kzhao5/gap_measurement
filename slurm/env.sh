# Shared environment for gap_measurement Slurm jobs
# FIPS-mode nodes: uv-managed python bundles its own OpenSSL which cannot
# parse the system FIPS openssl.cnf -> bypass it. (opencv-python-headless
# was also uninstalled from the venv: its bundled OpenSSL 1.1.1k aborts on
# FIPS self-test; re-remove it if vllm is ever reinstalled.)
export OPENSSL_CONF=/dev/null
# Compute nodes have no nvcc: flashinfer's JIT sampler cannot build (code=127).
# Use the torch-native sampler; affects only the RNG stream, not logprob
# semantics (we record the decode-time logprob of whatever token was sampled).
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_CACHE=/home/kzhao2/nobackup/autodelete/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false
# home quota hovers at 100% (other running jobs): keep stdout minimal so a
# failed flush can't kill the task, and logs now live on nobackup anyway.
export HF_HUB_DISABLE_PROGRESS_BARS=1
source /home/kzhao2/gap_measurement/.venv/bin/activate
cd /home/kzhao2/gap_measurement
