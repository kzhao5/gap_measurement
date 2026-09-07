"""Deep debug: compare hook router-logits-derived topk against the native
routing_data ROW-BY-ROW in the same flat batch order, per layer.

If per-row agreement is high -> flat order + topk math fine, and the bug is
in per-request attribution. If a layer permutation fixes it -> layer order.
If low everywhere -> topk semantics differ.
"""

import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import GLOBAL_SEED, HF_CACHE, MAX_MODEL_LEN, MODELS  # noqa: E402
from common import routing_from_router_logits  # noqa: E402

os.environ.setdefault("HF_HUB_CACHE", HF_CACHE)

from vllm import LLM, SamplingParams  # noqa: E402

import gen_vllm  # noqa: E402

spec = MODELS["moe"]
L, K = spec["num_layers"], spec["top_k"]

llm = LLM(
    model=spec["path"], dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
    max_num_seqs=8, enforce_eager=True, enable_prefix_caching=False,
    async_scheduling=False, seed=GLOBAL_SEED, gpu_memory_utilization=0.85,
    enable_return_routed_experts=True, disable_log_stats=True,
)
cap = gen_vllm.install_capture(llm, L, K)
cap.enabled = True

orig_on_step = cap.on_step
def stashing_on_step(req_order, counts):
    cap.last_step_logits = [t.detach().clone() for t in cap.step_logits]
    return orig_on_step(req_order, counts)
cap.on_step = stashing_on_step

from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # noqa: E402
wrapped = GPUModelRunner.execute_model
agree = np.zeros(L, dtype=np.int64)
agree_rev = np.zeros(L, dtype=np.int64)
rows_tot = {"n": 0}

def outer(self, scheduler_output, *a, **k):
    out = wrapped(self, scheduler_output, *a, **k)
    re = getattr(out, "routed_experts", None)
    sl = getattr(cap, "last_step_logits", None)
    if re is not None and sl is not None and len(sl) == L:
        nat = re.routing_data
        if hasattr(nat, "cpu"):
            nat = nat.cpu().numpy()
        nat = np.asarray(nat)
        n = nat.shape[0]
        for li, logits in enumerate(sl):
            ids_l, _ = routing_from_router_logits(logits[:n], K)
            hook_l = np.sort(ids_l.cpu().numpy(), -1)
            agree[li] += (hook_l == np.sort(nat[:, li, :], -1)).all(-1).sum()
            agree_rev[li] += (
                hook_l == np.sort(nat[:, L - 1 - li, :], -1)
            ).all(-1).sum()
        rows_tot["n"] += n
    cap.last_step_logits = None
    return out

GPUModelRunner.execute_model = outer

sp = SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=32,
                    logprobs=0, seed=1)
prompts = ["Compute 2+2.", "What is the capital of France?",
           "Name a prime number above 100.", "Translate 'hello' to Spanish."]
outs = llm.generate(prompts, sp)
cap.enabled = False

n = max(rows_tot["n"], 1)
print(f"rows compared: {rows_tot['n']}")
print("per-layer agreement (same order):", (agree / n).round(3).tolist())
print("per-layer agreement (reversed):  ", (agree_rev / n).round(3).tolist())

# also dump one step's native dtype/shape info
print("native routing dtype/shape example available above")
