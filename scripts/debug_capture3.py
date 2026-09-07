"""Ultimate arbitration: hook router-logits topk vs the capturer's own
device buffer, same step, same flat order. Tests row order, layer order,
and topk semantics independently."""

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
agree_same = np.zeros(L, dtype=np.int64)
agree_rev = np.zeros(L, dtype=np.int64)
best_perm_votes = np.zeros((L, L), dtype=np.int64)  # [hook_layer, native_layer]
rows_tot = {"n": 0, "steps": 0}
THRESHOLDS = [0.0, 1e-4, 1e-3, 5e-3, 1e-2]
cond_tot = np.zeros(len(THRESHOLDS), dtype=np.int64)
cond_agree = np.zeros(len(THRESHOLDS), dtype=np.int64)
disagree_margins = []
agree_margins_min = {"v": 1.0}

def outer(self, scheduler_output, *a, **k):
    out = wrapped(self, scheduler_output, *a, **k)
    sl = getattr(cap, "last_step_logits", None)
    total = scheduler_output.total_num_scheduled_tokens
    if sl is not None and len(sl) == L and total > 0:
        buf = self.routed_experts_capturer.get_device_buffer()
        nat = buf[:total].cpu().numpy()  # [total, L, K] presumably
        if rows_tot["steps"] == 0:
            print("device buffer shape:", tuple(buf.shape), "dtype:", buf.dtype)
        nat_sorted = np.sort(nat, -1)
        for li, logits in enumerate(sl):
            ids_l, margin_l = routing_from_router_logits(logits[:total], K)
            hook_l = np.sort(ids_l.cpu().numpy(), -1)
            same = (hook_l == nat_sorted[:, li, :]).all(-1)
            agree_same[li] += same.sum()
            agree_rev[li] += (hook_l == nat_sorted[:, L - 1 - li, :]).all(-1).sum()
            mg = margin_l.cpu().numpy()
            disagree_margins.extend(mg[~same].tolist())
            agree_margins_min["v"] = min(
                agree_margins_min["v"],
                float(mg[same].min()) if same.any() else 1.0,
            )
            # margin-conditional agreement
            for thr_i, thr in enumerate(THRESHOLDS):
                m = mg > thr
                cond_tot[thr_i] += m.sum()
                cond_agree[thr_i] += (same & m).sum()
            if rows_tot["steps"] < 3:
                for lj in range(L):
                    best_perm_votes[li, lj] += (
                        (hook_l == nat_sorted[:, lj, :]).all(-1).sum()
                    )
        rows_tot["n"] += total
        rows_tot["steps"] += 1
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
print(f"rows compared: {rows_tot['n']} over {rows_tot['steps']} steps")
print("per-layer agreement (same layer):", (agree_same / n).round(3).tolist())
print("per-layer agreement (reversed):  ", (agree_rev / n).round(3).tolist())
print("argmax native layer per hook layer (first steps):",
      best_perm_votes.argmax(1).tolist())
dm = np.array(disagree_margins)
print(f"disagreements: n={len(dm)}, margin median={np.median(dm):.2e}, "
      f"q95={np.quantile(dm, 0.95):.2e}, max={dm.max():.2e}" if len(dm)
      else "no disagreements")
for thr, t, a in zip(THRESHOLDS, cond_tot, cond_agree):
    print(f"  agreement | margin>{thr:g}: {a}/{t} = {a / max(t, 1):.5f}")
