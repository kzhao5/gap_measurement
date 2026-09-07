"""Interactive debug: why did MarginCapture record nothing?

Counts wrapper invocations and hook invocations, prints record keys vs
RequestOutput.request_id, prints the model runner class actually in use.
"""

import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import GLOBAL_SEED, HF_CACHE, MAX_MODEL_LEN, MODELS  # noqa: E402

os.environ.setdefault("HF_HUB_CACHE", HF_CACHE)

from vllm import LLM, SamplingParams  # noqa: E402

import gen_vllm  # noqa: E402

spec = MODELS["moe"]

llm = LLM(
    model=spec["path"],
    dtype="bfloat16",
    max_model_len=MAX_MODEL_LEN,
    max_num_seqs=4,
    enforce_eager=True,
    enable_prefix_caching=False,
    async_scheduling=False,
    seed=GLOBAL_SEED,
    gpu_memory_utilization=0.85,
    enable_return_routed_experts=True,
    disable_log_stats=True,
)

cap = gen_vllm.install_capture(llm, spec["num_layers"], spec["top_k"])
cap.enabled = True

# instrument (hooks hold the original bound method, so count via on_step's
# view of step_logits instead of replacing cap.hook)
step_calls = {"n": 0}
orig_on_step = cap.on_step
def counting_on_step(req_order, counts):
    step_calls["n"] += 1
    if step_calls["n"] <= 3:
        print(f"  on_step #{step_calls['n']}: reqs={req_order} counts={counts} "
              f"step_logits={len(cap.step_logits)}")
    return orig_on_step(req_order, counts)
cap.on_step = counting_on_step

# check engine internals
core = llm.llm_engine.engine_core
inner = getattr(core, "engine_core", core)
executor = inner.model_executor
driver = getattr(executor, "driver_worker", None)
worker = getattr(driver, "worker", driver)
mr = worker.model_runner
print("model runner class:", type(mr).__module__, type(mr).__name__)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
print("is GPUModelRunner instance:", isinstance(mr, GPUModelRunner))
print("execute_model is wrapped:",
      type(mr).execute_model.__qualname__, type(mr).execute_model)

sp = SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=16,
                    logprobs=0, seed=1)
outs = llm.generate(["Compute 2+2.", "What is the capital of France?"], sp)
cap.enabled = False

print(f"on_step_calls={step_calls['n']}")
print("record keys:", list(cap.records.keys())[:8])
for o in outs:
    print("request_id:", repr(o.request_id),
          "P:", len(o.prompt_token_ids),
          "T:", len(o.outputs[0].token_ids),
          "routed_experts:",
          None if o.outputs[0].routed_experts is None
          else o.outputs[0].routed_experts.shape)
    rec = cap.records.get(str(o.request_id))
    print("  margin entries:", None if rec is None else len(rec))
