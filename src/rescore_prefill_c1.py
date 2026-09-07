"""Control C1: same-engine (vLLM) prefill re-scoring of sampled sequences.

Purpose (protocol section 2): isolate the "consumption mode" difference --
decode-path kernels vs prefill-path kernels inside the SAME engine with
the SAME weights. The main measurement forbids prefill re-scoring for
logp_infer; here it is the entire point.

  logp_c1[k] = prompt_logprobs[P+k][gen_ids[k]]  (vLLM prefill path)

Compare against logp_infer (decode path) from Stage A.

Usage:
  python rescore_prefill_c1.py --arch moe --shard 0 --subsample 0.05
"""

import argparse
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA_ROOT, GLOBAL_SEED, MAX_MODEL_LEN, MODELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["moe", "dense"], required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--subsample", type=float, default=0.05)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    spec = MODELS[args.arch]
    shard_tag = "smoke" if args.smoke else f"shard{args.shard:03d}"
    gen_dir = os.path.join(DATA_ROOT, "gen", args.arch)
    trajs = pq.read_table(
        os.path.join(gen_dir, f"trajs_{shard_tag}.parquet")
    ).to_pylist()

    rng = np.random.RandomState(4321 + args.shard)
    trajs = [t for t in trajs if rng.rand() < args.subsample]
    if not trajs:
        print("[c1] empty subsample, nothing to do")
        return

    # Same engine config as Stage A generation, minus the capture knobs.
    llm = LLM(
        model=spec["path"],
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        seed=GLOBAL_SEED,
        gpu_memory_utilization=0.85,
        disable_log_stats=True,
    )

    sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0)
    reqs = [
        TokensPrompt(
            prompt_token_ids=t["prompt_token_ids"] + t["gen_token_ids"]
        )
        for t in trajs
    ]
    outputs = llm.generate(reqs, sp)

    out = {"traj_id": [], "pos": [], "logp_c1": []}
    for t, o in zip(trajs, outputs):
        P = len(t["prompt_token_ids"])
        T = len(t["gen_token_ids"])
        plps = o.prompt_logprobs
        assert len(plps) == P + T, (len(plps), P, T)
        for k in range(T):
            entry = plps[P + k]
            tok = t["gen_token_ids"][k]
            out["traj_id"].append(t["traj_id"])
            out["pos"].append(P + k)
            out["logp_c1"].append(entry[tok].logprob)

    table = pa.table(
        out,
        schema=pa.schema([
            ("traj_id", pa.int64()), ("pos", pa.int32()),
            ("logp_c1", pa.float32()),
        ]),
    )
    outdir = os.path.join(DATA_ROOT, "c1", args.arch)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"c1_{shard_tag}.parquet")
    pq.write_table(table, path)
    print(f"[c1] wrote {path} ({table.num_rows} rows)")


if __name__ == "__main__":
    main()
