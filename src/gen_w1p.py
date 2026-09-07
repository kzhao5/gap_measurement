"""(W1') lean Stage A: vLLM generation with decode-time logprob capture only.

Same protocol as gen_vllm.py (temperature=1, top_p=1, no truncation; logp_infer
is the sampling step's own logprob for the sampled token -- raw engine value,
no post-hoc rescoring) but WITHOUT the routing/margin capture machinery, so
CUDA graphs stay on and generation is fast. Output schema matches Stage A
(tokens_/trajs_ parquet) so recompute_train.py runs unchanged; shard tags are
offset to 9xx to keep a separate namespace from the main campaign.

Usage: python gen_w1p.py --shard 0 --num-shards 4
"""

import argparse
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    DATA_ROOT,
    GLOBAL_SEED,
    MAX_MODEL_LEN,
    MODELS,
    PROMPTS_JSONL,
    SAMPLING,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--max-prompts", type=int, default=250,
                    help="prompts per shard (n_traj = this * group)")
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--tag-base", type=int, default=900)
    ap.add_argument("--skip-prompts", type=int, default=0,
                    help="skip this many prompts per shard (disjoint batches)")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="added to all RNG seeds (independent replications)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    spec = MODELS["moe"]
    prompts = [json.loads(l) for l in open(PROMPTS_JSONL)]
    prompts = [p for i, p in enumerate(prompts)
               if i % args.num_shards == args.shard]
    prompts = prompts[args.skip_prompts: args.skip_prompts + args.max_prompts]

    tok = AutoTokenizer.from_pretrained(spec["path"])
    chat_texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p["text"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    llm = LLM(
        model=spec["path"],
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=128,
        enable_prefix_caching=False,
        # MANDATORY on this venv (vllm 0.26 + torch 2.11): the CUDA-graph
        # decode path emits a corrupted distribution (gibberish tokens,
        # instant im_end, log_k ~ -3 vs the training-side forward). Every
        # healthy run (campaign gen, GSM8K evals) used eager. Verified
        # 2026-08-12 on job 13157939's first attempt.
        enforce_eager=True,
        seed=GLOBAL_SEED + 77 + args.shard + args.seed_offset,
        gpu_memory_utilization=0.85,
        disable_log_stats=True,
    )

    sps, metas = [], []
    for gi, p in enumerate(prompts):
        for s in range(args.group):
            # +500 offsets keep traj ids and RNG streams disjoint from the
            # main campaign's (s in 0..7 there).
            sps.append(SamplingParams(
                temperature=SAMPLING["temperature"],
                top_p=SAMPLING["top_p"],
                top_k=SAMPLING["top_k"],
                max_tokens=SAMPLING["max_tokens"],
                logprobs=0,
                seed=GLOBAL_SEED * 7 + p["prompt_id"] * 100 + 50 + s
                     + args.seed_offset,
            ))
            metas.append({"prompt_id": p["prompt_id"],
                          "traj_id": p["prompt_id"] * 1000 + 500 + args.seed_offset + s,
                          "chat_idx": gi})

    outputs = llm.generate([chat_texts[m["chat_idx"]] for m in metas], sps)

    tok_rows = {"traj_id": [], "prompt_id": [], "pos": [], "token_id": [],
                "logp_infer": []}
    traj_rows = {"traj_id": [], "prompt_id": [], "prompt_token_ids": [],
                 "gen_token_ids": [], "finish_reason": [], "seed": []}
    n_tokens = 0
    for i, out in enumerate(outputs):
        m = metas[i]
        comp = out.outputs[0]
        P = len(out.prompt_token_ids)
        gen_ids = list(comp.token_ids)
        T = len(gen_ids)
        if T == 0:
            continue
        lps = [comp.logprobs[t][gen_ids[t]].logprob for t in range(T)]
        traj_rows["traj_id"].append(m["traj_id"])
        traj_rows["prompt_id"].append(m["prompt_id"])
        traj_rows["prompt_token_ids"].append(list(out.prompt_token_ids))
        traj_rows["gen_token_ids"].append(gen_ids)
        traj_rows["finish_reason"].append(str(comp.finish_reason))
        traj_rows["seed"].append(sps[i].seed)
        for t in range(T):
            tok_rows["traj_id"].append(m["traj_id"])
            tok_rows["prompt_id"].append(m["prompt_id"])
            tok_rows["pos"].append(P + t)
            tok_rows["token_id"].append(gen_ids[t])
            tok_rows["logp_infer"].append(lps[t])
        n_tokens += T

    # Sanity guard: healthy math generations run hundreds of tokens; a tiny
    # median T means the decode distribution is corrupted -> fail loudly
    # instead of writing poison for the analysis stage.
    lens = sorted(len(g) for g in traj_rows["gen_token_ids"])
    med_T = lens[len(lens) // 2] if lens else 0
    if med_T < 30:
        raise RuntimeError(
            f"median gen length {med_T} < 30: decode path corrupted "
            "(see enforce_eager note); refusing to write output")

    outdir = os.path.join(DATA_ROOT, "gen", "moe")
    os.makedirs(outdir, exist_ok=True)
    tag = f"shard{args.tag_base + args.shard:03d}"
    fields = [("traj_id", pa.int64()), ("prompt_id", pa.int32()),
              ("pos", pa.int32()), ("token_id", pa.int32()),
              ("logp_infer", pa.float32())]
    pq.write_table(
        pa.table({k: tok_rows[k] for k, _ in fields}, schema=pa.schema(fields)),
        os.path.join(outdir, f"tokens_{tag}.parquet"))
    pq.write_table(pa.table(traj_rows),
                   os.path.join(outdir, f"trajs_{tag}.parquet"))
    print(f"[gen_w1p] wrote {tag}: {len(traj_rows['traj_id'])} trajs, "
          f"{n_tokens} tokens")


if __name__ == "__main__":
    main()
