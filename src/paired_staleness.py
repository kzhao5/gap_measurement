"""Paired fresh/stale staleness measurement (VERIFY_Z_B_CONTAMINATION_MODEL).

Same tokens, same prefixes, scored under multiple engine versions of one
RL-trained policy (kt-lgrid/lg0-e3: v28 / v57 / v86 = epoch checkpoints):

  stage gen  : vLLM(v57, eager) generates GSM8K rollouts (the STALE engine,
               as in real RL) + decode-time logp        -> logp_stale_dec
  stage pre  : vLLM(X, eager) prefill-rescore same seqs -> logp_pre_<tag>
               for X in {v57 (control), v86 (fresh), v28 (very stale)}
  stage hf   : HF(v86, bf16, sdpa) teacher-force        -> logp_old

Then B^(29) = pre_v86 - pre_v57, B^(58) = pre_v86 - pre_v28 (paired,
same stack so stack numerics cancel), logk0 = old - pre_v86 (clean
numerical mismatch), p = exp(logp_old).

Usage:
  python paired_staleness.py --stage gen --shard 0
  python paired_staleness.py --stage pre --shard 0 --model v57
  python paired_staleness.py --stage hf  --shard 0
"""

import argparse
import json
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA_ROOT

CKROOT = ("/home/kzhao2/nobackup/autodelete/areal_rl/experiments/"
          "checkpoints/kzhao2/kt-lgrid/lg0-e3/default")
CKPTS = {
    "v28": f"{CKROOT}/epoch0epochstep28globalstep28",
    "v57": f"{CKROOT}/epoch1epochstep28globalstep57",
    "v86": f"{CKROOT}/epoch2epochstep28globalstep86",
}
OUTDIR = os.path.join(DATA_ROOT, "paired")
N_PROMPTS = 250
GROUP = 4
MAX_TOKENS = 512
SEED = 20260813


def prompts_for(shard, num_shards):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    qs = [(i, ds[i]["question"]) for i in range(len(ds))
          if i % num_shards == shard][:N_PROMPTS]
    return qs


def traj_path(shard):
    return os.path.join(OUTDIR, f"trajs_s{shard}.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "pre", "hf"], required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--model", default="v57", choices=list(CKPTS))
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    if args.stage == "gen":
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        tok = AutoTokenizer.from_pretrained(CKPTS["v57"])
        qs = prompts_for(args.shard, args.num_shards)
        chats = [tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False,
            add_generation_prompt=True) for _, q in qs]
        llm = LLM(model=CKPTS["v57"], dtype="bfloat16", max_model_len=1024,
                  enforce_eager=True, enable_prefix_caching=False,
                  seed=SEED + args.shard, gpu_memory_utilization=0.85,
                  disable_log_stats=True)
        sps, metas = [], []
        for gi, (pid, _) in enumerate(qs):
            for s in range(GROUP):
                sps.append(SamplingParams(
                    temperature=1.0, top_p=1.0, top_k=-1,
                    max_tokens=MAX_TOKENS, logprobs=0,
                    seed=SEED * 7 + pid * 10 + s))
                metas.append((pid * 10 + s, pid, gi))
        outs = llm.generate([chats[m[2]] for m in metas], sps)
        rows = {"traj_id": [], "prompt_id": [], "prompt_token_ids": [],
                "gen_token_ids": [], "logp_stale_dec": []}
        for (tid, pid, _), o in zip(metas, outs):
            comp = o.outputs[0]
            gen = list(comp.token_ids)
            if not gen:
                continue
            rows["traj_id"].append(tid)
            rows["prompt_id"].append(pid)
            rows["prompt_token_ids"].append(list(o.prompt_token_ids))
            rows["gen_token_ids"].append(gen)
            rows["logp_stale_dec"].append(
                [comp.logprobs[t][gen[t]].logprob for t in range(len(gen))])
        pq.write_table(pa.table(rows), traj_path(args.shard))
        n = sum(len(g) for g in rows["gen_token_ids"])
        print(f"[gen] shard {args.shard}: {len(rows['traj_id'])} trajs, "
              f"{n} tokens")
        return

    trajs = pq.read_table(traj_path(args.shard)).to_pylist()

    if args.stage == "pre":
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        llm = LLM(model=CKPTS[args.model], dtype="bfloat16",
                  max_model_len=1024 + MAX_TOKENS, enforce_eager=True,
                  enable_prefix_caching=False, seed=SEED,
                  gpu_memory_utilization=0.85, disable_log_stats=True)
        sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0)
        reqs = [TokensPrompt(
            prompt_token_ids=t["prompt_token_ids"] + t["gen_token_ids"])
            for t in trajs]
        outs = llm.generate(reqs, sp)
        col = {"traj_id": [], "pos": [], "logp": []}
        for t, o in zip(trajs, outs):
            P = len(t["prompt_token_ids"])
            gen = t["gen_token_ids"]
            plps = o.prompt_logprobs
            for k, y in enumerate(gen):
                col["traj_id"].append(t["traj_id"])
                col["pos"].append(P + k)
                col["logp"].append(plps[P + k][y].logprob)
        pq.write_table(pa.table(col), os.path.join(
            OUTDIR, f"pre_{args.model}_s{args.shard}.parquet"))
        print(f"[pre {args.model}] shard {args.shard}: {len(col['pos'])} rows")
        return

    # stage hf: teacher-forcing logp under v86 (the trainer / pi_old)
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        CKPTS["v86"], dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda:0")
    model.eval()
    col = {"traj_id": [], "pos": [], "logp": []}
    order = sorted(range(len(trajs)), key=lambda i: -(
        len(trajs[i]["prompt_token_ids"]) + len(trajs[i]["gen_token_ids"])))
    B = 8
    with torch.no_grad():
        for bs in range(0, len(order), B):
            seqs = [trajs[i] for i in order[bs:bs + B]]
            lens = [len(t["prompt_token_ids"]) + len(t["gen_token_ids"])
                    for t in seqs]
            mx = max(lens)
            ii = torch.zeros((len(seqs), mx), dtype=torch.long)
            am = torch.zeros((len(seqs), mx), dtype=torch.long)
            for r, t in enumerate(seqs):
                ids = t["prompt_token_ids"] + t["gen_token_ids"]
                ii[r, :len(ids)] = torch.tensor(ids)
                am[r, :len(ids)] = 1
            lg = model(input_ids=ii.cuda(), attention_mask=am.cuda(),
                       use_cache=False).logits
            for r, t in enumerate(seqs):
                P = len(t["prompt_token_ids"])
                T = len(t["gen_token_ids"])
                rows = torch.arange(P - 1, P + T - 1, device=lg.device)
                gen = torch.tensor(t["gen_token_ids"], device=lg.device)
                ls = torch.log_softmax(lg[r, rows].float(), -1)
                lp = ls.gather(1, gen.view(-1, 1)).squeeze(1)
                col["traj_id"].extend([t["traj_id"]] * T)
                col["pos"].extend(range(P, P + T))
                col["logp"].extend(lp.cpu().tolist())
    pq.write_table(pa.table(col), os.path.join(
        OUTDIR, f"hf_v86_s{args.shard}.parquet"))
    print(f"[hf] shard {args.shard}: {len(col['pos'])} rows")


if __name__ == "__main__":
    main()
