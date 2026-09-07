"""Gradient-proxy harvest for the token-level MSE fixed-point experiment.

One extra teacher-forcing forward over the campaign trajectories (same
batching/position convention as recompute_train.py: row t-1 produces token t),
storing per generated token:
  h_norm : ||h||_2 of the lm_head INPUT (post final-norm hidden), captured
           via a forward hook on model.model.norm
  g_proj : (e_y - pi)^T R, R = vocab x 128 N(0,1)/sqrt(128), seed 777
           (identical across shards: numpy-generated then moved to GPU)

Usage: python harvest_grad.py --shard 0 [--smoke]
"""

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA_ROOT, MODELS

TOKEN_BUDGET = 8192
PROJ_DIM = 128
HPROJ_DIM = 32
PROJ_SEED = 777


def build_batches(trajs, token_budget=TOKEN_BUDGET):
    order = sorted(range(len(trajs)), key=lambda i: -trajs[i]["total_len"])
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = trajs[i]["total_len"]
        new_max = max(cur_max, L)
        if cur and new_max * (len(cur) + 1) > token_budget:
            batches.append(cur)
            cur, cur_max = [i], L
        else:
            cur.append(i)
            cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    spec = MODELS["moe"]
    tag = f"shard{args.shard:03d}"
    trajs = pq.read_table(
        os.path.join(DATA_ROOT, "gen", "moe", f"trajs_{tag}.parquet")
    ).to_pylist()
    if args.smoke:
        trajs = trajs[:24]
    for t in trajs:
        t["total_len"] = len(t["prompt_token_ids"]) + len(t["gen_token_ids"])

    model = AutoModelForCausalLM.from_pretrained(
        spec["path"], dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda:0")
    model.eval()

    # capture the lm_head input (post final-norm hidden state)
    cap = {}
    model.model.norm.register_forward_hook(
        lambda mod, i, o: cap.__setitem__("h", o))

    V = model.get_output_embeddings().weight.shape[0]
    H = model.config.hidden_size
    rng = np.random.RandomState(PROJ_SEED)
    R = torch.tensor(
        (rng.randn(V, PROJ_DIM) / np.sqrt(PROJ_DIM)).astype(np.float32),
        dtype=torch.float32, device="cuda")
    rng_h = np.random.RandomState(PROJ_SEED + 1)
    Rh = torch.tensor(
        (rng_h.randn(H, HPROJ_DIM) / np.sqrt(HPROJ_DIM)).astype(np.float32),
        dtype=torch.float32, device="cuda")

    out = {"traj_id": [], "pos": [], "h_norm": [], "g_proj": [],
           "h_proj": []}
    batches = build_batches(trajs)
    with torch.no_grad():
        for bi, batch in enumerate(batches):
            seqs = [trajs[i] for i in batch]
            maxlen = max(t["total_len"] for t in seqs)
            input_ids = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for r, t in enumerate(seqs):
                ids = t["prompt_token_ids"] + t["gen_token_ids"]
                input_ids[r, : len(ids)] = torch.tensor(ids)
                attn[r, : len(ids)] = 1
            res = model(input_ids=input_ids.cuda(), attention_mask=attn.cuda(),
                        use_cache=False)
            h = cap["h"]                    # [B, S, H] bf16
            logits = res.logits
            for r, t in enumerate(seqs):
                P = len(t["prompt_token_ids"])
                T = len(t["gen_token_ids"])
                rows = torch.arange(P - 1, P + T - 1, device=logits.device)
                gen = torch.tensor(t["gen_token_ids"], device=logits.device)
                hrow = h[r, rows].float()
                hn = hrow.norm(dim=-1)                            # [T]
                hp = hrow @ Rh                                    # [T, 32]
                probs = torch.softmax(logits[r, rows].float(), -1)  # [T, V]
                gp = R[gen] - probs @ R                           # [T, 128]
                out["traj_id"].extend([t["traj_id"]] * T)
                out["pos"].extend(range(P, P + T))
                out["h_norm"].extend(hn.cpu().tolist())
                out["g_proj"].extend(
                    gp.half().cpu().numpy().tolist())
                out["h_proj"].extend(
                    hp.half().cpu().numpy().tolist())
            if bi % 20 == 0:
                print(f"[harvest] batch {bi}/{len(batches)}", flush=True)

    # sanity: h_norm positive/finite, g_proj finite and mean-ish small
    hn = np.asarray(out["h_norm"])
    gp = np.asarray(out["g_proj"], dtype=np.float32)
    assert np.isfinite(hn).all() and (hn > 0).all(), "bad h_norm"
    assert np.isfinite(gp).all(), "bad g_proj"
    print(f"[harvest] sanity: h_norm med {np.median(hn):.2f}, "
          f"|g_proj| med {np.median(np.abs(gp)):.4f}, "
          f"mean g_proj {gp.mean():.2e}")

    outdir = os.path.join(DATA_ROOT, "harvest", "moe")
    os.makedirs(outdir, exist_ok=True)
    stem = "smoke" if args.smoke else tag
    table = pa.table({
        "traj_id": pa.array(out["traj_id"], pa.int64()),
        "pos": pa.array(out["pos"], pa.int32()),
        "h_norm": pa.array(out["h_norm"], pa.float32()),
        "g_proj": pa.array(out["g_proj"], pa.list_(pa.float16(), PROJ_DIM)),
        "h_proj": pa.array(out["h_proj"], pa.list_(pa.float16(), HPROJ_DIM)),
    })
    pq.write_table(table, os.path.join(outdir, f"harv_{stem}.parquet"))
    print(f"[harvest] wrote harv_{stem}.parquet ({len(hn)} tokens)")


if __name__ == "__main__":
    main()
