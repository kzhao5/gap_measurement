"""Join smoke-run outputs and print sanity statistics.

Checks:
  - token tables join 1:1 on (traj_id, pos)
  - D = logp_train - logp_infer: finite, plausible sigma
  - MoE: flip rate, margin ranges, flip vs |D| relation
  - C1/C2 columns join and their deltas are small
"""

import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import DATA_ROOT, MODELS, route_flip


def load(path):
    return pq.read_table(path).to_pandas()


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else "moe"
    spec = MODELS[arch]
    is_moe = spec["is_moe"]

    tok = load(os.path.join(DATA_ROOT, "gen", arch, "tokens_smoke.parquet"))
    rec = load(
        os.path.join(DATA_ROOT, "recompute", arch, "recomp_smoke_bf16.parquet")
    )
    print(f"[{arch}] gen tokens: {len(tok)}, recompute rows: {len(rec)}")

    df = tok.merge(rec, on=["traj_id", "pos"], how="inner", validate="1:1")
    assert len(df) == len(tok) == len(rec), (len(df), len(tok), len(rec))

    D = df["logp_train"].to_numpy() - df["logp_infer"].to_numpy()
    assert np.isfinite(D).all(), "non-finite D"
    mad = np.median(np.abs(D - np.median(D)))
    sigma = mad / 0.6745
    print(f"D: mean={D.mean():.3e} median={np.median(D):.3e} "
          f"sigma_MAD={sigma:.3e} max|D|={np.abs(D).max():.3e}")

    if is_moe:
        L, K = spec["num_layers"], spec["top_k"]
        ri = np.stack(df["route_infer"].to_numpy()).reshape(-1, L, K)
        rt = np.stack(df["route_train"].to_numpy()).reshape(-1, L, K)
        flip = route_flip(ri, rt)
        eps = flip.mean()
        n_layer_mismatch = (
            (np.sort(ri, -1) != np.sort(rt, -1)).any(-1).sum(-1)
        )
        mi = np.stack(df["margin_infer"].to_numpy())
        print(f"flip rate eps={eps:.4f}  "
              f"layers-mismatched (flipped tokens): "
              f"{n_layer_mismatch[flip].mean() if flip.any() else 0:.2f}")
        print(f"margin_infer: min={mi.min():.3e} med={np.median(mi):.3e}")
        if flip.any() and (~flip).any():
            print(f"|D| consistent: {np.abs(D[~flip]).mean():.3e}  "
                  f"|D| flipped: {np.abs(D[flip]).mean():.3e}")
        # margin_min for flipped vs not
        mmin = mi.min(axis=1)
        if flip.any():
            print(f"margin_min: flipped med={np.median(mmin[flip]):.3e}  "
                  f"consistent med={np.median(mmin[~flip]):.3e}")

    c1p = os.path.join(DATA_ROOT, "c1", arch, "c1_smoke.parquet")
    if os.path.exists(c1p):
        c1 = load(c1p)
        j = df.merge(c1, on=["traj_id", "pos"], how="inner")
        dc1 = j["logp_c1"].to_numpy() - j["logp_infer"].to_numpy()
        print(f"C1 (same-engine prefill - decode): n={len(j)} "
              f"mean={dc1.mean():.3e} sigma_MAD="
              f"{np.median(np.abs(dc1 - np.median(dc1))) / 0.6745:.3e}")

    c2a = os.path.join(
        DATA_ROOT, "recompute", arch, "recomp_smoke_bf16_c2a.parquet"
    )
    c2b = c2a.replace("c2a", "c2b")
    if os.path.exists(c2a) and os.path.exists(c2b):
        a = load(c2a).rename(columns={"logp_train": "lp_a"})
        b = load(c2b).rename(columns={"logp_train": "lp_b"})
        j = a[["traj_id", "pos", "lp_a"]].merge(
            b[["traj_id", "pos", "lp_b"]], on=["traj_id", "pos"]
        )
        d22 = (j["lp_a"] - j["lp_b"]).to_numpy()
        print(f"C2 determinism floor: n={len(j)} max|delta|={np.abs(d22).max():.3e} "
              f"nonzero frac={(d22 != 0).mean():.4f}")

    print("SMOKE OK")


if __name__ == "__main__":
    main()
