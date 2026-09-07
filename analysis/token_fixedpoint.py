"""Token-level MSE fixed-point iteration (token_level_fixedpoint_guide.md).

Estimand: G = E[k l'], l' = A_hat * grad log pi. MSE(M) = ||E[(M-k)l']||^2
+ (1/n) E[M^2 ||l'||^2]. Pointwise best response per p-bucket:
  C(p) = n <-b, v(p)> / s(p),  b = E[(m-k)l'], v = E[l'|p], s = E[||l'||^2|p]

Two gradient proxies:
  norm      gnorm = |A| * (1-p) * h_norm            (scalar, rho=1)
  direction gvec  = (A * h_norm) * g_proj           (128-dim projection)

Pre-registered readout: regress converged log C*(p) on log(1-p) -> beta.
beta ~ -1: token-MSE wants increasing cap (refuted by RL, negative closure)
beta ~ 0: constant (same as sequence level)
beta > 0: decreasing band -> MSE route revives; compare (1-p)^1 vs GPD form.

Usage: python token_fixedpoint.py [--shards 0,1,2,3] [--smoke]
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import CODE_ROOT, DATA_ROOT, MODELS

N_TOK = 2.5e5          # tokens per gradient step in our RL recipe (center)
N_SENS = (2.5e4, 2.5e6)
N_BUCKET = 16
ETA = 0.5
MAX_IT = 40
TOL = 0.02
ESS_MIN = 50
OUT = os.path.join(CODE_ROOT, "results", "tokenfp")
FIG = os.path.join(CODE_ROOT, "results", "figs", "tokenfp")


# ------------------------------------------------------------------ rewards
BOX = re.compile(r"\\boxed\{([^{}]+)\}")
ANS = re.compile(r"[Aa]nswer[^0-9\-]*(-?[\d.,/]+)")
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def extract(text):
    m = BOX.findall(text)
    if m:
        return m[-1]
    m = ANS.findall(text)
    if m:
        return m[-1]
    m = NUM.findall(text)
    return m[-1] if m else None


def norm_ans(s):
    if s is None:
        return None
    s = str(s).strip().rstrip(".").replace(",", "").replace("$", "") \
        .replace("%", "").replace(" ", "")
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            return round(float(a) / float(b), 6)
        return round(float(s), 6)
    except (ValueError, ZeroDivisionError):
        return s


def compute_A(shards):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS["moe"]["path"])
    gold = {int(k): v for k, v in json.load(
        open(os.path.join(CODE_ROOT, "data", "prompts_answers.json"))).items()}
    rows = []
    for s in shards:
        tr = pq.read_table(os.path.join(
            DATA_ROOT, "gen", "moe", f"trajs_shard{s:03d}.parquet"),
            columns=["traj_id", "prompt_id", "gen_token_ids"]).to_pylist()
        texts = tok.batch_decode([t["gen_token_ids"] for t in tr],
                                 skip_special_tokens=True)
        for t, tx in zip(tr, texts):
            r = float(norm_ans(extract(tx)) == norm_ans(gold[t["prompt_id"]]))
            rows.append((t["traj_id"], t["prompt_id"], r))
    df = pd.DataFrame(rows, columns=["traj_id", "prompt_id", "r"])
    g = df.groupby("prompt_id")["r"]
    df["A"] = (df["r"] - g.transform("mean")) / (g.transform("std") + 1e-6)
    print(f"[A] acc={df.r.mean():.3f}, frac |A|>0 = "
          f"{(df.A.abs() > 1e-9).mean():.3f}")
    return df[["traj_id", "A"]], float(df.r.mean())


# ------------------------------------------------------------------ iteration
def bucket_stats(pbin, g, g2):
    v = np.bincount(pbin, weights=g, minlength=N_BUCKET)
    s = np.bincount(pbin, weights=g2, minlength=N_BUCKET)
    n = np.bincount(pbin, minlength=N_BUCKET)
    g4 = np.bincount(pbin, weights=g2 ** 2, minlength=N_BUCKET)
    ess = np.where(g4 > 0, s ** 2 / np.maximum(g4, 1e-300), 0.0)
    return v / np.maximum(n, 1), s / np.maximum(n, 1), n, ess


def interp_gated(logC, ess):
    """replace low-ESS buckets by neighbor interpolation."""
    bad = ess < ESS_MIN
    if bad.any() and not bad.all():
        idx = np.arange(N_BUCKET)
        logC[bad] = np.interp(idx[bad], idx[~bad], logC[~bad])
    return logC


def iterate(k, pbin, G, n_tok, logC0, gvec=None):
    """norm version if gvec is None else direction version.
    G = gnorm (scalar proxy). Returns (logC, history)."""
    g2 = G ** 2 if gvec is None else (gvec ** 2).sum(1)
    v_b, s_b, n_b, ess = bucket_stats(pbin, G, g2)
    if gvec is not None:
        vdir = np.zeros((N_BUCKET, gvec.shape[1]))
        for b in range(N_BUCKET):
            m = pbin == b
            vdir[b] = gvec[m].mean(0)
    logC = logC0.copy()
    hist = []
    for r in range(MAX_IT):
        m = np.minimum(k, np.exp(logC[pbin]))
        if gvec is None:
            bias = float(np.mean((m - k) * G))
            num = np.maximum(-bias, 1e-12) * v_b
        else:
            bvec = ((m - k)[:, None] * gvec).mean(0)
            num = np.maximum(vdir @ (-bvec), 1e-12)
        C_new = n_tok * num / np.maximum(s_b, 1e-300)
        logC_next = (1 - ETA) * logC + ETA * np.log(np.maximum(C_new, 1e-8))
        logC_next = interp_gated(logC_next, ess)
        hist.append(dict(r=r, dmax=float(np.max(np.abs(logC_next - logC)))))
        if hist[-1]["dmax"] < TOL:
            logC = logC_next
            break
        logC = logC_next
    return logC, hist


def heldout_mse(k, pbin, G, logC, n_tok, gvec=None):
    m = np.minimum(k, np.exp(logC[pbin]))
    if gvec is None:
        return float(np.mean((m - k) * G) ** 2
                     + np.mean(m ** 2 * G ** 2) / n_tok)
    b = ((m - k)[:, None] * gvec).mean(0)
    return float(b @ b + np.mean(m ** 2 * (gvec ** 2).sum(1)) / n_tok)


def one_step_Cstar(k, pbin, G, logC, n_tok, gvec=None):
    """one BR step evaluated on (held-out) data -> C* per bucket."""
    g2 = G ** 2 if gvec is None else (gvec ** 2).sum(1)
    v_b, s_b, n_b, ess = bucket_stats(pbin, G, g2)
    m = np.minimum(k, np.exp(logC[pbin]))
    if gvec is None:
        bias = float(np.mean((m - k) * G))
        num = np.maximum(-bias, 1e-12) * v_b
    else:
        vdir = np.zeros((N_BUCKET, gvec.shape[1]))
        for b in range(N_BUCKET):
            vdir[b] = gvec[pbin == b].mean(0)
        bvec = ((m - k)[:, None] * gvec).mean(0)
        num = np.maximum(vdir @ (-bvec), 1e-12)
    Cs = n_tok * num / np.maximum(s_b, 1e-300)
    return np.log(np.maximum(Cs, 1e-8)), ess, n_b


def wls(x, y, w):
    sw = np.sqrt(w)
    X = np.column_stack([np.ones_like(x), x]) * sw[:, None]
    beta, *_ = np.linalg.lstsq(X, y * sw, rcond=None)
    resid = y - (beta[0] + beta[1] * x)
    return beta, float(np.sum(w * resid ** 2))


def bic(w_rss, n_pts, kpar):
    return n_pts * np.log(w_rss / n_pts) + kpar * np.log(n_pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="0,1,2,3")
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    shards = [int(s) for s in args.shards.split(",")]
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    # ---- assemble ---------------------------------------------------------
    parts = []
    for s in shards:
        h = pq.read_table(os.path.join(
            DATA_ROOT, "harvest", "moe", f"harv_shard{s:03d}.parquet")
        ).to_pandas()
        rec = pq.read_table(os.path.join(
            DATA_ROOT, "recompute", "moe", f"recomp_shard{s:03d}_bf16.parquet"),
            columns=["traj_id", "pos", "logp_train"]).to_pandas()
        tokp = pq.read_table(os.path.join(
            DATA_ROOT, "gen", "moe", f"tokens_shard{s:03d}.parquet"),
            columns=["traj_id", "pos", "logp_infer"]).to_pandas()
        parts.append(h.merge(rec, on=["traj_id", "pos"])
                     .merge(tokp, on=["traj_id", "pos"]))
    df = pd.concat(parts, ignore_index=True).sort_values(
        ["traj_id", "pos"], ignore_index=True)
    Adf, acc = compute_A(shards)
    df = df.merge(Adf, on="traj_id", how="left")
    df["p"] = np.exp(df["logp_train"].astype(np.float64))
    df["k"] = np.exp(df["logp_train"].astype(np.float64)
                     - df["logp_infer"].astype(np.float64))
    print(f"[assemble] {len(df)} tokens, {df.traj_id.nunique()} trajs")

    report = {"n_tokens": int(len(df)), "n_traj": int(df.traj_id.nunique()),
              "reward_acc": acc, "n_tok": N_TOK}

    # ---- C1/C2 checks -----------------------------------------------------
    # float32 logprobs contain exact zeros -> p == 1.0 -> log(1-p) = -inf;
    # clip the uncertainty floor once and use it everywhere.
    u_tok = np.clip(1.0 - df["p"].values, 1e-9, None)
    x = np.log(u_tok)
    lh = np.log(df["h_norm"].values)
    (b0, b1), _ = wls(x, lh, np.ones_like(x))
    report["C1_hnorm_slope"] = float(b1)
    if abs(b1) > 0.02:
        lh = lh - b1 * (x - x.mean())
        report["C1_residualized"] = True
    hn = np.exp(lh)
    mA = df["A"].abs().values > 1e-9
    la = np.log(df["A"].abs().values[mA])
    (a0, a1), _ = wls(x[mA], la, np.ones_like(la))
    report["C2_absA_slope"] = float(a1)
    A = df["A"].values.copy()
    if abs(a1) > 0.02:
        A = np.where(mA, np.sign(A) * np.exp(
            np.log(np.abs(np.where(mA, A, 1.0)))
            - a1 * (x - x[mA].mean())), 0.0)
        report["C2_residualized"] = True

    # ---- proxies ----------------------------------------------------------
    p = df["p"].values
    k = df["k"].values
    gnorm = np.abs(A) * u_tok * hn
    gproj = np.stack(df["g_proj"].values).astype(np.float32)
    gvec = (A * hn)[:, None] * gproj
    del gproj

    # ---- split, buckets ---------------------------------------------------
    tid = df["traj_id"].values
    utr = df["traj_id"].unique()
    rng = np.random.RandomState(42)
    te_ids = set(rng.choice(utr, size=int(0.3 * len(utr)), replace=False))
    te = df["traj_id"].isin(te_ids).values
    tr = ~te
    PB = np.quantile(p[tr], np.linspace(0, 1, N_BUCKET + 1))
    pbin = np.clip(np.digitize(p, PB[1:-1]), 0, N_BUCKET - 1)
    p_mid = np.array([np.median(p[tr][pbin[tr] == b])
                      for b in range(N_BUCKET)])
    u_mid = np.clip(1.0 - p_mid, 1e-9, None)

    starts = {
        "const2": np.full(N_BUCKET, 2.0),
        "increasing": np.clip(2.0 - np.log(u_mid), 0, 6),
        "decreasing": 2.3 * u_mid,
        "loose4": np.full(N_BUCKET, 4.0),
        "tight05": np.full(N_BUCKET, 0.5),
    }

    results = {}
    for ver, gv in (("norm", None), ("direction", gvec)):
        finals, mse_curves = {}, {}
        for sname, s0 in starts.items():
            gtr = gnorm[tr]
            gvtr = None if gv is None else gv[tr]
            logC, hist = iterate(k[tr], pbin[tr], gtr, N_TOK, s0, gvtr)
            finals[sname] = logC
            mse_curves[sname] = heldout_mse(
                k[te], pbin[te], gnorm[te], logC, N_TOK,
                None if gv is None else gv[te])
        F = np.stack(list(finals.values()))
        spread = float(np.max(F.max(0) - F.min(0)))
        best = min(mse_curves, key=mse_curves.get)
        logC_star_tr = finals[best]
        # held-out one-step C*
        logC_star, ess_te, n_te = one_step_Cstar(
            k[te], pbin[te], gnorm[te], logC_star_tr, N_TOK,
            None if gv is None else gv[te])
        logC_star = interp_gated(logC_star, ess_te)
        results[ver] = dict(finals={k2: v.tolist() for k2, v in finals.items()},
                            spread=spread, best_start=best,
                            heldout_mse=mse_curves,
                            logC_star=logC_star.tolist())

        # shape extraction with trajectory bootstrap
        w = n_te.astype(float)
        x1 = np.log(u_mid)
        x2 = u_mid
        (i1, s1), r1 = wls(x1, logC_star, w)
        (i2, s2), r2 = wls(x2, logC_star, w)
        mean_c = np.average(logC_star, weights=w)
        r0 = float(np.sum(w * (logC_star - mean_c) ** 2))
        results[ver]["fits"] = dict(
            beta_log=float(s1), slope_linear=float(s2),
            bic_power=bic(r1, N_BUCKET, 2), bic_linear=bic(r2, N_BUCKET, 2),
            bic_const=bic(r0, N_BUCKET, 1))
        # bootstrap beta over held-out trajectories
        te_tid = tid[te]
        st = np.r_[0, np.nonzero(te_tid[1:] != te_tid[:-1])[0] + 1,
                   len(te_tid)]
        kte, pbte, gte = k[te], pbin[te], gnorm[te]
        gvte = None if gv is None else gv[te]
        betas = []
        n_traj_te = len(st) - 1
        for _ in range(args.boot):
            pick = rng.choice(n_traj_te, size=n_traj_te, replace=True)
            idx = np.concatenate([np.arange(st[i], st[i + 1]) for i in pick])
            lcs, ess_b, n_b = one_step_Cstar(
                kte[idx], pbte[idx], gte[idx], logC_star_tr, N_TOK,
                None if gvte is None else gvte[idx])
            lcs = interp_gated(lcs, ess_b)
            (_, sb), _ = wls(x1, lcs, n_b.astype(float))
            betas.append(sb)
        results[ver]["beta_ci"] = [float(np.percentile(betas, 2.5)),
                                   float(np.percentile(betas, 97.5))]

    # ---- n sensitivity (norm version, best start) -------------------------
    sens = {}
    for n_alt in N_SENS:
        logC, _ = iterate(k[tr], pbin[tr], gnorm[tr], n_alt,
                          starts["const2"])
        lcs, ess_a, n_a = one_step_Cstar(k[te], pbin[te], gnorm[te],
                                         logC, n_alt)
        lcs = interp_gated(lcs, ess_a)
        (_, sb), _ = wls(np.log(u_mid), lcs, n_a.astype(float))
        sens[f"{n_alt:.0e}"] = dict(beta=float(sb),
                                    mean_logC=float(lcs.mean()))
    report["n_sensitivity"] = sens

    report["results"] = results
    b_norm = results["norm"]["fits"]["beta_log"]
    b_dir = results["direction"]["fits"]["beta_log"]
    agree = abs(b_norm - b_dir) < 0.3
    beta = b_dir  # direction version is the arbiter per the guide
    ci = results["direction"]["beta_ci"]
    if beta > 0.15 and ci[0] > 0:
        verdict = ("DECREASING BAND: token-level MSE revives the phi-band; "
                   "compare (1-p)^1 vs GPD exponent")
    elif beta < -0.5:
        verdict = ("INCREASING CAP: token-MSE prediction opposite to RL "
                   "-> token-level MSE also refuted by RL (negative closure)")
    else:
        verdict = ("CONSTANT-ish: p-dependence absent at token level too; "
                   "matches sequence-level fixed point")
    report["verdict"] = verdict + ("" if agree else
                                   " | NORM/DIRECTION DISAGREE (report rho)")
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---- figures ----------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for ver, c in (("norm", "C0"), ("direction", "C1")):
        lcs = np.array(results[ver]["logC_star"])
        ax[0].plot(u_mid, lcs, "o-", ms=4, c=c,
                   label=f"{ver}: beta={results[ver]['fits']['beta_log']:.2f} "
                         f"{results[ver]['beta_ci']}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("1-p")
    ax[0].set_ylabel("log C*(p)"); ax[0].legend(fontsize=7)
    ax[0].set_title("converged cap curves (held-out)")
    for sname in starts:
        ax[1].plot(u_mid, results["norm"]["finals"][sname], "o-",
                   ms=3, label=sname)
    ax[1].set_xscale("log"); ax[1].legend(fontsize=7)
    ax[1].set_title(f"multi-start (norm), spread={results['norm']['spread']:.3f}")
    mses = results["norm"]["heldout_mse"]
    ax[2].bar(range(len(mses)), list(mses.values()))
    ax[2].set_xticks(range(len(mses)), list(mses.keys()),
                     rotation=30, fontsize=7)
    ax[2].set_yscale("log"); ax[2].set_title("held-out MSE by start (norm)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "tokenfp_main.png"), dpi=150)

    print(json.dumps({kk: vv for kk, vv in report.items()
                      if kk != "results"}, indent=2))
    print("norm fits:", results["norm"]["fits"])
    print("dir fits:", results["direction"]["fits"])
    print(f"[tokenfp] verdict: {report['verdict']}")


if __name__ == "__main__":
    main()
