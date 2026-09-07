"""What does eps0 = 0.1 (vs 5e-3) change in log M = min(log k, 2.3*max(1-p, eps0))?

Three data sources, no new GPU work:
  campaign (fresh, no staleness): affected-token share + clean false-clip rate
  ktdump (real RL, champion arm): clip rate / contamination enrichment /
          precision / recall / clean FPR per eps0, plus a TIS reference
  paired lag-29 staleness: what fraction of contaminated tokens each eps0 caps
"""
import glob, json, os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import CODE_ROOT, DATA_ROOT
from w1prime_delta import load as load_campaign

LAM = 2.3
EPS_GRID = [5e-3, 0.02, 0.05, 0.1, 0.2]
OUT = os.path.join(CODE_ROOT, "results", "eps0")
os.makedirs(OUT, exist_ok=True)
rep = {}

def band(u, eps): return LAM * np.maximum(u, eps)

# ---------------- campaign fresh ----------------
df = load_campaign("full")
lk = df["log_k"].values
u = np.clip(1 - np.exp(df["logp_train"].astype(np.float64).values), 0, 1)
rep["campaign"] = {
    "n": int(len(lk)),
    "share_u_lt": {str(t): float((u < t).mean()) for t in (0.01, 0.05, 0.1, 0.2)},
    "clean_false_clip_rate": {str(e): float((lk > band(u, e)).mean()) for e in EPS_GRID},
    "clean_false_clip_rate_TIS2": float((lk > np.log(2.0)).mean()),
    "clean_false_clip_rate_const0.23": float((lk > 0.23).mean()),
}
del df, lk, u

# ---------------- ktdump real RL ----------------
rows = []
for f in sorted(glob.glob("/home/kzhao2/nobackup/autodelete/areal_rl/ktdump/recipe_sigmatis/*.parquet")):
    t = pq.read_table(f).to_pandas()
    t["lag"] = t["version"].max() - t["version"]
    rows.append(t)
dd = pd.concat(rows, ignore_index=True)
lkd = (dd["prox_logp"].astype(np.float64) - dd["old_logp"].astype(np.float64)).values
ud = np.clip(1 - np.exp(dd["prox_logp"].astype(np.float64).values), 0, 1)
delta = (dd["lag"].values >= 1)
eps1 = float(delta.mean())
def clipstats(A):
    return dict(clip_rate=float(A.mean()),
                precision=float(delta[A].mean()) if A.any() else float("nan"),
                recall=float(A[delta].mean()),
                enrichment=float(delta[A].mean() / eps1) if A.any() else float("nan"),
                clean_fpr=float(A[~delta].mean()),
                clipped_share_in_u_lt_01=float((ud[A] < 0.1).mean()) if A.any() else float("nan"),
                mean_removed_logmass=float(np.mean(np.maximum(lkd[A] - 0, 0))) if A.any() else 0.0)
kt = {}
for e in EPS_GRID:
    kt[str(e)] = clipstats(lkd > band(ud, e))
kt["TIS_cap2.0"] = clipstats(lkd > np.log(2.0))
kt["const_0.23"] = clipstats(lkd > 0.23)
rep["ktdump"] = dict(n=int(len(lkd)), eps_lag_ge1=eps1, table=kt)

# ---------------- paired lag-29 ----------------
parts = []
P = os.path.join(DATA_ROOT, "paired")
for s in range(4):
    base = pq.read_table(os.path.join(P, f"trajs_s{s}.parquet")).to_pylist()
    flat = {"traj_id": [], "pos": []}
    for t in base:
        Pl = len(t["prompt_token_ids"])
        for k in range(len(t["gen_token_ids"])):
            flat["traj_id"].append(t["traj_id"] + s * 1000000); flat["pos"].append(Pl + k)
    d = pd.DataFrame(flat)
    for name in ("pre_v57", "pre_v86", "hf_v86"):
        e = pq.read_table(os.path.join(P, f"{name}_s{s}.parquet")).to_pandas()
        e["traj_id"] = e["traj_id"] + s * 1000000
        d = d.merge(e.rename(columns={"logp": name}), on=["traj_id", "pos"])
    parts.append(d)
dp = pd.concat(parts, ignore_index=True)
hf = dp["hf_v86"].astype(np.float64).values
lk29 = hf - dp["pre_v57"].astype(np.float64).values      # what trainer sees at lag 29
B29 = dp["pre_v86"].astype(np.float64).values - dp["pre_v57"].astype(np.float64).values
up = np.clip(1 - np.exp(hf), 0, 1)
contam = B29 > 0.1
pr = {}
for e in EPS_GRID:
    A = lk29 > band(up, e)
    pr[str(e)] = dict(clip_rate=float(A.mean()), recall_B_gt_01=float(A[contam].mean()),
                      recall_B_gt_03=float(A[B29 > 0.3].mean()),
                      clean_fpr=float(A[B29 < 0.01].mean()))
A = lk29 > np.log(2.0)
pr["TIS_cap2.0"] = dict(clip_rate=float(A.mean()), recall_B_gt_01=float(A[contam].mean()),
                        recall_B_gt_03=float(A[B29 > 0.3].mean()), clean_fpr=float(A[B29 < 0.01].mean()))
rep["paired_lag29"] = dict(n=int(len(lk29)), share_contam_B_gt_01=float(contam.mean()), table=pr)

json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=2)
print(json.dumps(rep, indent=2))
