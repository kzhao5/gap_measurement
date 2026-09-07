"""Training-dynamics multi-panel (the signature figure of the TIS/IcePop/KPop line):
per-step task reward, max |log k| gap, entropy, grad norm, and intervention rate for the six
main arms (seed 1), parsed from AReaL trial logs; plus held-out-vs-step from epoch checkpoints
if evaluated. Also computes per-method affected-token rates from kt-dumps and step wall time."""
import re, glob, json, os
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R="/home/kzhao2/nobackup/autodelete/areal_rl"; ROOT="/home/kzhao2/gap_measurement"; PF="/home/kzhao2/icepop-paper/figures"; OUT=f"{ROOT}/results/paper"
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5, "legend.fontsize": 7.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "figure.dpi": 200, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
PANEL=(2.05, 2.05); MARG=dict(left=0.30, right=0.97, bottom=0.21, top=0.79)
LEG=dict(frameon=True, fancybox=False, edgecolor="#bbbbbb", framealpha=1.0, handlelength=1.3, handletextpad=0.35, columnspacing=0.7, borderpad=0.3, borderaxespad=0.0)
ARMS={"No correction":("kt-recipe/nocorr-moe-e3","#7f7f7f","-"), "Exact ratio":("kt-recipe/fullis-moe-e3","#9467bd","-"), "TIS":("kt-recipe/tis-moe-e3","#2ca02c","-"),
      "IcePop":("kt-recipe/icepop-moe-e3","#ff7f0e","-"), "KPop":("kt-recipe/kpop-moe-e3","#8c564b","-"), "Ours":("kt-lgrid/lg0-e3","#c8102e","-")}
KEYS={"reward":"ppo_actor/task_reward/avg","gapmax":"ppo_actor/update/logp_abs_diff/max","gapavg":"ppo_actor/update/logp_abs_diff/avg",
      "entropy":"ppo_actor/update/entropy/avg","grad":"ppo_actor/update/grad_norm","rs":"ppo_actor/update/rs_filtered_fraction",
      "stale":"ppo_actor/update/version_stats/sample_staleness_theta_avg","seqlen":"ppo_actor/seq_len/avg"}
PAIR=re.compile(r"│ (ppo_actor/[A-Za-z0-9_/]+)\s+│\s+(-?[0-9.]+e[+-]?[0-9]+|-?[0-9.]+|nan|inf)\s*(?=│)")
def parse(path):
    steps=[]; cur={}
    for line in open(path, errors="ignore"):
        for k,v in PAIR.findall(line):
            if k=="ppo_actor/task_reward/avg" and "ppo_actor/task_reward/avg" in cur:
                steps.append(cur); cur={}
            try: cur[k]=float(v)
            except: pass
    if cur: steps.append(cur)
    df=pd.DataFrame(steps); df["step"]=np.arange(1,len(df)+1); return df
series={}
for name,(d,col,ls) in ARMS.items():
    df=parse(f"{R}/experiments/logs/kzhao2/{d}/main.log"); series[name]=df
    print(name, len(df), "steps; keys ok:", all(k in df.columns for k in KEYS.values()))
def smooth(y,w=5): return pd.Series(y).rolling(w,center=True,min_periods=1).mean().values
def panel():
    fig, ax = plt.subplots(figsize=PANEL); fig.subplots_adjust(**MARG); return fig, ax
def save(fig,name): fig.savefig(f"{PF}/{name}.pdf"); fig.savefig(f"{OUT}/render_{name}.png",dpi=180); plt.close(fig)
def draw(key, name, ylabel, log=False, ylim=None, w=5):
    fig, ax = panel()
    from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter
    for arm,(d,col,ls) in ARMS.items():
        df=series[arm]
        if KEYS[key] not in df: continue
        ax.plot(df.step, smooth(df[KEYS[key]].values,w), color=col, lw=1.3 if arm=="Ours" else 1.0, ls=ls, alpha=1.0 if arm=="Ours" else 0.9)
    if log: ax.set_yscale("log")
    if ylim: ax.set_ylim(*ylim)
    if key=="gapmax": ax.yaxis.set_major_locator(FixedLocator([5,10,20,40])); ax.yaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    if key=="grad": ax.yaxis.set_major_locator(FixedLocator([1,3,10,30])); ax.yaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("training step"); ax.set_ylabel(ylabel); save(fig,name)
draw("reward","dyn_reward","training reward")
draw("gapmax","dyn_gapmax",r"max $|\log k_t|$", log=True, ylim=(4,60))
draw("entropy","dyn_entropy","entropy")
draw("grad","dyn_grad","gradient norm", log=True)
draw("rs","dyn_rs","intervened share", log=True, ylim=(1e-4,1))
# held-out vs step (base + epoch checkpoints where evaluated)
res={}
for f in glob.glob(f"{R}/logs/evaltp_*.out"):
    for line in open(f, errors="ignore"):
        m=re.match(r"RESULT (\S+) acc=([0-9.]+)", line)
        if m: res[m.group(1)]=float(m.group(2))*100
tagmap={"No correction":"nocorr","Exact ratio":"fullis","TIS":"tis","IcePop":"icepop","KPop":"kpop","Ours":"ours"}
final={"No correction":56.10,"Exact ratio":64.59,"TIS":62.40,"IcePop":64.82,"KPop":59.44,"Ours":64.67}
curve={}
for arm,tag in tagmap.items():
    pts=[(0,52.31)]
    for st,E in ((28,"epoch0epochstep28globalstep28"),(57,"epoch1epochstep28globalstep57")):
        k=f"curve_{tag}_{E}"
        if k in res: pts.append((st,res[k]))
    pts.append((86,final[arm])); curve[arm]=pts
fig, ax = panel()
for arm,(d,col,ls) in ARMS.items():
    xs,ys=zip(*curve[arm]); ax.plot(xs,ys,"o-",ms=3,lw=1.3 if arm=="Ours" else 1.0,color=col)
ax.set_xlabel("training step"); ax.set_ylabel("held-out accuracy (%)"); ax.set_xticks([0,28,57,86]); save(fig,"dyn_heldout")
# standalone legend strip (one row) for the composite figure
fig=plt.figure(figsize=(5.5,0.32)); H=[Line2D([],[],color=c,lw=1.6 if n=="Ours" else 1.2) for n,(d,c,ls) in ARMS.items()]
fig.legend(H,list(ARMS.keys()),loc="center",ncol=6,fontsize=8,**LEG); fig.savefig(f"{PF}/dyn_legend.pdf"); fig.savefig(f"{OUT}/render_dyn_legend.png",dpi=180); plt.close(fig)
# ---- affected-token rates per method (own dumps) and step wall time ----
def load(name):
    dfs=[]
    for f in sorted(glob.glob(f"{R}/ktdump/{name}/*.parquet")): dfs.append(pq.read_table(f).to_pandas())
    d=pd.concat(dfs,ignore_index=True); d["lk"]=d.prox_logp.astype(float)-d.old_logp.astype(float); d["p"]=np.exp(d.prox_logp.astype(float)); return d
rates={}
try:
    d=load("recipe_tis"); rates["TIS"]=float((d.lk>np.log(2.0)).mean())
    d=load("recipe_icepop"); rates["IcePop"]=float(((d.lk<np.log(0.5))|(d.lk>np.log(5.0))).mean())
    d=load("recipe_kpop"); p=np.clip(d.p.values,1e-12,1-1e-12); q=np.clip(np.exp(d.old_logp.astype(float).values),1e-12,1-1e-12)
    bkl=p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q)); rates["KPop"]=float((bkl>2.0).mean())
    d=load("lgrid_0"); rates["Ours"]=float((d.lk>2.3*np.maximum(1-d.p.values,5e-3)).mean())
except Exception as e: print("rates err", e)
# step wall time from timestamps in main.log (median seconds between consecutive reward tables)
TS=re.compile(r"(\d{8}-\d{2}:\d{2}:\d{2}\.\d+)")
walls={}
for arm,(d,col,ls) in ARMS.items():
    ts=[]
    for line in open(f"{R}/experiments/logs/kzhao2/{d}/main.log", errors="ignore"):
        if "ppo_actor/task_reward/avg" in line or "Memory-Usage ppo update" in line:
            m=TS.search(line)
            if m: ts.append(pd.to_datetime(m.group(1), format="%Y%m%d-%H:%M:%S.%f"))
    if len(ts)>3: walls[arm]=float(np.median(np.diff(pd.Series(ts).drop_duplicates().values).astype("timedelta64[s]").astype(float)))
summ={arm:{"final_reward":float(series[arm][KEYS["reward"]].iloc[-1]),"ep3_reward":float(series[arm][KEYS["reward"]].iloc[-29:].mean()),
           "gapmax_last10":float(series[arm][KEYS["gapmax"]].iloc[-10:].mean()),"entropy_last10":float(series[arm][KEYS["entropy"]].iloc[-10:].mean()),
           "grad_last10":float(series[arm][KEYS["grad"]].iloc[-10:].mean()),"rs_last10":float(series[arm][KEYS["rs"]].iloc[-10:].mean()) if KEYS["rs"] in series[arm] else None} for arm in ARMS}
json.dump({"summary":summ,"affected_rates":rates,"step_wall_s":walls,"heldout_curve":curve,"eval_results_found":{k:v for k,v in res.items() if k.startswith("curve_")}}, open(f"{OUT}/dynamics.json","w"), indent=2)
print(json.dumps({"affected_rates":rates,"step_wall_s":walls,"heldout_curve":curve},indent=1)); print({a:{k:round(v,3) for k,v in s.items() if v is not None} for a,s in summ.items()})
