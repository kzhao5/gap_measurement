"""Fig 1b: dual-scale WITH staleness. Same tokens under fresh vs stale engine
(paired lg0-e3 v86/v57), plus champion training dump (all lags, c per phase)."""
import glob, json, os
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT="/home/kzhao2/gap_measurement"; P="/home/kzhao2/nobackup/autodelete/gap_measurement/paired"; R="/home/kzhao2/nobackup/autodelete/areal_rl/ktdump/lgrid_0"
FIG=f"{ROOT}/results/figs/paper"; OUT=f"{ROOT}/results/paper"; EPS0, KAPPA, NB = 5e-3, 5, 16
BLUE, ORANGE, GRAY = "#1f77b4", "#ff7f0e", "#9a9a9a"
plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "axes.grid": True, "grid.color": "#ececec", "axes.spines.top": False, "axes.spines.right": False})
def mad(v): return 1.4826*np.median(np.abs(v-np.median(v)))
def dual(lk, u, c, label):
    phi=np.maximum(u,EPS0); Z=lk/(c*phi); edges=np.quantile(u,np.linspace(0,1,NB+1)); bi=np.clip(np.digitize(u,edges[1:-1]),0,NB-1)
    rows=[]
    for b in range(NB):
        m=bi==b; mo=m&(np.abs(Z)>KAPPA)
        rows.append(dict(phi=float(np.median(phi[m])), n=int(m.sum()), bulk=mad(lk[m]), n_out=int(mo.sum()),
                         out=float(np.median(np.abs(lk[mo]))) if mo.sum() else np.nan, out_pos=float((lk[mo]>0).mean()) if mo.sum() else np.nan))
    d=pd.DataFrame(rows); d["src"]=label; return d
def cfit(lk,u):
    edges=np.quantile(u,np.linspace(0,1,21)); bi=np.clip(np.digitize(u,edges[1:-1]),0,19)
    sc=pd.DataFrame([(np.median(u[bi==b]),mad(lk[bi==b]),(bi==b).sum()) for b in range(20) if (bi==b).sum()>100],columns=["u","mad","n"])
    core=sc[sc.u>=1e-3]; w=core.n.values.astype(float); return float(np.sum(w*core.u*core["mad"])/np.sum(w*core.u**2))
# paired
parts=[]
for s in range(4):
    base=pq.read_table(f"{P}/trajs_s{s}.parquet").to_pylist(); flat={"traj_id":[],"pos":[]}
    for t in base:
        Pl=len(t["prompt_token_ids"])
        for k in range(len(t["gen_token_ids"])): flat["traj_id"].append(t["traj_id"]+s*1000000); flat["pos"].append(Pl+k)
    d=pd.DataFrame(flat)
    for nm in ("pre_v57","pre_v86","hf_v86"):
        e=pq.read_table(f"{P}/{nm}_s{s}.parquet").to_pandas(); e["traj_id"]=e["traj_id"]+s*1000000; d=d.merge(e.rename(columns={"logp":nm}),on=["traj_id","pos"])
    parts.append(d)
dp=pd.concat(parts,ignore_index=True); hf=dp.hf_v86.astype(float).values; u=np.clip(1-np.exp(hf),0,1)
lk0=hf-dp.pre_v86.astype(float).values; lk29=hf-dp.pre_v57.astype(float).values
c_fresh=cfit(lk0,u); D0=dual(lk0,u,c_fresh,"paired fresh (lag 0)"); D29=dual(lk29,u,c_fresh,"paired stale (lag 29)")
# champion dump, all lags, c per phase
dfs=[]
for i,f in enumerate(sorted(glob.glob(f"{R}/*.parquet"))):
    x=pq.read_table(f).to_pandas(); x["lag"]=x.version.max()-x.version; x["step"]=x.version.max(); dfs.append(x)
kd=pd.concat(dfs,ignore_index=True); kd["lk"]=kd.prox_logp.astype(float)-kd.old_logp.astype(float); kd["u"]=np.clip(1-np.exp(kd.prox_logp.astype(float)),0,1)
smax=kd.step.max(); kd["phase"]=pd.cut(kd.step,[-2,smax/3,2*smax/3,smax+1],labels=["early","mid","late"])
DK=[]
for ph in ["early","mid","late"]:
    x=kd[kd.phase==ph]; c=cfit(x[x.lag==0].lk.values,x[x.lag==0].u.values); DK.append(dual(x.lk.values,x.u.values,c,f"champion train {ph} (c={c:.2f})"))
allD=pd.concat([D0,D29]+DK,ignore_index=True); allD.to_csv(f"{OUT}/fig1b_dualscale_stale.csv",index=False)
json.dump({"c_paired_fresh":c_fresh,"highconf_out_med":{d.src.iloc[0]: float(np.nanmedian(d[d.phi<0.02].out)) for d in [D0,D29]+DK},
           "highconf_bulk":{d.src.iloc[0]: float(np.nanmedian(d[d.phi<0.02].bulk)) for d in [D0,D29]+DK}}, open(f"{OUT}/fig1b_params.json","w"),indent=2)
fig,ax=plt.subplots(1,3,figsize=(16,4.6))
for a,(title,ds) in zip(ax,[("(i) same tokens, FRESH engine (lag 0)",[D0]),("(ii) same tokens, STALE engine (lag 29)",[D29]),("(iii) champion training tokens (all lags)",DK)]):
    xs=np.geomspace(EPS0/3,1,50); a.axvspan(EPS0/3,EPS0,color="#eeeeee")
    for j,d in enumerate(ds):
        cc=c_fresh if j==0 and title[:3]!="(ii" else c_fresh
        lab=d.src.iloc[0]; alpha=1-0.25*j
        a.plot(d.phi,d.bulk,"o-",color=BLUE,ms=3,alpha=alpha,label=f"bulk MAD — {lab}")
        ok=d.n_out>=50; a.plot(d.phi[ok],d.out[ok],"s-",color=ORANGE,ms=3,alpha=alpha,label=f"core-out |Z|>{KAPPA} median — {lab}")
    a.plot(xs,c_fresh*xs,"--",color=GRAY,lw=1,label=f"c_fresh φ (c={c_fresh:.2f})"); a.axhline(0.12,ls=":",color=GRAY,lw=1)
    a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(1e-4,3); a.set_xlabel("φ"); a.set_title(title,fontsize=10); a.legend(fontsize=6.5,loc="upper left")
ax[0].set_ylabel("|log k| scale"); fig.tight_layout(); fig.savefig(f"{FIG}/fig1b_dualscale_stale.png")
print("c_fresh",c_fresh); print(allD.groupby("src").apply(lambda d: pd.Series({"out_med_phi<0.02":np.nanmedian(d[d.phi<0.02].out),"bulk_phi<0.02":np.nanmedian(d[d.phi<0.02].bulk),"out_med_phi>0.3":np.nanmedian(d[d.phi>0.3].out),"out_pos_hi":np.nanmedian(d[d.phi<0.02].out_pos)})).round(4).to_string())
