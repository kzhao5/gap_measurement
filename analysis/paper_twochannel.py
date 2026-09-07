"""Fig for Sec 4.1: scale channel (width ∝ 1-p) vs residual channel (B_t independent of 1-p),
measured on the same tokens: fresh engines -> log k^(0); stale snapshot -> B_t = log k^(29) - log k^(0)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PF="/home/kzhao2/icepop-paper/figures"; OUT="/home/kzhao2/gap_measurement/results/paper"
KAPPA, ZETA = 5e-3, 5
BLUE, ORANGE, GRAY, INK = "#1f77b4", "#ff7f0e", "#8c8c8c", "#111111"
plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9.5, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 200,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
LEG=dict(frameon=True, fancybox=False, edgecolor="#bbbbbb", framealpha=1.0, handlelength=1.6, handletextpad=0.5, columnspacing=1.0, borderpad=0.4)
d=pd.read_parquet("/tmp/paper/paired_ulkB.parquet"); u=d.u.values; lk0=d.lk0.values; B=d.B.values
def mad(v): return 1.4826*np.median(np.abs(v-np.median(v)))
edges=np.geomspace(1e-4,1,17); bi=np.clip(np.digitize(u,edges[1:-1]),0,15)
um=np.array([np.median(u[bi==b]) if (bi==b).sum() else np.nan for b in range(16)])
w0=np.array([mad(lk0[bi==b]) if (bi==b).sum()>50 else np.nan for b in range(16)])
ok=(um>=1e-3)&np.isfinite(w0); c=float(np.sum(um[ok]*w0[ok])/np.sum(um[ok]**2))
res=(np.abs(B)>0.05)&(u>=1e-4)   # displacement events: |B_t| above ten times the storage resolution
absB=np.abs(B[res]); ures=u[res]
bmed=np.array([np.median(absB[np.clip(np.digitize(ures,edges[1:-1]),0,15)==b]) if (np.clip(np.digitize(ures,edges[1:-1]),0,15)==b).sum()>30 else np.nan for b in range(16)])
bpos=np.array([(B[res][np.clip(np.digitize(ures,edges[1:-1]),0,15)==b]>0).mean() if (np.clip(np.digitize(ures,edges[1:-1]),0,15)==b).sum()>30 else np.nan for b in range(16)])
rng=np.random.RandomState(0); sub=rng.choice(len(absB),size=min(2500,len(absB)),replace=False)
blue=(np.abs(lk0)>0)&(u>=1e-4); bi_idx=np.nonzero(blue)[0]; bsub=rng.choice(bi_idx,size=min(2500,len(bi_idx)),replace=False)
fig, ax = plt.subplots(figsize=(2.75,2.85)); fig.subplots_adjust(left=0.19,right=0.97,bottom=0.16,top=0.80)
ax.axvspan(1e-4,KAPPA,color="#efefef",zorder=0); ax.text(KAPPA*0.5,3.5,r"$\kappa$",fontsize=7,ha="center",color="#666666")
ax.scatter(u[bsub],np.abs(lk0[bsub]),s=2.5,color=BLUE,alpha=0.22,linewidths=0,label=r"$|\log k_t|$")
ax.scatter(ures[sub],absB[sub],s=2.5,color=ORANGE,alpha=0.28,linewidths=0,label=r"$|B_t|$")
xs=np.geomspace(1e-4,1,100); ax.plot(xs,c*xs,"--",color=INK,lw=1.1,label=r"$c\,(1-p_t)$")
ax.plot(um[ok],w0[ok],"o",color=BLUE,ms=4,markeredgecolor="white",markeredgewidth=0.4,label="width per bin")
ax.plot(um,bmed,"s-",color="#c8102e",ms=3,lw=1.1,label=r"median $|B_t|$")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-4,1.1); ax.set_ylim(3e-5,8)
ax.set_xlabel(r"$1-p_t$"); ax.set_ylabel(r"$|\log k_t|$ scale")
h,l=ax.get_legend_handles_labels(); order=[0,3,2,1,4]
fig.legend([h[i] for i in order],[l[i] for i in order],loc="upper center",bbox_to_anchor=(0.58,1.0),ncol=3,fontsize=6.5,markerscale=1.5,**{**LEG,"handlelength":1.3,"columnspacing":0.7,"handletextpad":0.35})
fig.savefig(f"{PF}/two_channel.pdf"); fig.savefig(f"{OUT}/render_two_channel.png",dpi=200)
print("c=",round(c,3),"n_res=",int(res.sum()))
