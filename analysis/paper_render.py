"""Publication renders: single-panel PDFs (no titles), boxed legends horizontal above the axes,
identical panel geometry so LaTeX can place (a)(b)(c) side by side at 0.325\linewidth each."""
import glob, json, os
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, FixedLocator, FuncFormatter
from scipy.stats import genpareto
ROOT="/home/kzhao2/gap_measurement"; PF="/home/kzhao2/icepop-paper/figures"; OUT=f"{ROOT}/results/paper"
os.makedirs(PF, exist_ok=True)
EPS0, KAPPA, LAM = 5e-3, 5, 2.3
BLUE, ORANGE, GRAY, RED, INK = "#1f77b4", "#ff7f0e", "#8c8c8c", "#c8102e", "#111111"
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5, "legend.fontsize": 7.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "figure.dpi": 200, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
PANEL=(2.05, 2.05); MARG=dict(left=0.30, right=0.97, bottom=0.21, top=0.79)
LEG=dict(frameon=True, fancybox=False, edgecolor="#bbbbbb", framealpha=1.0, handlelength=1.3, handletextpad=0.35, columnspacing=0.7, borderpad=0.3, borderaxespad=0.0)
def panel():
    fig, ax = plt.subplots(figsize=PANEL); fig.subplots_adjust(**MARG); return fig, ax
def toplegend(fig, ax, ncol, handles=None, labels=None, **kw):
    if handles is None: handles, labels = ax.get_legend_handles_labels()
    kw.setdefault("fontsize", 7)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.6, 0.995), ncol=ncol, **{**LEG, **kw})
def save(fig, name):
    fig.savefig(f"{PF}/{name}.pdf"); fig.savefig(f"{OUT}/render_{name}.png", dpi=180); plt.close(fig)
def mad(v): return 1.4826*np.median(np.abs(v-np.median(v)))
nums={}
# ================= static MoE ==================
t = pd.read_parquet(f"{ROOT}/results/tokens_moe.parquet", columns=["logp_train","D"])
D = t["D"].to_numpy(np.float64); u = np.clip(1-np.exp(t["logp_train"].to_numpy(np.float64)), 1e-7, 1); phi = np.maximum(u, EPS0); del t
p4=json.load(open(f"{OUT}/params_moe.json")); c=p4["c"]; Z=D/(c*phi); out=np.abs(Z)>KAPPA; zc=Z[u>=1e-3]
tau_star=p4["fig4"]["tau_star_equalizer"]; xi=p4["gpd_tail"]["xi"]; beta=p4["gpd_tail"]["beta"]; thr=p4["gpd_tail"]["threshold_q99"]; eps_hat=p4["fig4"]["eps_hat_kappa"]
nums.update(c_moe=c, P_Z_gt_tau_star=float((zc>tau_star).mean()), P_Z_gt_lam_over_c=float((zc>LAM/c).mean()), P_absZ_gt_kappa=float((np.abs(zc)>KAPPA).mean()))
rng=np.random.RandomState(0)
# ---- mechanism (single wide panel; legend inside, boxed) ----
edges=np.unique(np.quantile(u,np.linspace(0,1,37))); idx=np.clip(np.searchsorted(edges,u,side="right")-1,0,len(edges)-2); nb=len(edges)-1
QS=[.001,.01,.1,.5,.9,.99,.999]; ux=np.array([np.median(u[idx==b]) for b in range(nb)]); quant=np.array([np.quantile(D[idx==b],QS) for b in range(nb)])
fig, ax = plt.subplots(figsize=(5.5, 3.0)); fig.subplots_adjust(left=0.1, right=0.98, bottom=0.15, top=0.97)
ax.fill_between(ux,quant[:,0],quant[:,6],color=BLUE,alpha=.12,lw=0); ax.fill_between(ux,quant[:,1],quant[:,5],color=BLUE,alpha=.28,lw=0)
ax.fill_between(ux,quant[:,2],quant[:,4],color=BLUE,alpha=.5,lw=0,label="scale channel (80/98/99.8%)")
ax.plot(ux,quant[:,3],color=INK,lw=1,label="median")
oi=np.nonzero(out)[0]; oi=rng.choice(oi,size=min(len(oi),15000),replace=False)
ax.scatter(u[oi],D[oi],s=1.4,color=ORANGE,alpha=.35,linewidths=0,label=r"residual tokens, $|Z|>5$")
xs=np.geomspace(1e-7,1,300)
ax.axhline(np.log(2.0),color=GRAY,ls="--",lw=1,label="TIS cap"); ax.axhline(np.log(5.0),color=GRAY,ls=":",lw=1,label="IcePop band"); ax.axhline(np.log(0.5),color=GRAY,ls=":",lw=1)
ax.plot(xs,LAM*np.maximum(xs,EPS0),color=RED,lw=1.8,label="ours")
ax.set_xscale("log"); ax.set_yscale("symlog",linthresh=1e-3,linscale=0.5); ax.set_ylim(-8,8); ax.set_xlim(1e-7,1.2)
ax.set_xlabel(r"$1-p_t$"); ax.set_ylabel(r"$\log k_t$"); ax.legend(loc="lower left",ncol=2,fontsize=8,**LEG)
save(fig,"mechanism")
# ---- pivot (a) ----
EE=pd.read_csv(f"{OUT}/EE_pivot_moe.csv")
fig, ax = panel(); cols=["#08306b","#2171b5","#6baed6","#000000","#fd8d3c","#e6550d","#a63603"]
for q,col in zip(QS,cols): ax.plot(EE.phi,EE[f"q{q}"],"o-",ms=2.2,lw=1,color=col,label=f"q{str(q)[1:]}")
ax.axvspan(EPS0/3,EPS0,color="#eeeeee"); ax.set_xscale("log"); ax.set_yscale("symlog",linthresh=1); ax.set_xlabel(r"$\varphi$"); ax.set_ylabel(r"$Z$")
toplegend(fig, ax, ncol=4, fontsize=6.5, handlelength=1.0, columnspacing=0.5); save(fig,"pivot_a")
# ---- pivot (b) ----
zs=rng.choice(zc,size=min(len(zc),3_000_000),replace=False); az=np.sort(np.abs(zs)); cc=1-np.arange(1,len(az)+1)/len(az)
fig, ax = panel()
ax.loglog(az[::300],cc[::300],".",ms=2,color=BLUE,label="empirical"); zz=np.geomspace(thr,az.max(),100)
ax.loglog(zz,0.01*genpareto.sf(zz-thr,xi,loc=0,scale=beta),"-",color=ORANGE,lw=1.2,label="GPD tail")
ax.axvline(tau_star,color=INK,ls="--",lw=1,label=r"$\tau^*$"); ax.axvline(LAM/c,color=GRAY,ls=":",lw=1,label=r"$\lambda_+/c$")
ax.set_xlim(1e-2,1e3); ax.set_ylim(1e-7,1.5); ax.set_xlabel(r"$|Z|$"); ax.set_ylabel("CCDF")
toplegend(fig, ax, ncol=2, columnspacing=0.8); save(fig,"pivot_b")
# ---- pivot (c) ----
taus=np.linspace(0.5,30,300); H=np.array([np.mean(np.maximum(zs-tt,0)) for tt in taus]); Dl=eps_hat*taus-(1-eps_hat)*H
fig, ax = panel()
ax.plot(taus,H**2,color=BLUE,lw=1.2,label="over-trunc."); ax.plot(taus,Dl**2,color=ORANGE,lw=1.2,label="leakage")
ax.plot(taus,np.maximum(H**2,Dl**2),color=INK,lw=2,alpha=.4,label="envelope"); ax.axvline(tau_star,color=INK,ls="--",lw=1)
ax.set_yscale("log"); ax.set_ylim(1e-6,1); ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"bias$^2$")
toplegend(fig, ax, ncol=2); save(fig,"pivot_c")
del D,u,phi,Z,out,zc,zs
# ================= dual scale (line styles per regime; compact legend) ==================
d1=pd.read_csv(f"{OUT}/EA_dualscale_moe.csv"); d1=d1[d1.kappa==KAPPA]; d2=pd.read_csv(f"{OUT}/fig1b_dualscale_stale.csv"); cf=json.load(open(f"{OUT}/fig1b_params.json"))["c_paired_fresh"]
LS=["-","--",":"]
def dual_panel(items, cref, name, ncol):
    fig, ax = panel(); ax.axvspan(EPS0/3,EPS0,color="#eeeeee")
    for j,(lab,d) in enumerate(items):
        ok=d.n_out>=50; bulk=d.bulk_mad if "bulk_mad" in d else d.bulk; outm=d.out_med if "out_med" in d else d.out
        ax.plot(d.phi,bulk,LS[j],marker="o",ms=2.2,lw=1,color=BLUE,label="bulk" if j==0 else None)
        ax.plot(d.phi[ok],outm[ok],LS[j],marker="s",ms=2.2,lw=1,color=ORANGE,label="residual" if j==0 else None)
    for j,(lab,d) in enumerate(items):
        if lab: ax.plot([],[],LS[j],color="#555555",lw=1,label=lab)
    xs=np.geomspace(EPS0/3,1,50); ax.plot(xs,cref*xs,"-.",color=GRAY,lw=0.9,label=r"$c\varphi$"); ax.axhline(0.12,ls=(0,(1,1)),color="#bbbbbb",lw=0.9)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-4,4); ax.set_xlabel(r"$\varphi$"); ax.set_ylabel(r"$|\log k_t|$")
    toplegend(fig, ax, ncol=ncol); save(fig,name)
dual_panel([("",d1)], c, "dualscale_a", 3)
dual_panel([("fresh",d2[d2.src=="paired fresh (lag 0)"]),("stale",d2[d2.src=="paired stale (lag 29)"])], cf, "dualscale_b", 3)
dual_panel([(s.split(" (")[0].replace("champion train ",""),d2[d2.src==s]) for s in ["champion train early (c=0.61)","champion train mid (c=1.28)","champion train late (c=1.67)"]], cf, "dualscale_c", 3)
# ================= trigger dynamics ==================
R="/home/kzhao2/nobackup/autodelete/areal_rl/ktdump"
def load(name):
    dfs=[]
    for f in sorted(glob.glob(f"{R}/{name}/*.parquet")):
        x=pq.read_table(f).to_pandas(); x["lag"]=x.version.max()-x.version; x["step"]=x.version.max(); dfs.append(x)
    d=pd.concat(dfs,ignore_index=True); d["lk"]=d.prox_logp.astype(float)-d.old_logp.astype(float); d["u"]=np.clip(1-np.exp(d.prox_logp.astype(float)),0,1); d["phi"]=np.maximum(d.u,EPS0); return d
d=load("lgrid_0"); d["trig"]=d.lk>LAM*d.phi; smax=d.step.max(); d["phase"]=pd.cut(d.step,[-2,smax/3,2*smax/3,smax+1],labels=["early","mid","late"])
cph=json.load(open(f"{OUT}/fig5_params.json"))["c_train_by_phase"]
per=d.groupby("step").agg(rate=("trig","mean"),stale=("lag",lambda s:(s>=1).mean())).reset_index()
ice=load("recipe_icepop"); ice["masked"]=(ice.lk<np.log(0.5))|(ice.lk>np.log(5.0)); cl=d[d.trig]
PH={"early":"#9ecae1","mid":"#4292c6","late":"#08306b"}
fig, ax = panel()
ax.plot(per.step,per.rate,"o-",ms=1.8,lw=1,color=BLUE,label="truncated"); ax.plot(per.step,per.stale,"s--",ms=1.5,lw=0.9,color=GRAY,label="stale")
ax.set_yscale("log"); ax.set_ylim(2e-3,1e-1); ax.yaxis.set_major_locator(FixedLocator([3e-3,1e-2,3e-2,1e-1])); ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{100*v:g}%")); ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("training step"); ax.set_ylabel("share of tokens"); toplegend(fig, ax, ncol=2); save(fig,"trigger_a")
fig, ax = panel(); bins=np.linspace(0,1,26)
for ph,col in PH.items(): ax.hist(1-cl[cl.phase==ph].u,bins=bins,density=True,histtype="step",lw=1.2,color=col,label=f"ours, {ph}")
ax.hist(1-ice[ice.masked].u,bins=bins,density=True,histtype="step",lw=1.2,color=ORANGE,ls="--",label="IcePop")
ax.set_xlabel(r"$p_t$ of affected token"); ax.set_ylabel("density"); toplegend(fig, ax, ncol=2); save(fig,"trigger_b")
fig, ax = panel()
for ph,col in PH.items():
    x=cl[cl.phase==ph]; ax.scatter(cph[ph]*x.phi.values[::25],np.abs(x.lk.values[::25]),s=2,alpha=.3,color=col,linewidths=0)
xs=np.geomspace(1e-4,3,50); ax.plot(xs,xs,"-.",color=GRAY,lw=0.9); ax.plot(xs,5*xs,":",color=GRAY,lw=0.9)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-3,3); ax.set_ylim(5e-3,10); ax.set_xlabel(r"$c\,\varphi_t$"); ax.set_ylabel(r"$|\log k_t|$")
H_=[Line2D([],[],color=PH["early"],marker="o",ls="",ms=4), Line2D([],[],color=PH["mid"],marker="o",ls="",ms=4), Line2D([],[],color=PH["late"],marker="o",ls="",ms=4), Line2D([],[],color=GRAY,ls="-.",lw=1), Line2D([],[],color=GRAY,ls=":",lw=1)]
toplegend(fig, ax, ncol=3, handles=H_, labels=["early","mid","late",r"$c\varphi$",r"$5c\varphi$"]); save(fig,"trigger_c")
json.dump(nums,open(f"{OUT}/paper_numbers2.json","w"),indent=2); print(json.dumps(nums,indent=1)); print("figs:", sorted(os.listdir(PF)))
