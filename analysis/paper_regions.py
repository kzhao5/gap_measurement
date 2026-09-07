"""Appendix: acceptance/intervention regions of four operators in the (pi_infer, pi_train)
plane, with the champion run's tokens overlaid (kept vs truncated)."""
import glob, numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/home/kzhao2/nobackup/autodelete/areal_rl/ktdump/lgrid_0"; PF="/home/kzhao2/icepop-paper/figures"; OUT="/home/kzhao2/gap_measurement/results/paper"
plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "figure.dpi": 200,
                     "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
LEG=dict(frameon=True, fancybox=False, edgecolor="#bbbbbb", framealpha=1.0, handlelength=1.3, handletextpad=0.35, columnspacing=0.8, borderpad=0.3)
EPS0, LAM = 5e-3, 2.3
q=np.linspace(1e-4,1,600); p=np.linspace(1e-4,1,600); Q,P=np.meshgrid(q,p)   # Q = pi_infer (x), P = pi_train (y)
lk=np.log(P/Q)
regions={"TIS (cap 2)": lk>np.log(2.0), "IcePop (mask [0.5,5])": (lk<np.log(0.5))|(lk>np.log(5.0)),
         "KPop": np.maximum(P*np.log(P/Q)+(1-P)*np.log(np.clip((1-P)/(1-Q),1e-12,None)), Q*np.log(Q/P)+(1-Q)*np.log(np.clip((1-Q)/(1-P),1e-12,None)))>2.0,
         "Ours ($\\lambda_+\\max(1-p_t,\\varepsilon_0)$)": lk>LAM*np.maximum(1-P,EPS0)}
dfs=[pq.read_table(f).to_pandas() for f in sorted(glob.glob(f"{R}/*.parquet"))[::6]]
d=pd.concat(dfs,ignore_index=True); pi_inf=np.exp(d.old_logp.astype(float).values); pi_trn=np.exp(d.prox_logp.astype(float).values)
trig=np.log(pi_trn/np.maximum(pi_inf,1e-12))>LAM*np.maximum(1-pi_trn,EPS0)
rng=np.random.RandomState(0); keep_i=rng.choice(np.nonzero(~trig)[0],size=min(20000,(~trig).sum()),replace=False); trig_i=np.nonzero(trig)[0]
fig, axs = plt.subplots(1,4,figsize=(5.5,1.8)); fig.subplots_adjust(left=0.1,right=0.99,bottom=0.36,top=0.97,wspace=0.32)
for ax,(name,mask) in zip(axs,regions.items()):
    ax.contourf(Q,P,mask.astype(float),levels=[0.5,1.5],colors=["#fdd9c4"],alpha=0.9)
    ax.contour(Q,P,mask.astype(float),levels=[0.5],colors=["#c8102e"],linewidths=1)
    ax.plot([0,1],[0,1],color="#999999",lw=0.6,ls="--")
    if name.startswith("Ours"):
        ax.scatter(pi_inf[keep_i],pi_trn[keep_i],s=0.6,color="#1f77b4",alpha=0.25,linewidths=0)
        ax.scatter(pi_inf[trig_i],pi_trn[trig_i],s=1.2,color="#c8102e",alpha=0.5,linewidths=0)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xlabel(r"$\pi^{\mathrm{inf}}(y_t)$")
    ax.set_xticks([0,0.5,1]); ax.set_yticks([0,0.5,1])
axs[0].set_ylabel(r"$\pi^{\mathrm{trn}}(y_t)$")
for ax in axs[1:]: ax.set_yticklabels([])
for ax,lab in zip(axs,"abcd"): ax.text(0.5,-0.42,f"({lab})",transform=ax.transAxes,ha="center",va="top",fontsize=8.5)
fig.savefig(f"{PF}/regions.pdf"); fig.savefig(f"{OUT}/render_regions.png",dpi=180)
pk=np.clip(pi_trn,1e-8,1-1e-8); qk=np.clip(pi_inf,1e-8,1-1e-8)
kf=pk*np.log(pk/qk)+(1-pk)*np.log((1-pk)/(1-qk)); kr=qk*np.log(qk/pk)+(1-qk)*np.log((1-qk)/(1-pk))
print("truncated share in sample:", trig.mean(), "n", len(d), "| KPop max-KL>2 rate on ours-run tokens:", float((np.maximum(kf,kr)>2).mean()))
