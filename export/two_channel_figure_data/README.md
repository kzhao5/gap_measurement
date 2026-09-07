# two_channel.pdf 图的数据包(train-infer mismatch, trained Qwen1.5-MoE)

## 图的直接输入
derived/paired_ulkB.parquet — 510,915 token,3 列:
  u   = 1 - p_t(p_t = 训练引擎对被采样 token 的概率,step-86 快照)
  lk0 = log k_t^(0) = logp_train(v86) - logp_infer(v86),同版本(lag 0)的训推失配
  B   = log k_t^(29) - log k_t^(0),把推理引擎换成 step-57 快照(滞后 29 步)引入的版本位移
画图脚本 scripts/paper_twochannel.py 直接读此文件:
  蓝点 = |lk0| 子样本;蓝圆 = 16 个几何 bin(1e-4..1)内 lk0 的 MAD*1.4826;
  虚线 = c*(1-p),c 由 median(1-p)>=1e-3 的 bin 加权最小二乘拟合(斜率固定 1),本数据 c≈0.487;
  橙点 = |B|>0.05 的 token;红线 = 每 bin median|B|。横纵轴均为对数刻度。

## 原始测量
raw_paired/*.parquet — 逐 token 配对重打分(2026-08-13):
  用 step-57 快照生成 4,000 条 GSM8K rollout;每 token 分别在
  pre_v28 / pre_v57(文件名 v57 缺省即 hf 前缀)/ hf_v86 快照下 prefill 重打分,4 个分片(s0-s3)。
  lk0/B 由 scripts/contamination_verify.py 的同一管线从这些分片对齐生成。

## 摘要与参数
summaries/fig1b_params.json — c_paired_fresh=0.522(注意:与图上 0.487 的差异来自 bin 方案不同,等质量 vs 几何)
summaries/B_paired_lag29/58.csv, clean_scale_paired_fresh.csv 等 — bin 级摘要。

## 已知核查结论(2026-08-28)
- 本数据自由幂律指数 beta=1.023(bootstrap 95% CI [1.020,1.026])→ 宽度 ∝ (1-p) 成立;
- base 模型 12.3M token 大测量(c=0.155 那份,不在本包,434MB,需要另传)beta=0.936;
- 逐 bin 条件均值≈0(|mean|/SD<0.15,高置信端为负,量级=Jensen 项 -SD^2/2),
  故 "log k = c(1-p)+eps(同方差)" 的加性漂移模型被否证;c(1-p) 在宽度上,不在均值上。
