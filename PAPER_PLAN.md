# 论文图表工作计划(Z+B 建模 + 实验 B 触发动态)
2026-08-15 · 常数:ε₀=5e-3,κ=5(核外判据),c=0.156 (MoE 静态) · τ*≈14.7

## 0. 开工前必须解决的一个事实问题(核对时发现)

**(a) champion 臂的 kt-dump 是 `ktdump/lgrid_0`(kt-lgrid/lg0-e3,(∞,2.3),ε₀=5e-3,
600 文件,每 10 次调用一存)。** 此前几处 ktdump 分析用的 `recipe_sigmatis`
实为 v1 中心臂(λ₋=1.0, 地板 3e-7,@79 步崩溃)——污染实验 H(截断富集)、
ε₀ 分析里的"真实 RL"表、以及"训练时 c=0.60"都需在 lgrid_0 上**重做**
(结论方向大概率不变,数字会变)。

**(b) 训练时失配尺度随训练增长且形状变超线性(lag=0 逐文件定义,champion 臂)**:

| 阶段 | c(1−p 线性拟合) | MAD@u≈0.4 | MAD@u≈0.9 | MAD@u≈0.04 | u<1e-3 |
|---|---|---|---|---|---|
| 静态(base, vLLM/HF) | 0.156 | ~0.06 | ~0.14 | ~0.006 | ~0 |
| 训练早期(step<29) | 0.54 | 0.24 | 0.53 | 0.025 | 3e-4 |
| 中期 | 1.19 | 0.34 | 1.17 | 0.015 | 1e-4 |
| 晚期 | 1.65 | 0.39 | 1.70 | 0.007 | ~0 |

高置信端(u<0.05)始终微小、(1−p) 缩放定性成立;**低置信端(u>0.4)的
失配随训练暴涨到 log-ratio MAD≈1.7**(两引擎概率典型差 5×)。含义:
- 论文里 c=0.156/τ*=14.7 是**base 模型静态标定**,Fig 1-4 用 campaign 数据自洽;
- Fig 5(c)"|log k| vs cφ_t"必须用**该训练阶段实测的 c**(0.54/1.19/1.65),
  不能用 0.156,否则"两尺度分离"被人为夸大;
- λ₊=2.3 在训练晚期对低置信 token 是 ~1.2σ、对高置信是 >10σ——
  "带截的是 epistemic 残差"在高置信端成立、在低置信端需重新验证
  (核外判据 |Z_train|>κ 用训练时尺度算);
- 机制待查(可选实验):增长来自策略锐化(bf16 相对误差放大)还是
  异步 in-flight 权重更新(序列尾部实为陈旧但 version 标签为 0)——
  给 kt-dump hook 加 `pos` 字段一行即可诊断。
- **需要你拍板论文口径**:把"训练中失配增长"作为一个正面发现写进去
  (强化"必须修正"),还是只在附录说明。

## 1. 数据保存清单(§1)——现状与待办

| 字段 | 现状 | 待办 |
|---|---|---|
| traj_id,pos,T | ✓ campaign parquet | — |
| logp_inf_raw | ✓ vLLM 采样步原始 logprob(T=1,无重缩放),float32 | 注明 |
| logp_train | ✓ HF bf16 teacher-forcing,同 checkpoint | **算 md5 入库** |
| route_flags | ✓ route_infer/route_train + margin(MoE 逐层) | — |
| gen/score_version | campaign 单版本;RL dump 有 version;配对有 v28/57/86 | — |
| engine_meta | vLLM 0.26.0 bf16 / sglang 0.5.10 / FSDP bf16 | 写 MANIFEST |
| log_m 另存 | ✓ 从未覆盖 log_k | — |
→ 产出:`data/MANIFEST.md`(路径、行数、dtype、md5、引擎版本)。0.5h。

## 2. 五项验证实验(campaign 新鲜数据为主)

| # | 做什么 | 已有 | 新增 | 产出 |
|---|---|---|---|---|
| E-A 双标度 | 按 φ 等质量分桶:主体 σ_MAD vs 核外(\|Z\|>5)\|log k\| 中位 | 主体 MAD 表 ✓ | 核外分桶 + 500 轨迹 bootstrap;κ∈{4,5,6} | Fig 1 |
| E-B 稀疏性 | ε̂=P(\|log k\|>κcφ) 全体/分桶;轨迹内成簇(lag 1–20 自相关/run test) | 部分 | 分桶 ε̂ + 成簇检验 | Fig 3 右 + 表 |
| E-C 符号包络 | 高置信桶核外 P(log k>0)、q95/q99 | ε−/ε+ 有 | 分桶表 | 表 |
| E-D 机制归因 | 核外 vs 硬翻转(margin>1e-3)对齐率;lag 0 vs 29(配对 B) | 数据全有 | 对齐率表 | 表/Fig 1 注 |
| E-E 枢轴塌缩 | 各 φ 桶 Z 分位曲线 | Z_pivotality csv(10 桶) | 重做 16–20 桶 + bootstrap | Fig 3 左 |
通用:等质量分桶、核外 n<50 打灰、bootstrap 500、κ 敏感性。约 4h。

## 3. 五张图

| 图 | 内容 | 依赖 | 脚本 | 估时 |
|---|---|---|---|---|
| Fig 1 双标度 | log-log,蓝主体贴 cφ 斜线,橙核外平坦≈0.1,ε₀ 阴影,角注 1.5σ/77σ | E-A | `analysis/paper_fig1_dualscale.py` | 1.5h |
| Fig 2 机制图 | V9 喇叭 + 核外橙散点 + TIS/IcePop 灰常数阈值 + 红 λ₊max(1−p,ε₀) 带 | 现有 V9 脚本 | 改造 V9 | 1h |
| Fig 3 标准化 | 左 Z 分位塌缩;右 \|Z\| CCDF log-log + GPD 尾(kt 选型 M5 参数)+ τ*=14.7 + ε̂ 质量 | E-B/E-E | `paper_fig3_pivot_ccdf.py` | 1.5h |
| Fig 4 Equalizer | x=τ:H(τ)²=[E(Z−τ)₊]² 降 vs D(τ)²=[ετ−(1−ε)H]² 升,max 包络,交点 τ*;**用实测 F_Z、ε̂** | E-B 的 ε̂ | `paper_fig4_equalizer.py` | 1h |
| Fig 5 触发动态 | (a) 触发率 vs step(lgrid_0,version 作 step 轴);(b) 被截 token p 直方图×3 阶段 + IcePop 被截(recipe_icepop 施 [0.5,5])对照;(c) 被截 \|log k\| vs 该阶段 cφ_t(训练时 c) | lgrid_0 + recipe_icepop dumps | `paper_fig5_trigger.py` | 2h |
判读预登记写进各 caption。绘图规范:蓝/橙,log 轴零值 clip 注明,点大小编样本量,脚本+分桶参数入 repo。

## 4. Fig 4 / Fig 3 的一个待确认点

τ*≈14.7 来自 c·z 标定;Fig 4 的交点 τ* 由实测 F_Z 与 ε̂(κ=5 定义)解出——
两者需一致或说明差异(此前用 ε=lag≥1 的 0.48% 解出 τ_theory=7.9)。
建议以 Fig 4 交点为"理论 τ*",与标定值 14.7、RL 最优 λ₊/c 三值并列报告。

## 5. 顺序与总工时

MANIFEST(0.5h)→ E-A~E-E 统一脚本(4h,CPU Slurm)→ Fig 1/3/4(4h)→
Fig 2 改造(1h)→ Fig 5 + lgrid_0 上重做 H/ε₀ 表(3h)→ 图-数-脚本三方核对。
合计约 1.5 个工作日;全部离线,无 GPU。
