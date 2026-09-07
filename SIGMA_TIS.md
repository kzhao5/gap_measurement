# σ-TIS 算法完整规格:版本、修改、参数(2026-08-12 定稿)

> 实现:`AReaL/areal/utils/functional/functional.py` token 级 clamp 分支(patch 03,
> 环境变量门控);训练配方与各臂结果见文末。伴随文档:ANALYSIS.md(推导与证据)。

---

## 1. 算子定义

**v1(论文原始形式,"各向异性截断")**

$$\log M^{\star}(K_t, p_t) = \operatorname{clip}\!\big(\log K_t,\; -\lambda_-(1-p_t),\; +\lambda_+(1-p_t)\big)$$

- K_t = π_θold(y_t|h_t) / π_infer(y_t|h_t):训练策略与行为策略的逐 token 重要性比;
- p_t = π_θold(y_t|h_t):训练策略对被采样 token 的概率;
- 带宽 ∝ (1−p_t):跟随实测噪声包络 σ_t = c₁(1−p_t)(V9,六个数量级);
- p→1 时带宽→0(算子→恒等)。**← 此性质被实战证伪,见 v2。**

**v2(修正形式,当前推荐)**

$$\boxed{\ \log M^{\star} = \operatorname{clip}\!\big(\log K_t,\; -\lambda_-\,\phi_t,\; +\lambda_+\,\phi_t\big),\qquad \phi_t = \max\big(1-p_t,\ \varepsilon_0\big)\ }$$

**推荐默认(canonical σ-TIS v2)**:λ₋ = ∞(单侧)、λ₊ = 2.3、ε₀ = 5×10⁻³,即

$$\log M^{\star} = \min\!\big(\log K_t,\; +2.3\cdot\max(1-p_t,\ 5\times10^{-3})\big)$$

## 2. v1 → v2 改了什么,为什么(全部有实验证据)

| 修改 | v1 | v2 | 证据 |
|---|---|---|---|
| **带宽地板** | 无(p→1 时带宽→0) | φ_t = max(1−p_t, ε₀), ε₀=5e-3 | v1 第 3 epoch 右触发达 **32%**,被裁 token 中位 \|log K\|=**0.0007**——全是 bf16 量化噪声;裁噪声→梯度偏置→失稳(0.87→0.45)→sglang KV 崩溃@79 步。加地板后右触发回落至 1.5–7%,被裁中位 \|log K\|=0.08–0.12(真实陈旧偏差)✓ |
| **单侧化(λ₋=∞)** | λ₋=1.0(支撑锚) | **默认不裁下侧** | 左触发裁的是真陈旧 token(中位 1.2–2.4,真实权重应为 0.1–0.3),截断把权重抬回≈1=给策略已不认的行为喂全额梯度(反保守正反馈)。λ₋=1.0 两版对照:v1 崩溃、v2+地板活但垫底(0.775 vs 单侧臂 0.86–0.88) |

理论注记:"下侧是墙"的支撑定理(log K ≥ −s_train)本身没错——错的是对墙**动手**:墙外确实无物,但墙内的陈旧质量不该被抬升。单侧化让定理的正确用法从"设 λ₋=1"变成"λ₋=∞,让下侧自然落体"——恰是 TIS 原始单侧哲学 min(ρ,C) 的证据化版本。

## 3. 各臂精确参数(环境变量映射)

实现公式:`band = [−A·max(C1·(1−p_t), FLOOR), +B·max(C1·(1−p_t), FLOOR)]`,
`w_t = exp(clip(log k_t, band))`。λ 参数化:`C1=1.0, A=λ₋, B=λ₊, FLOOR=ε₀`。

| 臂 | KT_SIGMA_A (λ₋) | KT_SIGMA_B (λ₊) | KT_SIGMA_FLOOR (ε₀) | 3-ep 末值/峰值/ep3均值 |
|---|---|---|---|---|
| v1 中心 | 1.0 | 2.3 | 3e-7(形同虚设) | 崩溃@79 / 0.869 / 0.639 |
| **v2 (∞,2.3)** | 1e9 | 2.3 | **5e-3** | **0.881 / 0.897 / 0.867(全场最佳)** |
| v2 (∞,1.6) | 1e9 | 1.6 | 5e-3 | 0.864 / 0.881 / 0.859 |
| v2 (∞,3.1) | 1e9 | 3.1 | 5e-3 | 0.860 / 0.892 / 0.861 |
| v2 (1.0,2.3) | 1.0 | 2.3 | 5e-3 | 0.775 / 0.869 / 0.827 |

λ₊ 稳健性:{1.6, 2.3, 3.1} 三档全部 ≥ 在位方法最好水平(TIS 0.861)。

## 4. 参数的校准来源(不是拍脑袋)

- **λ₊ = 2.3 = ĉ·z\***:ĉ=0.156(标准化实验的尺度律拟合,R²=0.999),z\*≈15(谱隙/门帽闭式);备选 z\*∈{12,20} 给出 1.9/3.1;
- **ε₀ = 5e-3**:bf16 量化梳的尺度(E2/E6 审计:高置信端离散梳量子 ~1e-4–1e-3,取其上包络);
- **λ₋ = ∞**:P-λ₋ 判决(上表)+ 触发分解因果链;
- c₁ 语义:若不用 λ 参数化,`C1=c₁(在线估计), A=a, B=b` 等价于 σ̂ 形式(见 ANALYSIS.md §6.1)。**换模型/引擎/精度时 c₁ 与 ε₀ 需重校准**(一次分桶回归,零额外开销)。

### 4.1 勘误与精度重标(2026-08-15,训练时 kt-dump 实测)

用 champion 臂 kt-dump 的 **lag=0 token(纯数值失配,无陈旧)** 重测尺度律:
**训练时 c = 0.60(bf16 sglang vs FSDP),不是静态标定的 0.156**——RL 使
策略锐化、失配随之放大(与污染实验 v86 策略 c=0.52 一致,mismatch grows
with training)。因此 λ₊=2.3 在真实训练中操作在 **≈3.8σ 而非 14.7σ**:
它裁掉清洁 token 的上尾 ≈1.8%(被裁 token 中 95% 是 lag=0,H 实验),
是"温和裁尾"而非"噪声永不触 cap"。这不改变冠军地位,但解释了:
λ₊∈{1.6,2.3,3.1} 全稳健(温和尾裁不敏感)、fp8 下 −4.6(误裁翻 3 倍)。

**p=1 时**:1−p=0 → 取地板 → 带宽 2.3×5e-3=0.0115 → k≤1.0116;此时
log k=−log p_infer≥0,下侧无事;p_infer<0.9886 的 token 被裁到 1.0116。
p 精确=1 的 token 占训练时 11%(bf16)/22%(fp8)。

**按精度重标配方(z 不变,零训练开销)**:目标精度 rollout 的 lag=0 token
分桶拟合 σ=c(1−p),λ₊,new = λ₊·c_new/c_ref(ε₀ 不动,地板带随 λ₊ 同缩)。
实测 fp8 KV: c=1.75(2.9× bf16 的 0.60),当前带清洁误裁 5.2%(bf16 1.8%),
陈旧比例 9.7%(bf16 0.48%)⟹ λ₊,fp8 ≈ 6.7。**验证(seed 1)**:

| fp8 rollout,Ours λ₊ | held-out | vs 2.3 |
|---|---|---|
| 2.3(bf16 标定,未重标) | 60.12 | — |
| 4.0(半量) | 59.82 | −0.3(噪声内) |
| **6.7(= 2.3 × c_fp8/c_bf16,原则值)** | **63.00** | **+2.9** |
| 参照:IcePop fp8 / TIS fp8 / Ours bf16 | 63.84 / 60.20 / 64.67 | |

**z 不变重标配方成立**:按 c 比值足量缩放收回 fp8 损失的 63%(−4.6→−1.7),
与 IcePop 并列头部(差 11 题,SE 内);半量(4.0)无效——需要的是完整
比值缩放,不是"稍微放松"。这把 §4 的"换精度需重校准"从注记变成实证,
也给出了跨精度部署的可操作规则:**测一次 lag=0 的 c,λ₊ 按比例缩放**。

## 5. 训练配方(所有臂共用,来自 AReaL 官方 MoE 配置)

Qwen1.5-MoE-A2.7B-Chat,GSM8K,8×A100/臂;lr **3e-6**、weight_decay 0.01、
eps_clip 0.2/higher 0.28、kl_ctl 0、reward_scaling 10/bias −0.5、n_samples 4、
temperature 1.0、max_new_tokens 1024、max_head_offpolicyness 2、
use_decoupled_loss true、FA2(sdpa 禁用!packed 前向会跨序列污染)、
mb 4096、3 epochs=87 步、seed 1。sbatch:`rl/lgrid.sbatch`;补丁:`rl/patches/03`。

## 6. 对照组结果(同配方同种子)

| 方法 | 训练末值/峰值/ep3均值 | **GSM8K test acc** |
|---|---|---|
| 朴素 GRPO(无修正,文献 baseline) | 0.746 / 0.762 / ~0.73 | 56.10% |
| full-IS | 0.831 / 0.876 / 0.843 | 64.59% |
| TIS (cap 2.0) | 0.861 / 0.883 / 0.847 | 62.40% |
| IcePop 官方 | 0.855 / 0.889 / 0.855 | **64.82%** |
| KPop 官方 | 0.859 / 0.887 / 0.852 | 59.44% |
| **σ-TIS v2 (∞,2.3)** | **0.881 / 0.897 / 0.867** | **64.67%** |
| σ-TIS v2 (∞,1.6) | 0.864 / 0.881 / 0.859 | 64.67% |
| σ-TIS v2 (∞,3.1) | 0.860 / 0.892 / 0.861 | 63.76% |
| σ-TIS v2 (1.0,2.3) 双侧 | 0.775 / 0.869 / 0.827 | **48.98%(低于 base 52.31%!)** |
| σ-TIS v1 中心(崩溃臂 2ep) | 崩溃@79 | 57.39% |
| base(未训练) | — | 52.31% |
| v3: 仿射带 +η=0.12 | 0.805 / 0.881 / 0.838 | 62.85% |
| v3: 下界 w≥0.001(\|logk\|≤6.91) | 0.851 / 0.898 / 0.851 | 63.15% |
| v3: 下界 w≥0.1(\|logk\|≤2.30) | 0.864 / **0.910** / **0.878** | **56.10%(掉回 nocorr 水平)** |

held-out 判读:acc 单臂标准误 ≈1.3%(≈17 题)——头部(IcePop/σ-TIS v2 单侧/full-IS,
853-855 题)**统计学并列**;双侧臂 held-out 崩到 base 之下 = 单侧化的最强消融证据。
限定:全部单种子。总图:`results/figs/V15_all_arms.png`。

### v3 消融判决(2026-08-13,单变量 vs 冠军 (∞,2.3))

**下界谱系(左侧动作 = lift 的剂量反应,held-out)**:
∞(不动)64.67% → w≥0.001 63.15%(−1.5,~1.2SE)→ w≥0.1 **56.10%**(−8.6,
决定性)→ λ₋=1.0(1−p) 48.98%(−15.7)。**单调恶化**:主动抬升(lift)k≈0
token 的伤害随抬升水平单调增大。"理论上要有下界"的答案:**要,但它已经
免费存在**——支撑墙 log K ≥ −s_train + 自消音天然有界,任何"clip 式下界"
都是在墙内抬升垃圾 token,只有害处。
**最危险的发现:w≥0.1 臂训练指标全场最优(ep3 均值 0.878 > 冠军 0.866,
峰值 0.910)而 held-out 掉回无修正水平**——lift 的伤害在训练 reward 上
完全不可见甚至呈正向(拟合噪声梯度抬高训练分数),只有 held-out 能暴露。
训练曲线好≠算法对,这是全 campaign 最强的"必须测 held-out"证据。

三重验证(2026-08-13,回应"确定没弄错"):① 评测日志中模型路径正确
(kt-v3race/lowb01-e3/.../globalstep86);② lowb01 与 nocorr 权重 md5 不同
(与 nocorr 同为 740/1319 是计数巧合);③ **ep1 中间 checkpoint 评测:
62.62%(826/1319)**——时间剂量曲线 base 52.31 → ep1 62.62 → final 56.10:
第 1 个 epoch RL 正常起效(接近冠军 −2.0),后两个 epoch 训练 reward
继续上涨、held-out 反跌 6.5 分。**伤害随训练步数累积**(策略越锐化,
落入 k<0.1 被 lift 的 token 越多),过拟合轨迹的教科书形态。

**仿射带 η=0.12**:62.85%(−1.8,~1.4SE)+ 训练侧全面略差(0.838 vs 0.866)。
拒绝加性加宽;与 (W1′) 测量一致——序列耦合是方差通道(Δ(φ) 斜率
−0.27~−0.88 < 0,见 results/w1prime/W1PRIME.md),低置信端理论上应更紧
而非更宽。乘性带 λ₊·max(1−p, ε₀) 维持不变。

**冠军算子确认:log M = clip(log k, −∞, +2.3·max(1−p_t, 5e-3))**,
v3 两个方向的改动均被否证,且两个否证方向互相独立(下界=左侧,仿射=右侧)。

### ε₀ 消融(2026-08-15,三种子):5e-3 vs 0.1

| ε₀ | s1 / s2 / s3 | 均值±std | 训练 ep3 |
|---|---|---|---|
| **5e-3(现行)** | 64.67 / 64.97 / 63.23 | **64.29 ± 0.93** | 0.867 |
| 0.1 | 60.73 / 60.35 / 61.94 | 61.01 ± 0.83 | 0.857–0.877(持平/略高) |

ε₀=0.1 使 p>0.9 区(58% token)带宽统一抬到 0.23,放走中等陈旧
(0.1<B<0.3;真实 RL 陈旧召回 19%→8.5%);held-out **逐种子全负、均值 −3.3**,
训练指标却持平——第三次"训练指标失明"案例。**否决;ε₀ 是量化噪声地板,
不得抬到陈旧尺度。** 详 results/eps0/EPS0.md。

### 多种子 + fp8 压测(2026-08-14/15,三种子全齐;fp8 6/6 齐)

**多种子 held-out acc 终表(3 种子)**:

| 方法 | s1 | s2 | s3 | **均值 ± std** | min |
|---|---|---|---|---|---|
| **Ours** | 64.67 | 64.97 | 63.23 | **64.29 ± 0.93** | **63.23** |
| TIS | 62.40 | 63.91 | 65.43 | 63.91 ± 1.52 | 62.40 |
| IcePop | 64.82 | 62.70 | 63.46 | 63.66 ± 1.07 | 62.70 |
| full-IS | 64.59 | 55.50 | 59.06 | 59.72 ± 4.58 | 55.50 |
| KPop | 59.44 | 59.29 | 57.85 | 58.86 ± 0.88 | 57.85 |
| nocorr | 56.10 | 54.44 | 56.10 | 55.55 ± 0.96 | 54.44 |

判读(n=3):
- **Ours 均值第一(64.29)、种子间方差最小(±0.93)、最差种子也最高
  (63.23)**——三个维度同时领先;
- 头部 {Ours, TIS, IcePop} 统计学并列(两两差 <0.7,t 检验不显著);
  TIS 多种子后上移(65.43 峰),IcePop 的 s1 64.82 显现为种子运气;
- **full-IS 决定性跌出头部**:59.72 ± 4.58,极差 9.1 分,2/3 种子 <60;
  且 s2 训练曲线完全健康(final 0.799/max 0.859)但 held-out 崩至无修正
  水平——未截断 IS 的种子间方差是"训练指标失明"的第二个案例(与 lift
  慢性毒同签名);
- 修正 vs 无修正主效应:64.29 vs 55.55 = **+8.7 分**,跨种子高度显著。

**fp8 压测(sglang KV-cache fp8_e5m2 量化 rollout,放大失配;seed 1)**:

| 方法 | bf16 acc (s1) | **fp8 acc** | Δ(fp8−bf16) | fp8 训练终值 |
|---|---|---|---|---|
| Ours(λ₊=2.3 未重标) | 64.67 | 60.12 | −4.6 | 0.822(稳) |
| **Ours(λ₊=6.7 重标)** | 64.67 | **63.00** | **−1.7** | — |
| TIS | 62.40 | **60.20** | −2.2 | 0.802 |
| full-IS | 64.59 | 57.62 | −7.0 | 0.821 |
| nocorr | 56.10 | 54.36 | −1.7 | 0.677(中途 0.76→0.64 崩落 + rollout 超时) |
| KPop | 59.44 | **50.19(低于 base 52.31!)** | **−9.3** | 0.785 |
| **IcePop** | 64.82 | **63.84** | **−1.0** | (dw 重跑) |

判读(6/6 齐):放大失配下 **IcePop 榜首(63.84,仅 −1.0)**,截断类
Ours/TIS 并列第二(60.1/60.2,差 1 题),比无修正 +5.8;full-IS(−7.0)与
KPop(−9.3,崩到 base 以下)大幅退化。两个 mask 方法命运相反:IcePop 的
ratio-mask [0.5,5] 在量化噪声下几乎免疫,KPop 的 binary-KL mask 崩溃——
说明 mask 本身不是问题,**KL 型判据对量化噪声敏感**。Ours 的 −4.6 提示:
KV-fp8 放大的是高置信区的数值噪声尺度(c 变大),而我们的 c=0.156 是在
bf16 下标定的——**换精度需重校准 c₁**(SIGMA_TIS §4 早已注明);这正是
ε₀/c 重校准价值的直接证据。单种子,待多种子确认。

### 五 benchmark 评测套件(2026-08-19,eval_suite.py,统一 \boxed + math-verify,greedy)
21/22 job 完成;kpop_s2 评测 job 异常(9 分钟跑完、全线 ~2%,旧 GSM8K 评同 ckpt 是 59.29,
判定为评测侧故障非 ckpt 损坏),已重提(13216383,附样例输出调试行)。

| 方法 | GSM8K | MATH500 | SVAMP | Minerva | Olympiad | **Avg** |
|---|---|---|---|---|---|---|
| base(no RL) | 55.50 | 21.20 | 59.00 | 4.04 | 4.90 | 28.93 |
| nocorr | 61.21±1.17 | 21.33±1.40 | 64.33±1.73 | 3.68±1.28 | 4.40±0.45 | 30.99±0.22 |
| full-IS | 62.07±5.80 | 22.00±1.22 | 65.11±7.12 | 5.51±0.64 | 4.60±0.15 | 31.86±2.23 |
| KPop | (s2 重评中) | | | | | |
| KPop-fix | 63.81±2.92 | 23.67±1.29 | 67.55±5.55 | 4.66±1.70 | 5.24±0.67 | 32.99±1.69 |
| IcePop | 66.16±0.96 | **24.33±1.03** | **69.78±1.02** | 5.51±0.64 | 5.09±1.11 | 34.18±0.55 |
| TIS | 61.64±4.51 | 20.27±4.87 | 65.78±4.00 | 4.66±1.39 | 3.91±0.60 | 31.25±2.63 |
| **Ours** | **67.88±1.64** | 24.20±1.59 | 69.45±1.95 | **5.88±0.64** | 4.94±0.99 | **34.47±0.48** |

判读:新协议(\boxed+math-verify)下排序与旧 EM 基本一致但间距拉开——Ours GSM8K
均值领先 IcePop 1.7 分(旧协议 0.6),Avg 第一(34.47±0.48,种子间 std 也最小);
IcePop 在 MATH500/SVAMP 以 <0.4 分名义领先(噪声内);TIS 在新协议下 s2 掉到 57.24,
均值滑至 61.64±4.51(方差第二大),full-IS 方差签名依旧(±5.80)。域外 hard 集
(Minerva/Olympiad)全体 4-6%,分辨力低,主要证明"不伤 OOD"。
注意:新旧协议 GSM8K 每 seed 差 ±1-7 分,论文统一用新协议,旧 EM 保留在 appendix
逐种子表;正文 RQ1 数字待 kpop_s2 重评后一次性改写。

### 新 baseline 臂预注册 + 提交(2026-08-19,job 13221854-76,4 方法 × 3 种子)
文献四层缺口补齐(见 MULTIMODEL_PLAN.md 分析),全部沿用 e3-recipe,先注册后跑:
- **seqtis**(序列级截断 IS,"When Speed Kills Stability" 线):整条轨迹乘积
  ratio w=min(∏k_t, 2),cap=2 对齐 token 级 TIS。实现:AReaL rejection_sampling
  level=sequence agg=sum action=clamp + functional.py 新 env 开关 KT_SEQ_PROD=1
  (原生序列 clamp 只作用于几何均值权重,CPU 单测证明是 no-op,故必须开
  KT_SEQ_PROD 才是文献语义;单测:prod=2.72→clamp 2.0,in-band 1.105 保留)。
- **seqmis**(序列级掩码 IS):∏k_t ∉ [0.5,2] 整条 mask,幸存者带乘积权重;
  同上 KT_SEQ_PROD=1,action=mask。
- **gspo**(Qwen,surrogate 级):importance_sampling_level=sequence(AReaL 内建
  几何均值),关闭 rejection_sampling(~actor.rejection_sampling),
  eps_clip 按 GSPO 原文缩至 3e-4/4e-4(seq 几何均值 ratio 偏离极小,0.2 等于不 clip)。
- **fp16**(系统级,"Defeating the Mismatch via FP16"):actor.dtype=float16
  (yaml 里 sglang.dtype=${actor.dtype},训推同时切换),无算子
  (use_decoupled_loss=false,同 nocorr 语义)。
理论预言(先于结果写下):GSM8K 轨迹短、∑log k 小(S0 闭包 λ*=0.07),
seqtis/seqmis 应退化到 full-IS/nocorr 一档;fp16 应压缩 c(尾数 10 位 vs 7 位)
→ 失配变小,无修正也比 bf16-nocorr 稳,但 B_t(版本陈旧)不受影响。
eval 自动链 5-benchmark suite(seeds.sbatch 已切到 eval_suite.sbatch)。

**作者代码对拍(2026-08-19,应用户要求 clone 官方 repo 验证新臂,参考仓库在
nobackup/autodelete/ref_repos/{slime,verl,Precision-RL}):**
- Seq-TIS/Seq-MIS ↔ slime `examples/train_infer_mismatch_helper/mis.py`(Miles/LMSYS 官方):
  sequence level=乘积 ∑log k、truncate=min(∏k,C)、mask 带默认 [1/C,C];**随机数据逐 token
  对拍 max diff = 0.0,mask 判定完全一致**;C=2 即作者默认 tis_upper_bound。作者 yaml 的
  可选项 rs_veto(1e-4)与 batch_normalize 属组合 recipe 开关,预注册为关闭(纯序列级算子)。
- GSPO ↔ verl `compute_policy_loss_gspo` + 官方示例 clip 3e-4/4e-4:AReaL 内建
  exp(mean log r)与 verl 的 sg-trick 是 GSPO 论文同一公式的等价实现;我们 eps_clip 已按
  官方示例设 3e-4/4e-4。
- FP16 ↔ sail-sg/Precision-RL `verl_fp16.patch`:训推同切 float16 ✓(actor.dtype 联动
  sglang.dtype);作者加 ShardedGradScaler——AReaL FSDP2 缺 loss scaling,已补 env-gated
  静态缩放 KT_FP16_LOSS_SCALE=1024(backward 前 ×S,clip 前 ÷S;AReaL 原有 non-finite
  grad_norm 跳步守卫提供 skip-on-overflow 语义,对应 scaler 的 inf-skip;无动态增长,
  已如实记录偏差)。fp16 三个 job 已重提(13224853-55),其余 9 臂配置与作者一致无需动。

### kpop_s2 定案(2026-08-19)+ 新臂首批结果
**更正**:此前判"kpop_s2 ~2% 是评测侧故障"是错的。重评(13216383)完全复现 2.05%,
调试样例显示模型在 chat template 下输出空串('\n\n\n\n');同一 checkpoint
(epoch2globalstep86)在旧协议裸补全 prompt 下 59.29%(evaltp_13170633)。
即:**KPop 未修正臂 seed2 发生 chat 格式塌缩**(权重没坏,chat 行为被训坏)。
另,先前笔记里"旧评同 ckpt 59.29"我曾误归为 suite_nocorr_s3 的 EM 行,实际
evaltp 日志确认就是 kpop-s2 本身的旧协议分数。
按预注册统一协议如实入表:KPop 行 = 40.79±33.67 / 14.53±10.34 / 42.67±34.93 /
3.68±0.37 / 3.76±1.99 / **Avg 21.08±16.17**;脚注注明塌缩与 59.3 的裸补全对照。
与 KPop†(32.99±1.69)构成"as-shipped 不稳→修正后稳"的最强对照。

**新臂进展**:fp16_s1 训练完成(dw,1h24m,顺利)+suite 已出:
60.27/22.40/61.00/5.15/4.60,Avg 30.68——与 bf16-nocorr(30.99±0.22)同档,
预注册预言前半兑现(fp16 稳定可跑,但不带回算子增益)。
seqtis/seqmis/gspo 首提因 hydra 语法失败(yaml 无 agg/importance_sampling_level
键需 + 前缀),已修 seeds.sbatch 重提 9 job(13241433-41,seqtis s1/s2、seqmis s1
转 dw87);fp16 s2/s3 在 m13h 排队。

### 新臂三种子定案(2026-08-25,cs/cs2 迁移后 5/6 跑完,全部作者代码语义)
| 臂 | GSM8K | MATH500 | SVAMP | Minerva | Olympiad | Avg | ep3 reward | 干预率 |
|---|---|---|---|---|---|---|---|---|
| Seq-TIS(2 种子,s3 重提 13502595) | 65.84±0.16 | 22.60±0.28 | 63.00±1.41 | 3.12±1.82 | 4.82±0.52 | 31.88±0.73 | 0.80 | 12–17% 序列被 cap |
| Seq-MIS | 60.98±1.50 | 22.07±0.50 | 63.56±1.17 | 4.41±0.64 | 4.75±0.26 | 31.15±0.67 | 0.76 | **95% 序列被整条 mask** |
| GSPO | 30.88±27.46 | 10.13±8.26 | 37.89±33.27 | 3.07±2.45 | 2.28±0.99 | 16.85±14.37 | 0.34/0.03/0.54 | 三种子 ep2 峰值 ~0.80 后全部塌 |
| FP16 | 60.02±1.16 | 22.67±0.46 | 62.78±1.54 | 5.03±0.21 | 4.60±0.45 | 31.02±0.39 | 0.74 | 无算子 |
判读(对照预注册预言):Seq-MIS/FP16 精确回到 nocorr 档(31.0)✓;Seq-MIS 之所以
无效是乘积 ratio 在几百 token 上几乎必然出 [0.5,2] → 丢掉 95% 数据;Seq-TIS 保住
GSM8K(65.8)但迁移没跟上(Avg 31.9),预言"退化到 full-IS 档"在 GSM8K 上偏保守;
GSPO(seq 级 surrogate、k_t 不修)三种子全塌、终点低于 base——surrogate 级挡不住
token 级通道的最强证据。seqtis s3 首提在 cs 节点 SGLang 启动被 -9 kill(基础设施),
已重提。全部已入 Overleaf Table 1(四行)+ RQ1 正文按新协议改写 + 附录 tab:extra_seeds。
**注意:集群提示 sponsor 账户 18 天后到期(~9/12),届时本账户会被禁用。**
Seq-TIS s3 补齐(2026-08-27,13502595 → suite 13505069):66.11/23.60/64.67/4.78/6.23,Avg 33.08;
三种子行 65.93±0.19 / 22.93±0.61 / 63.56±1.39 / 3.68±1.60 / 5.29±0.89 / **32.28±0.86**,ep3 奖励 0.81,cap 触发 12–17%。
Olympiad 5.29 成列最优(取代 KPop† 5.24)。Qwen1.5-MoE 块 12 行全部定稿。

### Cell D/E 全臂扩展启动(2026-09-01,用户指令:两个新模型跑全部 baseline+ours)
- Cell D = deepseek-ai/DeepSeek-V2-Lite-Chat;Cell E = Qwen/Qwen3-30B-A3B;权重已缓存(30G/57G)。
- 基础设施:rl/cells.sbatch(CELL×METHOD×SEED,新 cell 用 ++actor.optimizer_dtype=bfloat16,
  q30b 另加 mb=2048);eval_suite.py 补 enable_thinking=False(Qwen3 评测必需,其他模型无影响);
  src/common.py MODELS 登记 dsv2/q30b(is_moe=False:路由钩子系 qwen2_moe 专用,
  λ₊ 编译只需 logp;routing 分析保持 cell A 专属)。
- 已提交:冒烟 nocorr s1 ×2(13548312 dsv2 / 13548314 q30b,SMOKE=1 即 1 epoch);
  base 模型五 bench 评测 ×2(13548316/318);静态测量 ×2(gen 8 shard + bf16 rescore)。
- 预注册流程:冒烟过(median log k≈0、reward 上行)→ 提交 10 baseline 方法 × 3 种子/cell;
  静态测量出 c/ξ₊/ε̂ → equalizer 解 τ* → λ₊=cτ* → 提交 ours × 3 种子/cell。
  两个 cell 的所有臂同配方(e3-recipe),只换算子;KPop/KPop† 阈值、TIS cap、IcePop 带、
  seq 级/GSPO/FP16 参数与 cell A 完全一致。
- 风险:sponsor 账户 9/12 到期;66 训练 job 的队列吞吐;q30b OOM(备选 offload_params)。

### q30b 静态测量 + 编译(2026-09-01,20.5M token/2 万轨迹,params_q30b.json)
c=0.1845(base),ε̂(κ=5)=0.27%(比 cell A 的 1.26% 稀疏),τ*(equalizer)=5.15,
**λ₊ = cτ* = 0.95(q30b CIS 臂预注册部署值,LAMP=0.95)**;GPD 尾 ξ₊=0.30(重尾,
截断必要;轻于 cell A 的 0.49)。paper_zb.py 加了空 bin/空尾保护(30B 高置信区空桶)。
注:base 五 bench 评测 GSM8K=95.2 → cell E 训练集待用户拍板(GSM8K 饱和 vs 改 MATH)。

### Cell D(dsv2)冒烟通过 + 批量提交(2026-09-01)
冒烟(13548937,triton 后端 + None-mask 补丁):31min/epoch,奖励 0.37→0.52,
ktdump cur−old median=-0.00001/MAD=0.0035(无跨序列污染)✓。
排障记录(全部一次性成本):HF 缓存需在 nobackup/autodelete/hf(env.sh 离线约定);
deepseek_v2 + fa3(MLA)在任何卡上都崩 → ++sglang.attention_backend=triton;
AReaL dict-mask 与 transformers5.3 deepseek_v2 不兼容 → fsdp_engine None-mask 分支加 deepseek_v2;
cs-3-1(B200)fa3 断言 SM80-90 → 排除;q30b d4 单卡放不下 30B → TP2 + 0.7 显存 +
disable_custom_all_reduce(sgl_kernel graph capture invalid argument,裸跑复现)。
**已提交 dsv2 30 job(10 方法×3 种子,s1→cs/cs2、s2→dw87、s3→m13h)**;
ours 等 dsv2 测量(13548660,已迁 dw87)出 c→λ₊。q30b 冒烟(13548938)运行中,
15 步已越过 d4 时代死点;q30b 批量仍等用户对训练集拍板(GSM8K 天花板 vs MATH)。

### dsv2 λ₊ 编译定案(2026-09-01):**c≈1.2,MLA cell 失配大一个数量级(论文要点)**
静态(vLLM vs HF,980 万 token):c=1.22、ε̂=1.72%、τ*=12.58、λ₊=15.35、GPD ξ=0.71;
训练配置(冒烟 ktdump,SGLang vs HF-FSDP,fresh 1.1 万 token 分 bin):c=1.15、ε̂=2.25%、
τ*=13.10、**λ₊=15.02**。两条独立引擎对交叉一致 → DeepSeek-V2-Lite(MLA)的尺度常数
是 Qwen1.5-MoE 的 7-8 倍、dense 的 30 倍,失配真实且巨大;修正算子在此 cell 的
预期效应最大。**部署 LAMP=15.0**,ours×3 种子已提交(s1 cs/s2 dw87/s3 m13h)。
勘误:此前"dump MAD=0.0035 说明训练系统失配小"读数错误——全体中位数被 φ 地板
高置信 token 主导;分 bin 后 φ=0.94 处 MAD=1.21。尺度族在 dsv2 上 R²=0.95(静态 0.83,
U 形偏离,附录如实报);E[k]−1=−0.003 校准通过。

### 引擎审计定案(2026-09-02):SGLang 0.5.10.post1 的架构选择性采样退化
精确对照(train 前 2000 题、AReaL 逐字 prompt、avg@4、t=1、mt1024,唯一变量=引擎):
| 模型 | SGLang(训练奖励起点) | vLLM | 差距 |
| qwen1.5moe | 0.70 | 0.7013 | 0.0(完美一致,方法学验证)|
| dsv2 | 0.35-0.39 | 0.7688 | −39 |
| q30b | 0.70 | 0.9550 | −25 |
勘误:此前三个 avg@8 对照(64.2/94.9/49.0)带 prompt 措辞 + split 混杂,作废。
含义:cell A 全部结果无恙;cell D/E 的 RL 是向被引擎打坏的行为策略学习,
"全线低于 base"由此而来;旁证:sglang#21696(0.5.9→0.5.10 质量回退)、#20069。
论文角度:引擎失配不是假想敌——部署栈在某些 MoE 家族上白送 25-39 分退化,
RL 全盘继承;审计方法(训练奖励起点 vs 同协议 vLLM)本身可写成一节。
待用户拍板的路线:(a) D/E 改用 vLLM rollout 重跑(AReaL 支持 vllm backend,
模型不换、设计不变,先冒烟);(b) D/E 转为审计/边界案例章节,主表保 cell A;
(c) 换 SGLang 版本尝试修复(工程未知数最大)。

### D/E 全量重启(2026-09-03,用户指令直接跑):vLLM rollout,TAG=vllm
66 job = 11 方法 × 3 种子 × 2 cell;dsv2 EXTRA=rollout.backend=vllm:d4p1t1(LAMP=15.0),
q30b vllm:d2p1t2(LAMP=0.95,16h);分流 dsv2 s1→dw87/s2→m13h/s3→cs(维护解除后起),
q30b s1→m13h/s2→dw87/s3→cs;排除维护节点 dw-1-3/1-4/1-5/2-1 及 dw-2-4。
SGLang 版 dsv2 s1 残队 7 job 已取消(引擎无效);SGLang 版已完成数据 → 附录引擎审计。
两个 SMOKE=1 canary(13568659/660)仍在,先到先验证。

### vLLM 后端开荒记录(2026-09-03~05,八+一层修复,全部运行时生效)
① venv 缺 vllm → uv 装 0.16.0(torch 2.9.1 钉死);② FIPS OpenSSL 杀服务器 → vllm_remote Popen 注入
OPENSSL_CONF=/dev/null;③ opencv-python-headless 自带 libcrypto 炸 FIPS → 卸载(OPSD 老坑);④ flashinfer-cubin
0.6.7 vs flashinfer 0.6.3 → 对齐 0.6.3;⑤ vLLM custom all-reduce 'invalid argument'(与 sgl_kernel 同病)→
vLLMConfig 加字段 + yaml disable_custom_all_reduce;⑥ awex 插件硬依赖 megatron → 装 megatron-core 0.16.1;
⑦ AReaL 包装器给 vLLM 0.16 的 build_app 传 model_config → 按签名自适应;
⑧ xccl 权重同步到 vLLM 挂死(FSDP COALESCED all-gather 等 2h 超时,q30b 13591013 复现)→ yaml
weight_update_mode: disk。q30b vLLM rollout 生成已确认正常(5-6k tok/s)。
教训:批量与修复并行时,应 hold 队列到 canary 绿灯;本轮因此损失约 100 个 job 的排队时延。
- 2026-09-05:vLLM 后端 ⑨ 权重同步:xccl 挂死→disk;disk 在多 job 并发下 reload 57GB 超时(>3600s)/500,
  q30b 首批 5 个 job 全在 update_weights 死;两个 "COMPLETED" 是静默失败(且 eval chain 误评了
  weight_update_v1 目录,已取消并修 chain 只取 epoch*)。批量 27 个 hold;request_timeout→10800;
  只放行 1 个 q30b(30h 限时)+ dsv2 canary 端到端验证。**首奖励 0.716 = SGLang 时代 0.70 → "引擎缺陷"
  结论存疑,差距在 AReaL 路径内部;离线复刻诊断(diag_q30b/dsv2)排队中。**

### DSV2 训练奖励 0.37 的真凶(2026-09-05):tokenizer 加载 bug,不是引擎
离线逐环节复刻 AReaL 路径(diag_areal_path.py):q30b 得 0.943(≈独立评测 0.955,差距不在
prompt/采样/奖励函数);dsv2 重现 0.361,PROMPT_TAIL 显示 prompt 空格全丢。根因:AReaL venv 的
transformers 5.3.0 把 DeepSeek-V2 的 byte-level BPE tokenizer 解析成 slow LlamaTokenizer,编码出
**不同的 token id**(28 vs 32,从第 4 个起全不同),模型看到分词错乱的题 → 可解率减半;5.14.1
(gap venv,独立评测所用)正确。修复:hf_utils.load_hf_tokenizer 在落到 LlamaTokenizer 时改用
PreTrainedTokenizerFast(TokenizersBackend),验证三模型(dsv2 32 token 正确;Qwen 系不变)。
推论:SGLang 版 dsv2 全部结果(base 70.7 → 各臂 38-60)是在错误 prompt 下训练的,作废理由
更新为 tokenizer bug;"SGLang 引擎缺陷"结论撤回(vLLM 下 q30b 起始奖励同为 0.716)。
q30b 的 0.716 vs 0.943 差距仍在在线路径中(tokenization 两 venv 一致),等 rollout dump。

### q30b 训练奖励 0.716 的真凶(2026-09-05):thinking 模式未关(AReaL 在线路径 bug)
rollout dump 1004 条:think_rate=1.000,trunc_rate=0.756(1024 上限),reward|finished=1.0、
reward|truncated=0.615,均值 0.709≈0.716。在线 prompt 尾部 '<|im_start|>assistant\n'(无空 think 块):
areal/v2/inference_service/data_proxy/tokenizer_proxy.py 调模板未传 enable_thinking,Qwen3 默认开
thinking;RLVRWorkflow 默认 False,离线复刻因此得 0.943。修复:kw.setdefault("enable_thinking", False)。
Qwen1.5/DSV2 无 thinking 模式,不受影响(cell A 无恙)。
**推论**:此前所有 q30b 运行(含 SGLang 冒烟 0.70→0.72)都在 thinking+截断的畸形 regime 下,作废。
修复后 q30b 在 GSM8K 上起始奖励 ≈0.94 → 饱和问题原样回归。
- q30b disk 模式实测 ~15 min/step → 87 步 ≈22h;批量 TimeLimit 统一延至 30h(2026-09-05)。canary 13591641 活过 4 步,disk 权重同步循环稳定。
- ⑩ RolloutCallback.request_timeout 600→7200(disk 重载在 NFS 争用下 >10min 触发 Read timeout,canary 13591641 第 5 步死于此)。
- ⑪ _update_weights_from_disk 的 name_resolve.wait(timeout=120) → 3600:trainer 落盘 57GB 超 120s 时服务器侧 500(13591014 首次更新即死)。
- ⑫ 真实 thinking 开关位置:areal/experimental/openai/client.py(ArealOpenAI 走 extra_body.chat_template_kwargs);tokenizer_proxy 补丁不在路径上。默认注入 enable_thinking=False。13593148 首奖励 0.7148、seq_len≈1045 证明前一补丁无效。
- ⑫(续)client.py 补丁首次未落地(字符串不匹配),用正则重打成功(line 540);所有旧配置 running job 已取消重提(nocorr s1/fullis s1 m13h,fullis s2/tis s2 dw)。
- ⑫(终):真正生效的模板调用在 client.py hf 分支(原 775 行,MathAgent 经 proxy→ArealOpenAI→此处);已在该处默认 enable_thinking=False,并让 MathAgent 显式传 extra_body.chat_template_kwargs。13593184(seq_len 1053、奖励 0.69)证明前两次补丁不在路径上。
- ⑫(完):client.py 三处 apply_chat_template(540/775/1230)均默认 enable_thinking=False,0 处裸调用残留;MathAgent 亦显式传 chat_template_kwargs。
- ✅ 2026-09-05 17:xx:全修复 q30b 首奖励 0.953、seq_len/avg 305(13593188 fullis s1)——thinking 已关,与独立评测 0.955 一致;vLLM+disk 管线 + 12 层修复全部生效。q30b GSM8K 起点 0.95 → 饱和确认,cell E 训练集决定待用户。
- 复核:13591026(q30b kpop s2,dw/A100)首奖励 0.952、seq_len 297 —— 两分区两 job 均确认 thinking 关闭。
- 13593188(q30b fullis s1)10 步存活,1h43m(≈10 min/step 含 disk 同步;87 步 ≈15h,30h 限时充裕);奖励平在 0.950-0.953(饱和)。vLLM+disk 管线在真实节奏下稳定。
- ✅ 2026-09-07:dsv2 canary(13591048)首奖励 0.794、seq_len 259 —— tokenizer 修复生效(SGLang 时代 0.37)。取消该 1-epoch 冒烟(trial 名与正式 nocorr s1 冲突),提交 dsv2 正式批量 33 job(TAG=vllm,LAMP=15.0)。
- 2026-09-07 清单核对(sacct SubmitLine):q30b 完成 4(fullis s1/kpop s2/kpopfix s2/nocorr s1,均 no-thinking 有效)、在队 19、补提缺失 10(tis/icepop/kpop/kpopfix/seqtis/seqmis/gspo/fp16/ours 的 s1 + nocorr s2);dsv2 33 全在队;清除 2 条污染评测行(fp16vllm_s1、nocorrvllm_s2 评的是 weight_update 快照)。
- 调度修正(09-07):cs QOS 上限 24h → 11 个 q30b s3 job 由 30h 改 24h(A100 实测 ~9 min/step,13h 足够);cs 再入维护 → 11 个 dsv2 s3 迁 m13h/gpu。队列 61 kt-cells(3 running)。
- 2026-09-07 Overleaf 992331c:Table 1 填 DeepSeek base 行(70.66/25.40/67.67/6.25/6.08)与 Qwen3-30B base+3 个单种子行(nocorr s1/exact s1/KPop† s2),caption 注明 in-progress 与 GSM8K 饱和。清单:dsv2 33 pending、q30b 4 done/3 run/26 pending,无缺口。
- ⑬ m13h 上 dsv2 vLLM worker 死于 Triton/inductor JIT 编译(gcc cuda_utils.c 失败,缺 Python.h):我在改 account-agnostic 时注释掉了 env.sh 的 C_INCLUDE_PATH,现改为从 venv 解释器动态取 include 目录。13602494 因此失败;13602497(同节点,已起跑)预计同死,后续 job 自动继承。
