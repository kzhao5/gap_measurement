# OWNERSHIP — tianruny 账号提交的 job

> 策略(2026-09-09 与 kzhao2 商定):**每个 cell×method 先只补一个 seed**,把 11 个算子的
> held-out 覆盖填满;variance(多 seed)留到覆盖完成之后再铺。
> 因此实际提交数远少于 TODO_QUEUE.md 的 40 个——归档掉无效行后,q30b 已有 10/11 个方法
> 各有一个 seed,dsv2 已有 4/11。

生成时间:2026-09-11 22:33:29 MDT

| JobID | cell | method | seed | 分区/QOS | 预期评测 tag |
|---|---|---|---|---|---|
| 13621177 | dsv2 | kpopfix | 1 | dw/dw87 | `suite_dsv2_kpopfixvllm_s1` |
| 13621178 | dsv2 | seqmis | 1 | dw/dw87 | `suite_dsv2_seqmisvllm_s1` |
| 13639468 | dsv2 | gspo | 1 | dw/dw87 | `suite_dsv2_gspovllm_s1` |
| 13621180 | dsv2 | fp16 | 1 | dw/dw87 | `suite_dsv2_fp16vllm_s1` |
| 13639641 | dsv2 | ours | 1 | dw/dw87 | `suite_dsv2_oursvllm_s1` |
| 13626596 | dsv2 | icepop | 3 | dw/dw87 | `suite_dsv2_icepopvllm_s3` |
| 13621183 | dsv2 | kpop | 3 | m13h/gpu | `suite_dsv2_kpopvllm_s3` |
| 13643927 | q30b | seqtis | 1 | dw/dw87 | `suite_q30b_seqtisvllm_s1` |

## 说明
- `ours` 臂用 LAMP=15.0(dsv2)/ 0.95(q30b);其余方法不加 LAMP。TAG 一律 `vllm`。
- dsv2 用 `rollout.backend=vllm:d4p1t1` + 12h;q30b 用 `vllm:d2p1t2` + 30h。
- dsv2 的 icepop / kpop 取 **seed 3** 而非 s1,因为 s1 可能是 kzhao2 侧仍在跑的那个 kt-cells job,
  换 seed 既不重复组合也不留空白;若其 s1 也落地,则该方法自然获得 2 个 seed。
- checkpoint 落在 `~/nobackup/autodelete/areal_rl/experiments/checkpoints/$USER/`,跨账号不冲突。

## 最终状态(2026-09-11):8 个组合全部产出结果,两个 cell 均 11/11

上表是**最终成功的** job。过程中共提交 13 次,6 次因挂起或 I/O 超时风险而重提:

| 组合 | 提交次数 | 最终 JobID | 重提原因 |
|---|---|---|---|
| dsv2 kpopfix s1 | 1 | 13621177 | — |
| dsv2 seqmis s1 | 1 | 13621178 | — |
| dsv2 kpop s3 | 1 | 13621183 | 收尾挂起,训练有效,手动接评测 |
| dsv2 fp16 s1 | 1 | 13621180 | 收尾挂起,训练有效,手动接评测 |
| dsv2 icepop s3 | 2 | 13626596 | 首次训练中挂死(12× rollout_complete 超时) |
| dsv2 gspo s1 | 2 | 13639468 | 首次在 dw-2-3 上重载 1400s,推算必超时,主动取消 |
| dsv2 ours(CIS) s1 | 2 | 13639641 | 首次在 epoch 边界挂死 |
| **q30b seqtis s1** | **3** | **13643927** | 前两次均在 epoch 边界挂死(共烧 12h14m) |

诊断与修法见 `NOTES_tianruny.md` 的 R1–R11。

## E3 dose–response(2026-09-13/14 新增,cell A = Qwen1.5-MoE-A2.7B-Chat)

来源:老师 `CIS_待补实验.md` 的 **E3(受控噪声注入的 dose–response)**。
脚本 `~/nobackup/autodelete/env_local/dose.sbatch` —— `rl/fp8.sbatch` 的参数化副本,
算子分支与 `rl/cells.sbatch:43-44` **逐字一致**(见 NOTES R30)。
**未改动任何 yaml / recipe / 算子参数**;唯一新增的是 `kv_cache_dtype` 与 `attention_backend` 的传参。

| JobID | kv_cache_dtype | method | 硬件 | mem_frac | attn | 状态 | GSM8K held-out |
|---|---|---|---|---|---|---|---|
| 13671240 | fp8_e4m3 | nocorr | m13h-1-2 (H200) | 0.8 | triton | COMPLETED 6:36:52 | **0.6217** (820/1319) |
| 13671276 | fp8_e4m3 | fullis | m13h-1-2 (H200) | 0.8 | triton | COMPLETED 6:41:56 | **0.6020** (794/1319) |
| 13671277 | fp8_e4m3 | ours(CIS) | cs-2-1 (H100) | 0.8 | triton | RUNNING epoch 3/3 | 待评测 |
| 13675244 | fp4_e2m1 | nocorr | cs-2-2 (H100) | 0.6 | triton | RUNNING epoch 2/3 | 待评测 |
| 13675697 | fp4_e2m1 | fullis | cs2 待分配 | 0.55 | triton | PENDING(起 09-14 11:30) | — |
| 13675698 | fp4_e2m1 | ours(CIS) | cs2 待分配 | 0.55 | triton | PENDING(起 09-14 12:36) | — |

上表的硬件 / `mem_frac` / `attn` **不是按提交意图记的,而是逐个从该 trial 已解析的
`experiments/logs/$USER/kt-dose/<trial>/config.yaml` 与 `sacct NodeList` 读出来的实际值**
(理由见 NOTES R22:目录名不可信)。

评测 job:`13673240`(nocorr,9:15)、`13676008`(fullis,5:13),均 COMPLETED,走 `rl/eval_tp.sbatch`。

### E3 的提交历史:9/13 起共 21 次,**只有 2 次落地**

| 批次 | JobID | 结局 | 原因 |
|---|---|---|---|
| A100 试跑 | 13664168 / 13664171 / 13671239 / 13671272 | FAILED(各 10–12 分钟) | A100(sm80)在 CUDA-graph 捕获路径上拒绝 fp8 KV,降显存无效 → 放弃 A100(NOTES R15/R17) |
| A100 批量 | 13664169/70、13671215–13671220 | CANCELLED | 同参数必然同样失败,主动取消 |
| **Hopper 落地** | **13671240 / 13671276** | **COMPLETED** | H200 + `attention_backend=triton`,不需要其他 workaround |
| fp4 首跑 | 13671278 | FAILED(21:26) | `mem_fraction=0.8` 下 mxfp4 挤压训练侧显存 → **训练进程** CUDA OOM(NOTES R25) |
| fp4 批量 | 13675133 / 13675134 | CANCELLED | 同上,主动取消以待验参数 |
| **fp4 降显存** | **13675244** | RUNNING | `mem_fraction_static=0.6` 跑通,证实 R25 的诊断(NOTES R26) |

两次失败的**归属不同**,这是 E3 最花时间的地方:A100 那批死在**推理侧**(架构硬门槛,不可调),
fp4 那次死在**训练侧**(colocate 显存挤压,是可调参数)。
按"失败即不可用"处理会错误放弃整个 fp4 档。

### E3 专用说明(与主表规则不同,勿套用)
- **`LAMP` 不传**:E3 是 cell A,不在 `dsv2=15.0 / q30b=0.95` 的规则内;
  三个 `ours` 臂一律取 `dose.sbatch` 的默认 `KT_SIGMA_B=2.3`(NOTES R30)。
- **TAG 不是 `vllm`**:E3 走 sglang + `eval_tp.sbatch`,评测 tag 形如 `dose_<dtype>_<method>`,
  **结果只写进日志的 `RESULT` 行,不进 `eval_suite.tsv`**。曲线用 `env_local/dose_curve.sh` 汇总。
- **`mem_fraction_static` 跨档不一致**(fp8_e4m3=0.8,fp4=0.6/0.55):
  它只控制 sglang 推理引擎的显存预留,不涉及算子选择或训练超参,档内对比不受影响,但需标注。
- **既有的 bf16 / fp8_e5m2 两行不可与本批合读** —— 硬件、attention backend、训练轨迹三重差异(NOTES R21)。

### 补充:第 7 个 arm —— bf16 对照(2026-09-14 03:15 提交)

| JobID | kv_cache_dtype | method | 分区/QOS | mem_frac | attn | TAGSFX | 状态 |
|---|---|---|---|---|---|---|---|
| 13676237 | **bf16** | nocorr | cs2 / cs | 0.8 | triton | `-ctrl` | PENDING |

目的:曲线里既有的 bf16 / fp8_e5m2 两行来自 2026-08 的另一批运行,与本批在
**硬件**与**训练轨迹**上不同(评测协议相同,见 NOTES R31)。
让 bf16 档也走同一条 dose 路径后,整条曲线可只用自跑数据,不再依赖旧行。

另有评测作业 `13676235`(`kt-suite` @ dw-1-5):对 `nocorr-fp8_e4m3-h200` 的最终 epoch
做**五 benchmark 复评**,tag `dose5_fp8_e4m3_nocorr`,用于测量两套评测协议之差。
该 tag 的行会写进 `results/eval_suite.tsv`,前缀 `dose5_` 与主表的 `suite_`、
曲线用的 `dose_` 均不冲突(`dose_curve.sh` 的正则只匹配 `RESULT dose_`,已验证不会误收)。

### 更正:`13676237` 已取消,bf16 档改投 m13h 并投满三臂(2026-09-14 03:34)

上一小节列出的 `13676237`(bf16 / nocorr / cs2)**已取消**,原因见 NOTES R36:
它在 cs2 的预计起始为 `2026-09-15T00:00:00`,而 cs2 仅有的两个节点被我自己的
`13671277` / `13675244` 占满 —— 它实际是在排我自己的队。m13h 经逐节点核算确有
三个整节点空闲,迁过去后立即开跑。同时投满 bf16 三臂(理由见 NOTES R37:
bf16 是 dose 曲线的锚点,缺 `ours` 则整条曲线没有起点)。

| JobID | kv_cache_dtype | method | 分区/QOS | mem_frac | attn | TAGSFX | 提交时状态 |
|---|---|---|---|---|---|---|---|
| 13676300 | bf16 | nocorr | m13h / gpu | 0.8 | triton | `-ctrl` | **RUNNING** |
| 13676301 | bf16 | fullis | m13h / gpu | 0.8 | triton | `-ctrl` | PENDING |
| 13676302 | bf16 | ours(CIS) | m13h / gpu | 0.8 | triton | `-ctrl` | PENDING |

`env_local/dose_jobids.txt` 已同步(移除 13676237,新增三行)。
`ours` 臂**不传 `LAMP`**,取默认 `KT_SIGMA_B=2.3`(cell A 不在 LAMP 规则内,见 NOTES R30)。

### 五 benchmark 复评(新协议 `eval_suite.py`)已完成两个 checkpoint

| JobID | 评测对象 | EVAL_TAG | 用时 | 状态 |
|---|---|---|---|---|
| 13676235 | `nocorr-fp8_e4m3-h200` / `epoch2epochstep28globalstep86` | `dose5_fp8_e4m3_nocorr` | 00:10:20 | COMPLETED |
| 13676298 | `fullis-fp8_e4m3-h200` / `epoch2epochstep28globalstep86` | `dose5_fp8_e4m3_fullis` | 00:11:55 | COMPLETED |

**为什么要对已经评过的 checkpoint 再评一次**:这两个 checkpoint 本来就有旧协议
(`eval_gsm8k.py`)的结果,再用新协议评一遍,**同一个 checkpoint 上的新旧之差
就是评测协议差的直接测量值**,无需重训。两次分别得 +6.06 / +5.99(极差 0.07),
恰好解释了仓库里 SIGMA_TIS 与主表之间那 5.66 分的差(见 NOTES R33 / R38)。

结果(等权五 benchmark 均值):`nocorr` **34.14**、`fullis` **32.56**。
逐 benchmark 数值、协议差分解、以及与主表 cell A 参照值的对照见 NOTES **R38 / R39 / R40**;
汇总脚本 `env_local/dose5_curve.sh`(不在仓库内)。

共 **12 行**写入 `results/eval_suite.tsv`,tag 前缀 `dose5_`,
与主表的 `suite_`、GSM8K 曲线用的 `dose_` 均不冲突
(`dose_curve.sh` 的正则只匹配 `RESULT dose_`,已验证不会误收)。

### fp8_e4m3 档收官:第三臂与两次评测(2026-09-14 04:2x–05:0x)

| JobID | 内容 | 结果 | 用时 |
|---|---|---|---|
| 13671277 | fp8_e4m3 / `ours` 训练 | 87/87 步,epoch 3/3,**干净收尾**,评测链自动触发 | 4:55:23 |
| 13676579 | `eval_tp`(旧协议 GSM8K) | `dose_fp8_e4m3_ours` **= 0.6361**(839/1319) | — |
| 13676577 | `eval_suite`(五 benchmark) | `dose5_fp8_e4m3_ours` 等权 **33.95** | 00:10:54 |

**fp8_e4m3 档(三臂齐全)**:

| 方法 | 旧协议 GSM8K | 五 benchmark 等权 |
|---|---|---|
| `nocorr` | 62.17 | 34.14 |
| `fullis` | 60.20 | 32.56 |
| `ours`(CIS) | 63.61 | 33.95 |

判读见 NOTES **R45**(预登记落在"与 nocorr 不可分")与 **R46**
(五 benchmark 独立印证;math-verify 下 `ours − nocorr` 仅 +0.08)。
`eval_suite.tsv` 新增 6 行,tag `dose5_fp8_e4m3_ours`。

### 更正:`13675698`(fp4 / `ours` @ 0.55)已取消,按 0.6 重投为 `13676652`(2026-09-14 04:5x)

上文 E3 表中 `13675698` 那一行(`0.55` / `PENDING(起 09-14 12:36)`)**已失效**。

**原因见 NOTES R47**:`13675697` 在 0.55 下跑通首步时报出 `device used/total = 78.89/79.18`,
**余量仅 0.29 GB**;而 `13675244` 在 **0.6** 下是 `77.34/79.18`,**余量 1.84 GB**。
两者 `allocated` 同为 53.41 —— 真实需求相同,降 `mem_fraction_static` 并未创造余量,
只是把它交给训练侧的缓存分配器吸收,总占用反而更逼近上限。
`ours` 臂还要额外跑 σ-TIS,在 0.29 GB 余量下 OOM 风险高,故改投更安全的 0.6。

| JobID | 档位 | 方法 | 分区/QOS | mem_frac | attn | TAGSFX | 状态 |
|---|---|---|---|---|---|---|---|
| ~~13675698~~ | fp4_e2m1 | ours | cs2 | ~~0.55~~ | triton | `-lowmem` | **CANCELLED(未开始)** |
| **13676652** | fp4_e2m1 | ours | cs2 / cs | **0.6** | triton | `-lowmem` | PENDING |

重投前已按 R4 核对目标目录 `ours-fp4_e2m1-lowmem` 不存在残留(该作业从未运行)。
`env_local/dose_jobids.txt` 已同步。
`13675697` 维持 0.55 不变 —— 已在正常推进,且该参数不改变实验语义。

**fp4 档三臂的最终参数**:`nocorr` 0.6(13675244)、`fullis` 0.55(13675697)、`ours` 0.6(13676652)。

### fp4_e2m1 / `nocorr` 收官,含一次收尾挂起的处置(2026-09-14 07:2x–07:5x)

| JobID | 内容 | 结果 |
|---|---|---|
| 13675244 | fp4_e2m1 / `nocorr` 训练 | 87/87 步、epoch 3/3 **训练有效**;但**收尾挂起**(RUNNING + 静默 754s),已人工处置 |
| 13677043 | `eval_suite`(五 benchmark) | `dose5_fp4_e2m1_nocorr` 等权 **12.73** |
| 13677051 | `eval_tp`(旧协议 GSM8K,人工补投) | `dose_fp4_e2m1_nocorr` **= 0.3609**(476/1319) |

**挂起处置流程**(可复用):核对三项条件齐备(Slurm `RUNNING` + `epoch 3/3` + 日志静默 >600s)
→ 判定训练有效、收尾失败 → 人工补投评测 → **取消挂起作业释放节点**。
本次释放的 cs-2-2 立即被排队中的 `13676652`(fp4/ours)接手,取消动作有直接调度收益。

**结果**:等权五 benchmark **12.73**,比未训练的 base(28.93)**低 16.20 分** ——
fp4_e2m1 下不加修正的 RL **把模型训坏了**。诊断与四条独立证据见 NOTES **R53**。

注:该 checkpoint 在提交评测前已做逐文件字节校验(`model.safetensors` 28,632,153,464,
与已知完好的 fp8_e4m3 checkpoint 完全一致),**排除了"评到半成品"这一解释**(R52)。

### bf16 / `ours` 收官(2026-09-14 08:0x–08:2x)

| JobID | 内容 | 结果 |
|---|---|---|
| 13676302 | bf16 / `ours` 训练 | 87/87、epoch 3/3,**干净收尾**,评测链自动触发 |
| 13677092 | `eval_tp`(旧协议 GSM8K) | `dose_bf16_ours` **= 0.6899**(910/1319) |
| 13677094 | `eval_suite`(五 benchmark) | `dose5_bf16_ours` 等权 **35.77** |

该臂的五 benchmark 评测由**预挂的 checkpoint 门控自动提交**(逐文件字节校验通过后第 1 次检查即命中),
无需人工介入;门控机制见 NOTES R52。等权均值落在 NOTES **R54** 预先算出的区间 [35.00, 36.16] 内。

**注意**:本臂数字**不可单独用于 R49 的判决**——判决量是 `ours − nocorr`,
需 `13676300`(bf16 / nocorr)落地后配对计算。

### fp4_e2m1 / `fullis` 收官(2026-09-14 09:0x)

| JobID | 内容 | 结果 |
|---|---|---|
| 13675697 | fp4_e2m1 / `fullis` 训练 | 87/87、epoch 3/3,**干净收尾**(同档 nocorr 则挂起),评测链自动触发 |
| 13678480 | `eval_tp`(旧协议 GSM8K) | `dose_fp4_e2m1_fullis` **= 0.6035**(796/1319) |
| 13678478 | `eval_suite`(五 benchmark) | `dose5_fp4_e2m1_fullis` 等权 **32.32** |

五 benchmark 评测由**三臂合并门控 `b80aena2f` 自动提交**(tag 与 EVAL_PATH 均经事前手工核对)。
等权均值落在 NOTES **R56** 预先算出的区间 [31.88, 32.91] 内。

**fp4 档(3 臂中 2 臂完成)**:

| 方法 | 旧协议 GSM8K | 五 benchmark 等权 |
|---|---|---|
| `nocorr` | 36.09 | **12.73**(比 base 28.93 低 16.20) |
| `fullis` | 60.35 | **32.32** |
| `ours`(CIS) | 运行中(13676652,约 12:35) | — |

**修正与否在该档相差 +19.58 分**(五 benchmark 刻度),诊断见 NOTES **R57**。

### bf16 档 `nocorr` 收官 = R49 判决臂(2026-09-14 09:5x–10:1x)

| JobID | 内容 | 结果 |
|---|---|---|
| 13676300 | bf16 / `nocorr` 训练 | 87/87、epoch 3/3,**干净收尾** |
| 13680153 | `eval_tp`(旧协议 GSM8K) | `dose_bf16_nocorr` **= 0.6194**(817/1319) |
| 13680122 | `eval_suite`(五 benchmark) | `dose5_bf16_nocorr` 等权 **34.99** |

五 benchmark 评测由替代门控 `bmt3a66e7` 自动提交 —— **原门控 `b80aena2f` 的 90 分钟寿命
约 09:48 到期,而本臂 checkpoint 直到之后才写完**;若未按 NOTES R58 及时重挂,
这个判决臂的五 benchmark 评测**根本不会被提交**。

**判决结果见 NOTES R61:登记结论为『无法分解』**
(两把尺子方向均指向管线假说,但距边界仅 0.89 / 0.13 个 SE,触发 R49 的附加条款)。
事后补做的对照检验(本管线 +0.78 vs 历史 +3.48,z=2.39)支持管线假说,但非预登记。

### bf16 档收官(2026-09-14 10:2x)

| JobID | 内容 | 结果 |
|---|---|---|
| 13676301 | bf16 / `fullis` 训练 | 87/87、epoch 3/3,干净收尾 |
| 13680408 | `eval_tp`(旧协议) | `dose_bf16_fullis` **= 0.6801**(897/1319) |
| 13680403 | `eval_suite`(五 benchmark) | `dose5_bf16_fullis` 等权 **34.89** |

**bf16 档三臂(五 benchmark 等权)**:`nocorr` 34.99 / `fullis` 34.89 / `ours` 35.77,
**跨度 0.88 分,三者统计不可分**(SE_diff 1.09,门槛 3.05)。诊断见 NOTES **R63**。

**E3 九格已填八格**,只剩 `fp4 / ours`(13676652,约 12:58),
其五 benchmark 评测已由门控 `bx4u3z21b` 预挂(寿命至 13:58),
旧协议评测与挂起由守卫 `bo85apg04` 覆盖。

### fp4_e2m1 / `ours` 收官 = E3 最后一格(2026-09-14 13:2x–13:4x)

| JobID | 内容 | 结果 |
|---|---|---|
| 13676652 | fp4_e2m1 / `ours` 训练 | 87/87、epoch 3/3、**COMPLETED 05:47:19**(守卫曾误报挂起,见 NOTES R65) |
| 13684795 | `eval_suite`(五 benchmark) | `dose5_fp4_e2m1_ours` 等权 **29.84** |
| 13685113 | `eval_tp`(旧协议,评测链自动提交) | 运行中 |

五 benchmark 评测由门控 `bx4u3z21b` 自动提交(第 187 次检查捕获 checkpoint 写完;
寿命 224 分钟用掉 187,余 37 —— 若沿用最初 90 分钟默认值会在写完前一小时退出)。

**E3 九格全部填满**,完整网格与判读见 NOTES **R67**;
CIS 的总体结论(含本格)见 NOTES **R64 + R67**。


### E3 全部结束(2026-09-14 13:4x)

`13685113`(`eval_tp`,评测链自动提交)返回 `dose_fp4_e2m1_ours` = **0.5792**(764/1319)。

**九个训练臂 × 两种评测协议 = 十八次评测,全部完成。** 完整网格与判读见 NOTES **R67**。

fp4 档旧协议行:`nocorr` 36.09 / `fullis` 60.35 / `ours` **57.92**;
`ours − fullis` 在旧协议为 **-2.43**、五 benchmark 为 **-2.48**,**两把尺子同向**。

---

## 补充实验第二批(2026-09-14,依 `docs/CIS_待补实验.md` 与结果文档 §10.4 提交)

### A. `fp8_e5m2` 一档 × 3 算子 —— 补齐老师四档阶梯里缺的那一级

| JobID | 档位 | 算子 | 分区/QOS | mem_frac | attn | TAGSFX |
|---|---|---|---|---|---|---|
| 13689388 | fp8_e5m2 | `nocorr` | m13h / gpu | 0.8 | triton | `-ctrl` |
| 13689389 | fp8_e5m2 | `fullis` | m13h / gpu | 0.8 | triton | `-ctrl` |
| 13689390 | fp8_e5m2 | `ours`(CIS) | m13h / gpu | 0.8 | triton | `-ctrl` |

**参数与 fp8_e4m3 档完全一致**(mem 0.8 / triton),以保证档间只有 `kv_cache_dtype` 一个变量。
提交前已按 R4 复查三个目标目录洁净;`env_local/dose_jobids.txt` 已同步,汇总脚本可直接取数。

**为什么补这一档**:老师的阶梯是 bf16(u=2⁻⁸)→ e4m3(2⁻⁴)→ **e5m2(2⁻³)** → fp4(2⁻²),
先前跑了第 1/2/4 级。而本实验发现的阶跃**恰好落在被跳过的那一级两侧**
(`nocorr` 由 e4m3 的 34.14 塌到 fp4 的 12.73)—— e5m2 是唯一能定位阈值的一级。

**提交时 Hopper 无任何节点有 8 张空卡**(m13h 最多 6),三臂均为 PENDING,属正常排队。

### B. 基线诊断测量 —— 老师设计的 P1(零代码改动)

| JobID | ARCH | 模型 | 分区 |
|---|---|---|---|
| 13689391 | `moe` | Qwen1.5-MoE-A2.7B-Chat | dw(4×A100) |
| 13689392 | `dense` | Qwen1.5-14B-Chat | dw(4×A100) |

**产出链**:`gen_vllm.py` → `tokens_<arch>.parquet`(含 `logp_infer`);
`recompute_train.py` → `logp_train`;二者给出 ε;
**μ = E[e^ε] 与 E[e^{2ε}] 由 `analysis/s0_closure.py:125-128` 直接算**(已有实现,无需新推导)。

**回答两个悬而未决的问题**:① kzhao2 的 A 项「7.07 对应哪个总体」;
② 给 E3 的 dose 曲线一个真实的 μ 分母 —— 目前 §10.3 的反向证据**因缺 μ 而只能是带条件的结论**。
`dense` 是老师 E2 明确要求的对照组(无路由的干净场)。

**注意**:这是本账号**首次**运行 measure 流程(此前 `DATA_ROOT` 为空、TODO B 项记为阻塞);
kzhao2 清除 `rl/` 与 `analysis/` 的硬编码后方才可行。

### 仍被阻塞、未提交的一项

**按档位的 μ(e4m3 / e5m2 / fp4)** 需给 `src/gen_vllm.py` 加量化参数
(argparse 现仅 `--arch/--shard/--num-shards/--smoke/--max-num-seqs`,`dtype="bfloat16"` 写死)。
属仓库代码改动,已建议并入 kzhao2 的 δ-dump(TODO **H** 节)。

### 更正:三个 `fp8_e5m2` 臂已迁移,作业号全部变更(2026-09-14 17:xx)

上一节表中的 `13689388 / 13689389 / 13689390` **均已取消并重投**。
原因与判据见 NOTES **R73 / R74**。

| 档位 | 算子 | 原 JobID | 现 JobID | 分区 / 实际 QOS | 预计起始 |
|---|---|---|---|---|---|
| fp8_e5m2 | `fullis` | ~~13689389~~ | **13689470** | cs2 / `cs` | 2026-09-14 21:40 |
| fp8_e5m2 | `nocorr` | ~~13689388~~ | **13689486** | cs2,eng,m13h / `gstandby` | 2026-09-16 07:00 |
| fp8_e5m2 | `ours` | ~~13689390~~ | **13689487** | cs2,eng,m13h / `gstandby` | 2026-09-16 07:00 |

**两条必须记住的坑**(R74):
1. 提交写 `--qos=standby`,Slurm **静默映射为 `gstandby`**。提交参数 ≠ 生效参数,
   必须用 `scontrol show job <id> | grep QOS=` 核对。
2. `sbatch --test-only` 回答的是「**再新增一个**作业能何时开跑」,
   **不计入自己已排队作业间的竞争**。同一提交规格,探测 03:40 而实际 9/16 07:00,差约 28 小时。
   **它的时刻不能当作已排队作业的 ETA。**

迁移的实得收益是 `ours` 臂 19:00 → 07:00,**12 小时**;`nocorr` 未变。
已决定不再按探测值反复重投(重投只会重复同一误判,且forfeit 排队位置)。

### B 项基线测量:已完成(2026-09-14 17:40)

| JobID | ARCH | 结果 | 用时 |
|---|---|---|---|
| 13689391 | `moe` | COMPLETED,8/8 分片,`MEASURE_DONE_moe` | 00:47:04 |
| 13689392 | `dense` | COMPLETED,8/8 分片,`MEASURE_DONE_dense` | 00:51:12 |

产出 48 个 parquet(gen 16+16、recompute 8+8)。**本账号首次跑通 measure 流程。**

**主要结果**(详见 NOTES **R76**):

| | moe | dense |
|---|---|---|
| token 级 μ = E[e^ε] | **1.000009** | **1.000007** |
| E[e^{2ε}] | 1.009756 | 1.000662 |
| 轨迹级 ESS(W) | 1.55% | 67% |
| min(ε) | **−13.01** | −0.874 |

- 与 `RESULTS.md:127` 的 `E[K]=1.000009/1.000007` **六位全同**,token 数与
  `report_{moe,dense}.json` 逐位相同 —— 管线复现了 2026-09-07 的历史运行。
- **token 级 μ 是恒等式**(`E[k]=Σp_train=1`),不随量化变化。
  老师 E3 预演的判据「μ 至少要涨一个数量级」因此不可用。
- **7.07 已定位**:`results/j1/report_full.json` 的 `/main/a2`,
  是 `j1_coupling.py:94` 方差对 σ 回归的**截距**,量纲是方差(实测 Var(log W)=6.849 对 7.067)。
- **1.0423 最接近的存量**是 `kt_moe_traj.json` 的 `table2[9].x`(负侧 0.999 分位数),
  12 个候选总体无一落在该值。

### 按档位诊断:已备妥,冒烟进行中(2026-09-14 18:17)

**仓库代码零改动。** 参数化副本置于 `~/nobackup/autodelete/env_local/`:

| 文件 | 相对原件的差异 |
|---|---|
| `gen_vllm_q.py` | ① `sys.path` 指向仓库 `src/`(副本不在 `src/` 下);② 新增 `--kv-cache-dtype`;③ 传入 `llm_kwargs`;④ `KT_DATA_ROOT` 重定向 `DATA_ROOT` |
| `recompute_train_q.py` | ①④ 同上 |
| `measure_q.sbatch` | cs3 / `qos=cs` / `gpu:b200:2`;8 分片改 **2 卡 4 轮**;带安全闸拒绝任何触碰基线目录的输出路径 |
| `smoke_q.sbatch` | 1 卡逐档冒烟,`auto` 档作正对照 |

**硬件约束**(NOTES R77):`fp8_*` 需 SM89+、`nvfp4` 需 **SM100**,
且**只有 FlashInfer 后端支持量化 KV** —— 全集群唯一入口是 **cs-3-1(B200)**,
且提交不抢占任何人(cs2 / m13h 的 `qos=cs`/`gpu` 会抢占 `fslcollab4` 的运行中作业,已排除)。

**冒烟作业 `13690930`**(cs-3-1,B200)`RUNNING`,逐档试 `auto / fp8_e4m3 / fp8_e5m2 / nvfp4`。
**结果尚未产出,不预判。**

**全量跑的切分**(由基线时长外推):基线 4 卡跑完 gen+recompute 用 47 分钟;
改 2 卡后轮数翻倍,约 **95 分钟/档**,四档合计约 **6.3 小时**。
贴着 8 小时墙跑单作业风险过高,故**一档一作业**,四个独立提交。

**必须随结果上报的限制**:E3 训练臂跑在 **sglang**(`fp8_e4m3`/`fp4_e2m1`),
本诊断走 **vLLM**(`fp8_e4m3`/`nvfp4`)—— 不同引擎、不同 kernel,fp4 两边格式名都不同。
它测的是「同档量化会引入多大噪声」,**不等于那几次训练实际经历的噪声**。

### 按档位诊断的实际经过(2026-09-14 18:0x–19:0x):四级计划缩为两级

**仓库代码零改动**;参数化副本在 `~/nobackup/autodelete/env_local/`
(`gen_vllm_q.py` / `recompute_train_q.py` / `measure_q2.sbatch` / `ckpt_gate.sh` / `ladder_moments.py`)。

| 作业 | 硬件 | 结果 |
|---|---|---|
| `13690930` 冒烟 | cs-3-1(B200,SM100) | **CANCELLED** —— FlashInfer MoE 为 SM100 做 JIT,静默烧 5.8 核 >12 分钟(NOTES **R78**) |
| `13691206` 冒烟 | eng-1-1(H200,SM90) | **COMPLETED 00:17:07**:`auto` OK、`fp8_e4m3` OK(43 秒)、`fp8_e5m2` **FAIL**(NOTES **R80**) |
| `13691373` 正式 | eng-1-1(H200,SM90) | **RUNNING**(起 18:53:14),三档 × 2 分片。**结果未出,不预判** |

**四级阶梯缩为两级的原因**:

| 档位 | 舍入 | 状态 |
|---|---|---|
| bf16 | 2⁻⁸ | ✓ 可测 |
| fp8_e4m3 | 2⁻⁴ | ✓ 可测 |
| ~~fp8_e5m2~~ | ~~2⁻³~~ | ✗ vLLM 0.26 内部不一致:`q` 恒为 `float8_e4m3fn` 而 `q_data_type` 取自 KV dtype(R80) |
| ~~fp4/nvfp4~~ | ~~2⁻²~~ | ✗ 需 SM100;B200 上 JIT 停滞;且与 sglang 的 `fp4_e2m1` 非同一格式(R78) |

**三项设计决定**(均有实测依据,非估计):
1. **2 分片而非 8**:28 个两分片组合的 `E[e^{2ε}]−1` 相对全量偏差 ≤**6.40%**,
   而要分辨的跨条件差异为 **14.7 倍**,余量约 230:1(NOTES **R79**)。每档约 35 分钟。
2. **bf16 也在 H200 重测**:基线测于 A100,跨档换硬件会混入硬件变量。
   分片选取确定性,故 H200 的 0/1 分片与 A100 基线的 0/1 分片是同一批轨迹。
3. **1 卡而非 2 卡**:eng-1-1 只余 1 张空卡;要 2 卡会抢占 `dgl2` 五个已跑 6+ 小时的作业。
   (`--qos=standby` 实际生效为 `gstandby`,**高于**他人的 `standby`,
   "standby 很客气"只在请求装得进空闲容量时成立 —— 以 `Preempts:` 行为准。)

**必须随结果上报的不对称**:训练臂跑 **sglang**(e5m2 / fp4_e2m1 均可用),
诊断走 **vLLM**(二者均不可用)。**能在 e5m2 上训练,却测不到 e5m2 的 ε 矩。**
因此 §10.3 那条"反向预测"的限制**不能被本次诊断完全解除**。

### 三个 e5m2 臂的评测门控

`dose.sbatch:55` 只自动接旧协议 GSM8K;五 benchmark 需人工门控。
`ckpt_gate.sh` 已备好并修正了一处**会评错 checkpoint** 的缺陷
(每个 epoch 的 checkpoint 都字节完整,只取"最后一个 epoch"会在训练中途误触发,见 NOTES **R81**)。
现要求 `epoch2` 前缀 + 字节完整 + tag 未在 `eval_suite.tsv` 中。

| 臂 | JobID | 门控 |
|---|---|---|
| fullis | 13689470(起 21:40) | 已挂(`bqiczg14a`,720 分钟) |
| nocorr | 13689486(起 09-16 07:00) | 待起跑后挂 |
| ours | 13689487(起 09-16 07:00) | 待起跑后挂 |
