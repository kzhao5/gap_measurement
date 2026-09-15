# 需要 kzhao2 侧完成的事项(tianruny 整理,2026-09-12)

背景:E3(受控噪声注入 dose–response)已由 tianruny 提交开跑,见文末。
**E1/E2/E4 被下面 4 项阻塞**,都不是 tianruny 能单方面解决的。

---

## A. 【需要你确认口径】μ = 7.07 在仓库里找不到

`CIS_待补实验.md` 的 E2 整段论证建立在"MoE 的 μ = 7.07,远超舍入噪声能产生的量级,
所以是尾巴撑起来的"。但仓库里的实测值是:

```
results/paper/paper_numbers.json
  E_k_moe   = 1.0000087      ← 应该就是 μ = E[k] = E[e^ε]
  E_k_dense = 1.0000072
  n_moe     = 12,293,115
```

`7.07` 在整个仓库里只出现过一次:`results/s0closure/report_moe.json` 的
`delta_R2 = 7.0797e-05`,与 μ 无关。

**为什么这条必须先定**:μ ≈ 1.0000087 ⟹ `σ_ε² ≈ 1.7e-5`、`σ_ε ≈ 0.0042`,
**完全落在 bf16 舍入噪声的量级内**。若如此,E2 的舍入模型是**直接对上的**,
不需要文档里设计的"两段式(主体对上 / 重尾没对上)"免责写法;而重尾的证据
应由 `ξ₊`(dense 0.066 vs MoE 0.491,仓库中已确认)和硬路由翻转富集来承担,
与 μ 是两个独立的量。

请确认 7.07 是:(a) 论文里另一个量(如 `E[e^{2ε}]`、某截断子集、Figure 2 某个 α 处的读数)、
(b) 笔误、还是 (c) 仓库里的 `E_k_moe` 算错了。tianruny 看不到 Overleaf 正文,无法判断。

### 【2026-09-14 已解决 —— 本节上面的内容保留为当时的判断,下面是结论】

**本节原文的「7.07 在整个仓库里只出现过一次」是错的。** 机械扫描 `results/` 下
全部 JSON/CSV 中落在 [6.9, 7.2] 的标量后命中:

```
results/j1/report_full.json    /main/a2 = 7.067329317760002
results/j1/report_fresh.json   /main/a2 = 7.148166680566494
```

`j1_coupling.py:94`:`a2, s2 = wls(ok["sigma"], ok["v"], ok["n"])`,
其中 `v = Sb.var(ddof=1)` 是桶内 `S` 的**方差**。
**`a2` 是「方差对 σ 回归」在 σ=0 处的截距,量纲是方差,不是 μ。**

tianruny 于 2026-09-14 跑完基线测量(job 13689391/13689392)后直接测得:

| | 实测 | 对照 |
|---|---|---|
| Var(log W)_moe | **6.849** | `a2` = 7.067,差 3.1% |
| sd(log W)_moe | **2.6171** | `sqrt(a2)` = 2.658,差 1.6% |

老师的推理是「μ = 7.07 ⟹ `μ ≈ exp(σ²/2)` ⟹ σ ≈ 2.0」。
**7.07 本身就是 σ²**(σ = 2.66)。量级判断成立,但读法多绕了一次开方-取指。

**本节第 15 行的原始读法是对的**:`E_k_moe = 1.0000087` **就是** μ = E[e^ε]。
tianruny 独立重跑测得 token 级 μ = **1.000009**(moe)/ **1.000007**(dense),
与 `RESULTS.md:127` 的 `E[K]=1.000009/1.000007` 六位全同,token 数也逐位相同。
`k_t = e^{ε_t} = p_train/p_infer` 在采样分布下满足 `E[k] = Σ p_train = 1`,
**这是恒等式,不是可调量**。

**关于你报的「base MoE 静态 μ = 1.0423」**:tianruny 枚举 12 个候选总体,
**没有一个落在 1.0423**(最近的是 token 级 `E[e^{-ε}] = 1.0585`)。
仓库里最接近的存量是 `results/kt/kt_moe_{time,traj}.json` 的
`table2[9] = {side:"-", level:0.999, x:1.04270218276978}`,即**负侧 0.999 分位数**。
**请确认你的 1.0423 是否取自该处**;若是,则它不是 μ。

完整推导、12 个总体的枚举表、以及两通道(moe/dense)对照见 `NOTES_tianruny.md` **R76**。

---

## B. 【需要你提供数据】原始测量 parquet 在 tianruny 账号里不存在

- `DATA_ROOT`(`~/nobackup/autodelete/gap_measurement`)是空的,**0 个 parquet**。
- `/home/kzhao2` 是 700,`Permission denied`,拿不到你那份。
- E1(测 δ)、E2(算 μ)、E4(Figure 1 双图)全部依赖它。

**两个选项,请选一个**:
1. 按 TRANSFER §7 第 6 条走 login 节点 /tmp 中转,把那 430MB 原始测量 parquet 传过来;
2. 或者告诉 tianruny 直接重跑测量。重跑是可行的——`data/prompts_math.jsonl`(2500 条)
   还在,`rl/measure.sbatch` 完整,且 **Qwen1.5-MoE-A2.7B-Chat 与 Qwen1.5-14B-Chat
   两个模型 tianruny 已经下载好了**(共 57GB)。E1 只需要 200–500 条序列,
   比现有 12.3M token 的规模小两个数量级,4 卡几十分钟即可。

---

## C. 【需要你改代码,或授权 tianruny 改】E1 的 δ dump

`src/gen_vllm.py` 目前**只存被采样 token 的 logprob**:
```
docstring 第 4–5 行:logp_infer is read from the sampling step itself
                    (SamplingParams.logprobs=0)
输出 schema(第 387–396 行):traj_id / prompt_id / pos / token_id / logp_infer
                            (+ route_infer / margin_infer)
```
E1 要的是**两个引擎的完整 pre-softmax logit 向量相减**,现在这条路径上没有。

**好消息是改动量不大**:
- 推理侧:`gen_vllm.py` 里**已经有** `GPUModelRunner.execute_model` 的包装器
  (第 170–186 行,原本用于抓 router logits),同一模式可以复用来抓最终 logits。
- 训练侧:`recompute_train.py` **已经拿到了完整的 `[T, V]` logits**
  (第 177 行 `sub = logits[r, rows].float()`),只是 gather 完就丢了,不丢即可。
- 存储建议:**只存 δ = logit_infer − logit_train 的差值**(一个向量而不是两个),
  V≈150k × float32,500 序列 × 若干位置约十几 GB,快盘放得下。

tianruny 目前遵守 HANDOFF 的"不改配方/yaml/算子参数",所有本地化都刻意绕开了仓库代码。
**请明确:这个改动由你做,还是授权 tianruny 做。**

---

## D. 【建议你合并的 bug 修复】`rl/fp8.sbatch` 的评测链会评错目录

```bash
# rl/fp8.sbatch 倒数第 2 行(现状):
E=$(ls "$CK" 2>/dev/null | tail -1)

# rl/cells.sbatch 的正确写法:
E=$(ls "$CK" 2>/dev/null | grep "^epoch" | tail -1)
```
**2026-09-14 补:这不再是推断,是在真实目录里观测到的。**
`13671277` 写完第三个 epoch 后,其 checkpoint 目录内容为:
```
epoch0epochstep28globalstep28
epoch1epochstep28globalstep57
weight_update_v82          <- 字典序排在 epoch* 之后
```
`ls | tail -1` 会选中 `weight_update_v82`。本次未受影响,
因为 `env_local/dose.sbatch` 已加 `grep "^epoch"`;但仓库里的 `rl/fp8.sbatch` 仍是原样。

`fp8.sbatch` 少了 `grep "^epoch"`,会选中 `weight_update_v*` 目录去评测——
正是 HANDOFF §6 里"评测行数值≈2%/空串"那一类故障。tianruny 的 E3 脚本
(`env_local/dose.sbatch`,未改仓库)已修正此处。

**另外一个路径不一致**(不影响正确性,但会让新账号困惑):
`analysis/build_dataset.py` 输出到 `DATA_ROOT/analysis/tokens_{arch}.parquet`,
而 `analysis/paper_zb.py` 读的是 `/home/kzhao2/gap_measurement/results/tokens_{ARCH}.parquet`,
中间有一个未文档化的拷贝步骤。

`analysis/` 下 **17 个脚本硬编码 `/home/kzhao2`**(含 E4 需要的 `key_figure.py`、
`paper_render.py`、`paper_zb.py`),TRANSFER §7 给的 sed 一行可解,但目前仓库里还没改。

---

## E. 【FYI】E3 已由 tianruny 提交,不需要你做

已提交 4 个 job(cell A / Qwen1.5-MoE / sglang,与现有 FP8 点同源):

| JobID | 档位 | 方法 | 说明 |
|---|---|---|---|
| 13664170 | fp8_e4m3 | ours(λ₊=2.3 未重标) | |
| 13664168 | fp8_e4m3 | nocorr | |
| 13664169 | fp8_e4m3 | fullis(exact ratio) | |
| 13664171 | **fp4_e2m1** | nocorr | **探针**,先验证硬件支持再补另外两个 |

**只需要跑 2 档而不是 3–4 档**:档 0(bf16)和档 2(fp8_e5m2)在 SIGMA_TIS /
MASTER_REPORT 里已有 cell A seed 1 的 6 方法完整数据,直接并进曲线即可。

**不需要写代码**:sglang 的 `kv_cache_dtype` 支持
`auto / fp8_e5m2 / fp8_e4m3 / bf16 / fp4_e2m1`,4 档全覆盖;cell A 默认后端
本来就是 `sglang:d4p1t1`(yaml 第 22 行)。

**fp4 的风险**:sglang 帮助文本只说要求 CUDA 12.8+ / PyTorch 2.8+(我们的
torch 2.9.1+cu129 满足),**未说明硬件门槛**;mxfp4 通常需要 Blackwell(sm100),
而集群可用的是 A100(sm80)/ H200(sm90)。所以只投了 1 个探针。

注:tianruny 侧运行期发现的 6 类基建问题(约 50% 静默挂起及其判别法、
disk 同步 I/O 拖垮、重提前需清陈旧 checkpoint 等)已写在
`results/NOTES_tianruny.md` 的 R1–R11,不在此重复。

---

## F. 【需要你回一句话】既有 bf16 / fp8_e5m2 两行的来源,与一个可疑的重复值

(2026-09-14 追加。本节两条都与"E3 能否沿用 2026-08 那批数据"有关。)

### F1. `SIGMA_TIS.md:172` 的 `nocorr` 三种子里,s1 与 s3 **完全相同**

```
| nocorr | 56.10 | 54.44 | 56.10 | 55.55 ± 0.96 | 54.44 |
```
`0.5610 × 1319 = 740.0` —— 两个种子同为 **740/1319**。
贪心解码下不同 seed 得到逐题完全一致的正确数,并非不可能,但请确认**不是误填或复制**。

**为什么要问**:我用这一行做了评测协议差的交叉验证(NOTES R43)——
把主表 `suite_nocorr` 的 61.21 减去实测协议效应 6.03 得 55.18,与此处 55.55 只差 **0.37**。
若 s1/s3 其中之一有误,该残差需重算,进而影响 F2 的判定基线。

### F2. 那两行是用什么 `attention_backend` 跑出来的?

**其中一项不确定性我已自行消除**:两边的**评测协议相同**
(都走 `rl/eval_gsm8k.py`;我在同一个 checkpoint 上实测了新旧协议差 = **+6.03**,
两个 checkpoint 复现、极差 0.07,见 NOTES R33 / R38)。
所以"不可比"的原因已从三项收窄到两项:**硬件** 与 **attention backend**。

E3 本批一律 `attention_backend=triton` —— 这是 Hopper 上跑 fp8/fp4 KV 的必要条件(NOTES R12/R13)。
若 2026-08 那批用的是默认后端(很可能是 fa3 / flashinfer),
则 bf16 档的差异里混着**后端效应**,而不只是 `kv_cache_dtype`。

**不需要你为此做实验**:我已提交 bf16 对照臂 `13676300 / 13676301 / 13676302`
(nocorr / fullis / ours,走与 fp8、fp4 完全相同的 dose 路径,m13h H200 + triton),
预计 2026-09-14 10:00 后落地,届时按 **NOTES R43 预先登记的判读规则**
即可判定既有两行能否并入曲线。
**只需要你方便时回一句当时用的后端**,可以把这个判定从"经验推断"升级为"直接确认"。

### 附:上文 E 节的 job 表已过期

E 节列的 `13664168/69/70/71` 四个 job **均已失败或取消**(A100 在 CUDA-graph 捕获路径上
拒绝 fp8 KV,见 NOTES R15/R17)。E3 现行的九个 arm(bf16 / fp8_e4m3 / fp4_e2m1 各三臂)
及其真实硬件、显存参数、提交史,见 `OWNERSHIP_tianruny.md` 的 **E3 小节**,以该处为准。


---

## G. 对 2026-09-14 修复批次的核对结果(tianruny 已 git pull 并逐项验证)

### G1. `analysis/` `scripts/` `rl/` `src/` `data/` —— 确认已归零 ✓
逐目录 grep `/home/kzhao2`,**命中 0**。评测链修复也已到位:
`rl/seeds.sbatch:61` 与 `rl/fp8.sbatch:53` 均已带 `grep "^epoch"`。

### G2. 但"仓库内硬编码路径归零"这一句**范围说大了**

限定到可执行文件类型(`*.py` `*.sh` `*.sbatch` `*.yaml`,排除注释行)后仍有:

| 目录 | 文件数 | 可执行行数 |
|---|---|---|
| `slurm/` | **13** | **27** |
| `export/two_channel_figure_data/scripts/` | **4** | **6** |

影响面最大的两处:
```
slurm/env.sh:11,19,20        ← 被所有 slurm/*.sbatch source
  export HF_HUB_CACHE=/home/kzhao2/nobackup/autodelete/hf
  source /home/kzhao2/gap_measurement/.venv/bin/activate
  cd /home/kzhao2/gap_measurement

export/two_channel_figure_data/scripts/common.py:17-19
  HF_CACHE / DATA_ROOT / CODE_ROOT 全部指向 /home/kzhao2
```
**为什么要紧**:E1 / E2 / E4 恰好要走这两条路径,而 `/home/kzhao2` 对 tianruny 是 700。
在 tianruny 账号下跑 `slurm/*.sbatch` 会 source 到不可读的 env.sh 而失败。

(注:若用不带 `--include` 的 grep 会得到 94 文件 / 221 行,那是把 `slurm/logs/*.out`
等 81 个非代码文件也算了进去 —— tianruny 第一次就是这么数错的,此处以可执行文件为准。)

### G3. 关于 μ(A 项):结论已收到,tianruny 侧不涉及改动
`μ := E[e^{ε}]` 与仓库里的 `E_k = 1.0000087`(校准恒等式 `E[k]=1`)是两个量,
此前 TODO 的 A 项前提确实错了。E2 的写法待 weightzero 确认 Figure 2 的数据总体后再定,
不影响 E3 与主表。

### G4. 关于 δ dump(C 项):设计无异议,补一个建议
抽样位置 + 两引擎同位置 dump 完整 logit 行,方案合理。
建议**把抽样位置的选取规则也写进 dump 的元数据**(如 seed、步长、是否按 p_t 分层),
否则两个引擎"相同位置"这一前提在事后无法独立核验 —— 这与 tianruny 在 E3 里
反复遇到的"假设覆盖、实则没有"是同一类风险。


---

## H. 【建议并入 C 项】`gen_vllm.py` 同时加一个量化参数,一次改完

你已要为 E1 的 δ dump 改 `src/gen_vllm.py`。**建议在同一次改动里再加一个 KV / logit 量化开关**,
理由是老师 E3 的 P1(零成本诊断扫描)也卡在同一个文件上。

**现状**(已逐行确认,非推测):
```
gen_vllm.py argparse(196-202):--arch / --shard / --num-shards / --smoke / --max-num-seqs
llm_kwargs(229-240):        dtype="bfloat16" 写死,无 kv_cache_dtype / quantization 键
```
因此**无法按档位重跑静态测量**,而老师 E3 要求每档记录 `μ = E[e^ε]`、`E[e^{2ε}]`、`c`、`ξ₊`。

**好消息是下游已经齐备**,只缺这一个入口:
- μ 与 `E[e^{2ε}]`:`analysis/s0_closure.py:125-128` 已实现(`logsumexp(eps)-log N`);
- `c`:`paper_numbers.json` 已有 `c_moe = 0.1553`;
- 数据与模型:`data/prompts_math.jsonl`(2500 条)、MoE 与 dense 各 27G 均已缓存;
- `rl/measure.sbatch` 与 `analysis/` 在你这次修复后**已无 `/home/kzhao2` 硬编码**,tianruny 可直接跑。

**一个可能省很多事的线索,建议改之前先验**:`gen_vllm.py` 用的是 `enforce_eager=True`
(注释:python gate hooks 无法在 CUDA graph 内触发)。而 tianruny 在训练侧遇到的
「A100 拒绝 fp8/fp4 KV」**发生在 CUDA-graph 捕获路径**(NOTES R15/R17)。
**若该限制只在 graph 路径上成立,则诊断扫描可以在 A100 上跑**,
不必占用稀缺的 Hopper —— `measure.sbatch` 本来就指向 dw 分区。

**为什么这件事现在优先级变高了**:E3 的训练部分已全部跑完,而**老师那条核心可证伪预测出现了反向证据**
(fp4 档 CIS 29.84 落后 exact-ratio 32.32,z=−2.26,见 `results/E3_CIS_dose_response.md` §10.3)。
**由于每档的 μ 没测,目前无法确认该反向证据针对的是「μ」还是仅仅「量化档位」** ——
这个参数正是把结论从「带条件」变成「干净」的关键。
