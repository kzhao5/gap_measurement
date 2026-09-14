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
