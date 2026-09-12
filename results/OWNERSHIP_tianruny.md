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
