# OWNERSHIP — tianruny 账号提交的 job

> 策略(2026-09-09 与 kzhao2 商定):**每个 cell×method 先只补一个 seed**,把 11 个算子的
> held-out 覆盖填满;variance(多 seed)留到覆盖完成之后再铺。
> 因此实际提交数远少于 TODO_QUEUE.md 的 40 个——归档掉无效行后,q30b 已有 10/11 个方法
> 各有一个 seed,dsv2 已有 4/11。

生成时间:2026-09-09 23:58:08 MDT

| JobID | cell | method | seed | 分区/QOS | 预期评测 tag |
|---|---|---|---|---|---|
| 13621177 | dsv2 | kpopfix | 1 | dw/dw87 | `suite_dsv2_kpopfixvllm_s1` |
| 13621178 | dsv2 | seqmis | 1 | dw/dw87 | `suite_dsv2_seqmisvllm_s1` |
| 13621179 | dsv2 | gspo | 1 | dw/dw87 | `suite_dsv2_gspovllm_s1` |
| 13621180 | dsv2 | fp16 | 1 | dw/dw87 | `suite_dsv2_fp16vllm_s1` |
| 13621181 | dsv2 | ours | 1 | dw/dw87 | `suite_dsv2_oursvllm_s1` |
| 13621182 | dsv2 | icepop | 3 | m13h/gpu | `suite_dsv2_icepopvllm_s3` |
| 13621183 | dsv2 | kpop | 3 | m13h/gpu | `suite_dsv2_kpopvllm_s3` |
| 13621184 | q30b | seqtis | 1 | m13h/gpu | `suite_q30b_seqtisvllm_s1` |

## 说明
- `ours` 臂用 LAMP=15.0(dsv2)/ 0.95(q30b);其余方法不加 LAMP。TAG 一律 `vllm`。
- dsv2 用 `rollout.backend=vllm:d4p1t1` + 12h;q30b 用 `vllm:d2p1t2` + 30h。
- dsv2 的 icepop / kpop 取 **seed 3** 而非 s1,因为 s1 可能是 kzhao2 侧仍在跑的那个 kt-cells job,
  换 seed 既不重复组合也不留空白;若其 s1 也落地,则该方法自然获得 2 个 seed。
- checkpoint 落在 `~/nobackup/autodelete/areal_rl/experiments/checkpoints/$USER/`,跨账号不冲突。
