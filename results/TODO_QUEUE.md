# TODO 队列(kzhao2 已取消,交由 tianruny 账号跑)

生成时间:2026-09-09。共 40 个组合。这些 job 在 kzhao2 账号上已 `scancel`,**不会**再产出结果。
tianruny 侧按 `HANDOFF.md` 的预检 + canary 通过后,逐个提交下表组合;每完成一个,在 `results/OWNERSHIP_tianruny.md` 记一行。

提交模板(dsv2 用 `vllm:d4p1t1` + 12h;q30b 用 `vllm:d2p1t2` + 30h;`ours` 臂加 LAMP):
```bash
cd ~/gap_measurement
# dsv2 (LAMP=15.0 只在 METHOD=ours 时加)
sbatch --time=12:00:00 --partition=dw --qos=dw87 --exclude=dw-2-4 \
  --export=ALL,CELL=dsv2,METHOD=<m>,SEED=<s>,TAG=vllm,EXTRA="rollout.backend=vllm:d4p1t1",LAMP=15.0 rl/cells.sbatch
# q30b (LAMP=0.95 只在 METHOD=ours 时加)
sbatch --time=30:00:00 --partition=m13h --qos=gpu \
  --export=ALL,CELL=q30b,METHOD=<m>,SEED=<s>,TAG=vllm,EXTRA="rollout.backend=vllm:d2p1t2",LAMP=0.95 rl/cells.sbatch
```

| # | cell | method | seed | 状态 |
|---|---|---|---|---|
| 1 | dsv2 | fp16 | 1 | TODO |
| 2 | dsv2 | gspo | 1 | TODO |
| 3 | dsv2 | kpopfix | 1 | TODO |
| 4 | dsv2 | ours | 1 | TODO |
| 5 | dsv2 | seqmis | 1 | TODO |
| 6 | dsv2 | seqtis | 1 | TODO |
| 7 | dsv2 | fp16 | 2 | TODO |
| 8 | dsv2 | fullis | 2 | TODO |
| 9 | dsv2 | gspo | 2 | TODO |
| 10 | dsv2 | kpop | 2 | TODO |
| 11 | dsv2 | kpopfix | 2 | TODO |
| 12 | dsv2 | nocorr | 2 | TODO |
| 13 | dsv2 | ours | 2 | TODO |
| 14 | dsv2 | seqmis | 2 | TODO |
| 15 | dsv2 | fp16 | 3 | TODO |
| 16 | dsv2 | gspo | 3 | TODO |
| 17 | dsv2 | icepop | 3 | TODO |
| 18 | dsv2 | kpop | 3 | TODO |
| 19 | dsv2 | kpopfix | 3 | TODO |
| 20 | dsv2 | ours | 3 | TODO |
| 21 | dsv2 | seqmis | 3 | TODO |
| 22 | dsv2 | seqtis | 3 | TODO |
| 23 | q30b | fp16 | 1 | TODO |
| 24 | q30b | gspo | 1 | TODO |
| 25 | q30b | icepop | 1 | TODO |
| 26 | q30b | kpop | 1 | TODO |
| 27 | q30b | kpopfix | 1 | TODO |
| 28 | q30b | ours | 1 | TODO |
| 29 | q30b | seqmis | 1 | TODO |
| 30 | q30b | seqtis | 1 | TODO |
| 31 | q30b | tis | 1 | TODO |
| 32 | q30b | nocorr | 2 | TODO |
| 33 | q30b | fp16 | 3 | TODO |
| 34 | q30b | gspo | 3 | TODO |
| 35 | q30b | icepop | 3 | TODO |
| 36 | q30b | kpop | 3 | TODO |
| 37 | q30b | kpopfix | 3 | TODO |
| 38 | q30b | ours | 3 | TODO |
| 39 | q30b | seqmis | 3 | TODO |
| 40 | q30b | seqtis | 3 | TODO |

## 优先级
1. **dsv2 全部 22 个**(每个 ~8h,信息量最高:该模型 RL 有真实增益空间)。
2. q30b 18 个(每个 ~15h;该模型 GSM8K 起点已 95%,主要用于证明饱和任务下的稳定性)。

## 已完成、不要重复跑的组合(截至 2026-09-09,结果已在 results/eval_suite.tsv)
- dsv2:nocorr s1/s3、fullis s1/s3、tis s2、seqtis s2
- q30b:nocorr s1、fullis s1/s2、tis s2、kpop s2、kpopfix s2、seqmis s2、gspo s2、fp16 s2、ours s2
- kzhao2 侧仍在跑(不要重复):dsv2 tis/icepop/kpop s1;q30b nocorr/fullis/tis s3、icepop s2
