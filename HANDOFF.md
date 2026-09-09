# 接力手册(给 tianruny 账号上的 Claude Code)

> 目标:在 tianruny 的 BYU 账号上接力跑 **DeepSeek-V2-Lite(dsv2)cell 的 RL 实验**,并把评测结果通过 git 回传。
> 读完本文再读 `TRANSFER.md` §7(环境搭建)和 `SIGMA_TIS.md` 末尾(2026-09-01 起的 13 层修复记录)。

## 0. 一句话背景
论文比较 11 个"训推失配修正算子"(nocorr / fullis / tis / icepop / kpop / kpopfix / seqtis / seqmis / gspo / fp16 / **ours=CIS**),每个模型 × 每个算子 × 3 种子,在 GSM8K 上用 AReaL 做 RL,训完自动在 5 个 benchmark 上评测(math-verify)。三个模型 cell:
- **cell A Qwen1.5-MoE**:已全部完成(kzhao2 账号),不要动。
- **cell D DeepSeek-V2-Lite(dsv2)**:**这是接力的对象**。
- **cell E Qwen3-30B-A3B(q30b)**:留在 kzhao2 账号跑,不要提交。

## 0.5 【2026-09-09 更新】要跑什么:见 `results/TODO_QUEUE.md`
kzhao2 账号已把 **40 个 pending job 全部取消**,交给 tianruny 账号跑。清单(cell/method/seed、提交模板、优先级、已完成不要重复的组合)在 **`results/TODO_QUEUE.md`**。
优先 dsv2 的 22 个(每个约 8h,该模型 RL 有真实增益),其次 q30b 的 18 个(每个约 15h)。
kzhao2 侧只保留 7 个正在跑的 job(TODO_QUEUE.md 末尾列出),跑完即止,不再提交新的。

## 1. 分工与去重规则(最重要)
- tianruny 账号跑 `results/TODO_QUEUE.md` 里的全部 40 个组合(dsv2 22 + q30b 18)。kzhao2 账号不再提交新 job。
- 提交前先 `git pull` 并运行 §5 的清单脚本,**已在 `results/eval_suite.tsv` 出现 `suite_dsv2_<method>vllm_s<seed>` 五行的组合不要再跑**。
- 提交后立即把清单(方法/种子/JobID)写进 `results/OWNERSHIP_tianruny.md` 并 push,kzhao2 侧据此取消对应的重复 job。
- 不改配方、不改 yaml、不改算子参数;TAG 固定为 `vllm`;dsv2 的 CIS 臂 `LAMP=15.0`(按该模型测得 c≈1.2 编译,详见 SIGMA_TIS)。

## 2. 环境预检(全部通过才能提交;任何一项不过先修,修法见 §6)
在 AReaL venv 里逐条验证:
```bash
source ~/gap_measurement/rl/env.sh; cd ~/AReaL && source .venv/bin/activate
python -c "import torch,vllm,sglang,awex,megatron; print(torch.__version__, vllm.__version__, sglang.__version__)"   # 2.9.1+cu129 / 0.16.0 / 0.5.10.post1
python -c "import cv2" 2>&1 | grep -q ModuleNotFound && echo "opencv absent OK" || echo "!! opencv must be uninstalled"
python -c "import importlib.metadata as m; print(m.version('flashinfer-python'), m.version('flashinfer-cubin'))"   # 0.6.3 0.6.3
echo "C_INCLUDE_PATH=$C_INCLUDE_PATH"; ls $C_INCLUDE_PATH/Python.h    # 必须存在(Triton JIT 需要)
grep -c "enable_thinking" areal/experimental/openai/client.py           # 应为 3
grep -n "weight_update_mode" examples/math/gsm8k_icepop.yaml            # 应为 disk
python - <<'PY'
import glob,json; p=glob.glob(__import__('os').path.expanduser('~/nobackup/autodelete/hf/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/*/config.json'))[0]
print('auto_map removed OK' if 'auto_map' not in json.load(open(p)) else '!! remove auto_map from '+p)
PY
sacctmgr show assoc user=$USER format=Account,QOS%40                     # 需要 cs, dw87, gpu
```
缓存必须已存在于 `~/nobackup/autodelete/hf`:模型 deepseek-ai/DeepSeek-V2-Lite-Chat;数据集 openai/gsm8k、HuggingFaceH4/MATH-500、ChilleD/SVAMP、math-ai/minervamath、OlympiadBench(文本子集)。

## 3. Canary(先跑一个 1-epoch 冒烟,通过才放批量)
```bash
cd ~/gap_measurement
sbatch --time=04:00:00 --partition=dw --qos=dw87 --exclude=dw-2-4 \
  --export=ALL,CELL=dsv2,METHOD=nocorr,SEED=1,TAG=smk,SMOKE=1,EXTRA="rollout.backend=vllm:d4p1t1" rl/cells.sbatch
```
日志在 `~/nobackup/autodelete/areal_rl/logs/seeds_<jobid>.out`。判据(全部满足才算通过):
1. 第一个 `task_reward/avg` ≈ **0.77±0.05**(SGLang/坏分词时代是 0.37——若看到 ~0.37 说明 tokenizer 修复没生效);
2. `seq_len/avg` ≈ 250–310;
3. 活过 ≥4 步(每步 3 行 task_reward),即 disk 权重同步循环正常;
4. 结束后日志末尾出现 `Submitted batch job <id>`(评测链已触发),且评测 job 评的是 `epoch*` 目录(不是 `weight_update_v1`)。
通过后 **取消该冒烟的评测行影响**:冒烟 TAG=smk,与正式 trial 目录不冲突,评测行 tag 为 `suite_dsv2_nocorrsmk_s1`,分析时忽略即可。

## 4. 批量提交(22 个)
```bash
cd ~/gap_measurement; DWX="--exclude=dw-2-4"
for m in nocorr fullis tis icepop kpop kpopfix seqtis seqmis gspo fp16 ours; do
  L=""; [ "$m" = "ours" ] && L="LAMP=15.0"
  sbatch --time=12:00:00 --partition=dw --qos=dw87 $DWX --export=ALL,CELL=dsv2,METHOD=$m,SEED=1,TAG=vllm,EXTRA="rollout.backend=vllm:d4p1t1",$L rl/cells.sbatch
  sbatch --time=12:00:00 --partition=m13h --qos=gpu       --export=ALL,CELL=dsv2,METHOD=$m,SEED=3,TAG=vllm,EXTRA="rollout.backend=vllm:d4p1t1",$L rl/cells.sbatch
done
```
分区提示:dw=A100(dw87 可抢占 gstandby)、m13h=H200(最快)、cs/cs2=A100/H100(QOS `cs`,上限 24h;经常维护)。B200 节点 cs-3-1 不能用(fa3 断言),用 `--exclude` 排除。可用 `scontrol update JobId=<id> Partition=<p> QOS=<q>` 在分区间迁移排队 job。每个 dsv2 job 约 6–9h(disk 权重同步),限时用 12h。

## 5. 状态核对脚本(提交前/汇报前都跑)
```bash
cd ~/gap_measurement && git pull -q && python3 - <<'PY'
import subprocess,re,os
methods=["nocorr","fullis","tis","icepop","kpop","kpopfix","seqtis","seqmis","gspo","fp16","ours"]
rows={l.split()[0] for l in open("results/eval_suite.tsv")}
sl=subprocess.run("sacct -u $USER -S 2026-09-01 -X -n --format=JobID,State%10,SubmitLine%300",shell=True,capture_output=True,text=True).stdout
q={(m.group(2),m.group(3)):m.group(1) for m in re.finditer(r"(PENDING|RUNNING)\s.*CELL=dsv2,METHOD=([a-z0-9]+),SEED=(\d)",sl)}
for s in "13":
  for m in methods:
    st="EVALUATED" if f"suite_dsv2_{m}vllm_s{s}" in rows else q.get((m,s),"MISSING")
    print(f"dsv2 {m:8s} s{s}: {st}")
PY
```

## 6. 已知故障签名 → 修法(全部已修入代码;若复现按此排查)
| 日志里的签名 | 原因 | 修法/位置 |
|---|---|---|
| `No module named 'vllm'` | venv 没装 vllm | `uv pip install -r env/requirements-areal.txt` |
| `FATAL FIPS SELFTEST FAILURE` | opencv 自带 libcrypto | `uv pip uninstall opencv-python-headless` |
| `flashinfer-cubin version ... does not match` | 双包版本错配 | 两者都 0.6.3 |
| `custom_all_reduce.cuh ... invalid argument` | 节点 IPC 问题 | yaml `vllm.disable_custom_all_reduce: true`(已在) |
| `No module named 'megatron'`(awex 插件) | awex 硬依赖 | `uv pip install megatron-core==0.16.1` |
| `build_app() got an unexpected keyword argument 'model_config'` | vLLM 0.16 签名 | client 包装已按签名自适应 |
| NCCL `Watchdog caught collective operation timeout` 于 update_weights | xccl 同步挂死 | yaml `weight_update_mode: disk`(已在) |
| `Read timed out (read timeout=600.0)` / `Timeout waiting for key ...update_weights_from_disk` | 落盘重载超时 | RolloutCallback 7200、name_resolve 3600(已在) |
| `gcc ... cuda_utils.c` / `InductorError`(Triton JIT) | 缺 Python.h | env.sh 的 C_INCLUDE_PATH(需 `uv python install 3.11`) |
| 首奖励 ≈0.37、prompt 无空格 | DeepSeek tokenizer 被解析成 slow LlamaTokenizer | hf_utils.load_hf_tokenizer 已走 fast backend |
| `'dict' object has no attribute 'ndim'` | deepseek_v2 不吃 dict mask | fsdp_engine 已加 None-mask 分支 |
| `Could not override 'x'` / `Key 'x' not in struct` | hydra:yaml 无该键 | 用 `++key=value` |
| 评测行数值≈2%/空串 | 评的是 weight_update 目录或塌缩 | chain 已只取 epoch*;塌缩属真实结果,如实记录 |
job 状态 `COMPLETED` 但日志有 `EngineCallError`/无 epoch 目录 = **静默失败**,按失败处理重提。

## 7. 结果回传协议
- 每次有新评测行:`git pull --rebase` → 只提交 **追加到 `results/eval_suite.tsv` 的行** + `results/OWNERSHIP_tianruny.md` + 你的记录文件 `results/NOTES_tianruny.md` → 推到分支 `tianruny-results`(`git push -u origin tianruny-results`)。不要改动别人已有的行、不要改代码文件(代码问题写进 NOTES 由 kzhao2 侧合并修)。
- 评测 tag 格式必须保持 `suite_dsv2_<method>vllm_s<seed>`,与本仓库一致。
- checkpoint 留在你的 scratch(`~/nobackup/autodelete/areal_rl/experiments/checkpoints/$USER/kt-dsv2/`),不进 git。

## 8. 汇报格式(给人看)
每次汇报:① 清单脚本输出(EVALUATED/RUNNING/PENDING/MISSING 计数);② 新出的五 benchmark 行;③ 失败 job 的签名与处置;④ 队列分布。不要在训练奖励上做结论(held-out 才是仲裁,见论文预注册)。
