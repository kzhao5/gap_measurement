# NOTES — tianruny 账号接力记录

> 本文件只记录 tianruny 侧的诊断与处置。代码层面的修复建议留给 kzhao2 侧合并。
> 仓库代码文件一律未改动;所有本地化产物放在 `~/nobackup/autodelete/env_local/` 与 `~/nobackup/autodelete/wheels/`。

## 2026-09-09 环境搭建(TRANSFER.md §7)——7 个阻塞问题

按 TRANSFER.md §7 在全新账号上搭环境,**7 处会让任何新账号卡住**。前 5 条是文档缺失,后 2 条是
`AReaL/pyproject.toml` 的 `[tool.uv] override-dependencies` 造成的**静默**偏差(最危险,不报错)。

### E1. `uv` 本身没有(文档未提)
集群没有 uv module,PATH 里也没有。需先 `curl -LsSf https://astral.sh/uv/install.sh | sh`(装到 `~/.local/bin`)。
本次使用 uv 0.12.12。**建议**:TRANSFER §7 步骤 3 之前补一步"安装 uv"。

### E2. `env/requirements-areal.txt:18` 是 `-e file:///home/kzhao2/AReaL`
指向 kzhao2 的 home(700 权限),新账号读不了,`uv pip install -r` 直接失败。
**处置**:生成本地化 requirements 时删除该行——它与我们自己的 `uv pip install -e .` 等价,无副作用。
**建议**:freeze 时用 `uv pip freeze --exclude-editable`,或在仓库里提供已剔除该行的版本。

### E3. `env/requirements-areal.txt:95` flash-attn 指向 `/tmp` 里已消失的 wheel
原行:`flash-attn @ file:///tmp/flash_attn-2.8.3+cu128torch2.9-cp311-cp311-linux_x86_64.whl`。
Dao-AILab 官方 v2.8.3 release **没有** torch2.9 的 wheel(只到 torch2.8),所以这个 wheel 是第三方预编译的。
**处置**:从 `mjun0812/flash-attention-prebuild-wheels` v0.4.17 下到**完全同名同版本**的
`flash_attn-2.8.3+cu128torch2.9-cp311-cp311-linux_x86_64.whl`,存到 `~/nobackup/autodelete/wheels/`。
**建议**:把该 wheel 放进仓库可达的位置(或写明上述来源 URL)。flash-attn 是训练必需的——
`areal/api/cli_args.py:1211` 的 `attn_impl` 默认 `flash_attention_2`,`gsm8k_icepop.yaml` 未覆盖。

### E4. `torch/torchvision/torchaudio` 的 `+cu129` local version 在 PyPI 上不存在
加 `--extra-index-url https://download.pytorch.org/whl/cu129` 会让 uv 把**每一个**包(连 `anthropic`)
都去 pytorch 源查一遍,触发限流 → `Failed to fetch .../cu129/anthropic/ ... operation timed out`。
**处置**:直接下载这 3 个 cu129 wheel 到本地,requirements 改成 `file://` 引用,其余全走 PyPI。
**建议**:文档写明用 `--index-strategy` 之外的办法,或提供 wheel 直链。

### E5. freeze 自相矛盾:`outlines==0.1.11` 要求 `outlines-core==0.1.26`,同文件却钉 `outlines-core==0.2.11`
`uv pip install -r` 报 `No solution found ... unsatisfiable`。源环境靠"不做依赖校验"共存
(vllm 走 outlines-core 0.2.x,sglang 钉 outlines 0.1.11)。
**处置**:改用 `uv pip install --no-deps -r <freeze>` 精确复刻,再单独 `--no-deps -e .`。
**建议**:TRANSFER §7 明确写"必须用 `--no-deps` 复刻 freeze",否则任何人都装不上。

### E6.(高危,静默)`megatron-core` 被 `override-dependencies` 整个丢弃
`AReaL/pyproject.toml` 的 `[tool.uv] override-dependencies` 含:
```toml
"megatron-core==0.17.0; python_version >= '3.12' and sys_platform == 'linux' and platform_machine == 'x86_64'",
```
override 会**替换该包的所有需求**。我们跑 Python **3.11**,marker 不成立 → megatron-core 被整个丢掉,
**uv 不报错、不提示**(日志里 "Installed 438 packages",而 requirements 有 439 行)。
后果就是 HANDOFF §6 表里那条 `No module named 'megatron'`(awex 插件 import 时炸)。

**关键陷阱**:§6 给的修法 `uv pip install megatron-core==0.16.1` 如果在 `~/AReaL` 目录下执行
(而 §2 预检恰好要求先 `cd ~/AReaL`),会读到同一份 pyproject、命中同一条 override,
输出 `Checked 1 package`(看起来成功)但**什么都没装**。
**正确修法**:`cd /tmp && uv pip install --no-config --python ~/AReaL/.venv/bin/python --no-deps megatron-core==0.16.1`。
**建议**:§6 那一行改成带 `--no-config` 的版本,并注明"必须在 AReaL 目录之外执行"。

### E7.(静默)同一 override 机制导致 2 个包版本漂移
用 `--no-deps` 复刻后全量核对(用 venv 自身的 `importlib.metadata`,PEP 503 归一化)发现:

| 包 | freeze 要求 | 实际装成 | 来源 override |
|---|---|---|---|
| `openai` | 2.33.0 | **3.11.0** | `"openai>=2.8.0"` → uv 取最新大版本 |
| `nvidia-cudnn-cu12` | 9.10.2.21 | **9.16.0.29** | `"nvidia-cudnn-cu12==9.16.0.29; linux x86_64"` |

`openai` 从 2.x 跳到 3.x 风险最高——rollout 客户端 `areal/experimental/openai/client.py` 走 OpenAI SDK。
**处置**:`cd /tmp && uv pip install --no-config --python <venv>/bin/python --no-deps "openai==2.33.0" "nvidia-cudnn-cu12==9.10.2.21"`。
修正后 AReaL venv 与 freeze 完全一致(439 需求 / 0 缺失 / 0 版本不符,仅多可编辑包 `areal`)。
**建议**:装完必须做一次全量核对,不能只看 `uv pip install` 的退出码。核对脚本见本文件末尾附录。

### 附:两个容易踩的环境细节
- 该账号 shell 预设了 `VIRTUAL_ENV=/home/tianruny/miniconda3`,导致 `uv pip list` 查的是 miniconda
  而不是 venv(会误报"包没装")。核对一律用 `<venv>/bin/python -c "import importlib.metadata"`,
  或显式 `--python <venv>/bin/python`。
- `--no-deps` 复刻出来的 venv 里**本来就没有** opencv(kzhao2 的 freeze 是卸掉 opencv 之后导出的),
  所以 `uv pip uninstall opencv-python-headless` 是空操作。但 `rl/env.sh` 末尾那行注释是对的:
  **任何一次重装 vllm 都会把 opencv 带回来**,这条要当常态检查而非一次性步骤。

### 附录:全量核对脚本(装完必跑)
```bash
$HOME/AReaL/.venv/bin/python - <<'PY'
import re,os,importlib.metadata as md
def norm(s): return re.sub(r'[-_.]+','-',s).lower()
req=os.path.expanduser('~/nobackup/autodelete/env_local/requirements-areal-tianruny.txt')
want={}
for l in open(req):
    l=l.strip()
    if not l or l.startswith('#'): continue
    m=re.match(r'^([A-Za-z0-9_.\-]+)\s*(==|@)\s*(.*)$', l)
    if m: want[norm(m.group(1))]=(m.group(2), m.group(3).strip())
have={norm(d.metadata['Name']): d.version for d in md.distributions() if d.metadata['Name']}
print("缺失:", [k for k in want if k not in have])
print("版本不符:", [(k,v,have[k]) for k,(op,v) in want.items() if op=='==' and k in have and have[k]!=v])
PY
```

## 2026-09-09 缓存模型/数据集(TRANSFER.md §7 步骤 4)——2 个坑

### C1. 登录节点下载时,`OPENSSL_CONF` 与 `HF_HUB_OFFLINE` 必须分开处理
`rl/env.sh` 同时设了两样东西:`OPENSSL_CONF=/dev/null`(绕开 FIPS)和 `HF_HUB_OFFLINE=1`(给计算节点)。
- 若 source env.sh 再下载 → `HF_HUB_OFFLINE=1` 让下载直接失败;
- 若不 source env.sh 就下载 → 丢掉 `OPENSSL_CONF`,FIPS 内核下 Python ssl 直接抛
  `ssl.SSLError: [SSL] error in system default config (_ssl.c:3036)`,连 `HfApi().model_info()` 都跑不了。

**正确姿势**(登录节点):
```bash
export OPENSSL_CONF=/dev/null
export HF_HOME=$HOME/nobackup/autodelete/hf HF_HUB_CACHE=$HOME/nobackup/autodelete/hf
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
```
**建议**:TRANSFER §7 步骤 4 的命令行补上 `OPENSSL_CONF=/dev/null` 并注明要 unset 两个 OFFLINE 变量。

### C2. 训练与评测用的是两个不同版本的 `datasets`,缓存要分别触发
- 训练(AReaL venv):`datasets 4.8.5`,走 `areal/dataset/gsm8k.py` 的
  `load_dataset(path="openai/gsm8k", name="main", split="train"/"test")`;
- 评测(gap venv):`datasets 5.0.1`,走 `rl/eval_suite.py` 的 5 个 benchmark。

两个大版本的缓存布局不保证通用,所以 gsm8k 在**两个 venv 里各触发一次**。
已用 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` 模拟计算节点复验:`rl/eval_suite.py <x> <tag> 4 --dry`
5 个 benchmark 全部命中本地缓存,数量与 eval_suite.py 文档字符串一致
(gsm8k 1319 / math500 500 / svamp 300 / minerva 272 / olympiad 674)。

### 附:tokenizer 修复已确认生效(canary 判据 1 的前提)
离线加载 `load_hf_tokenizer("deepseek-ai/DeepSeek-V2-Lite-Chat")`:
- 类型 `TokenizersBackend`,`is_fast=True`(签名表里"首奖励≈0.37"对应的是被解析成 slow LlamaTokenizer);
- chat template 输出 `'<｜begin▁of▁sentence｜>User: Natalia sold clips to 48 friends.\n\nAssistant:'`
  ——**空格正常**(签名表提到坏分词时 "prompt 无空格");
- 分词 round-trip 正确。

### 附:transformers 5.3 对 DeepSeek config 的 rope 警告(非阻塞)
删掉 auto_map 后走原生 deepseek_v2 实现,加载时有三条警告:
```
`rope_parameters`'s factor field must be a float >= 1, got 40
`rope_parameters`'s beta_fast field must be a float, got 32
`rope_parameters`'s beta_slow field must be a float, got 1
```
是 config 里这几个字段写成了 int 而非 float,transformers 5.3 只告警不报错,tokenizer/模型均正常加载。
先记录,不做修改(改 config 会偏离 kzhao2 的既有 checkpoint 条件)。如 canary 数值异常再回头查这里。

## 2026-09-10 运行期观察

### R1. 新故障签名(不在 HANDOFF §6 表里):`rollout_complete` 回调 30s 超时 → 挂死
**job 13621182(dsv2 icepop s3 @ m13h-2-1)**,05:29 启动,08:31:48 完成 Epoch 2/3 Step 28/29 之后
**再无任何推进**;日志最后写入 09:37:13,到 11:18 已静默 1h41m,最后的活动是
`ProxyRolloutServer` 连续清理 stale session(`Cleaned up 57 stale sessions` / `Cleaned up 11 stale sessions`)。

日志里累计 **12 次**:
```
[RemoteInfEngine Rank 0] ERROR: Callback to http://<ip>:<port>/callback/rollout_complete
failed: HTTPConnectionPool(...): Read timed out. (read timeout=30)
```
时间分布 05:51 / 06:02 / 06:06 / 06:20 / 06:32 / 07:13 ×2 / 07:19 / 07:33 / 07:50 / 08:02 / 08:26,
**频率递增**,最后一次(08:26)之后 5 分钟就停止推进。

**与 §6 已知条目的区别**:§6 里的是 `Read timed out (read timeout=600.0)` 和
`Timeout waiting for key ...update_weights_from_disk`,属**落盘权重重载**路径,已由
RolloutCallback 7200 / name_resolve 3600 修掉。本条是 **rollout_complete 回调**、超时值 **30 秒**,
端点和量级都不同,现有的两个超时加长**覆盖不到它**。

**判定为偶发而非配方缺陷**的依据:同一时间窗、同型号节点(m13h-2-2)上的 13621183(kpop s3)
**一次都没出现**,且已跑到 Epoch 3/3。

**处置**:按"静默失败当失败处理"取消 13621182(当时已烧 5h50m,若放任会占 8 张 H200 到 12h 上限),
原样重提为 **13626596**。checkpoint 只存到 `epoch1...globalstep57`(3 个 epoch 只有 2 个),
符合 HANDOFF 对静默失败的判据。

**给 kzhao2 的建议**:这个 30s 回调超时值得加进 §6 表,并考虑把 `rollout_complete` 回调的
`read timeout` 也调大(与 RolloutCallback 7200 同一量级),或在回调失败时重试而非让 session 变 stale。
识别方法:**日志 mtime 停滞 + `Train step` 长时间不推进**,而非等 job 状态变化——它会一直是 RUNNING。

### R2. 步速实测(用于判断 12h 限时是否够)
| job | 节点 | 稳态步速 | 87 步预估 |
|---|---|---|---|
| 13621183 kpop s3 | m13h-2-2 (H200) | ~4.0 分/步 | ~6h |
| 13621182 icepop s3 | m13h-2-1 (H200) | ~3.2 分/步(卡死前) | — |
| 13621177 kpopfix s1 | dw-1-3 (A100) | ~4.4 分/步 | ~6.4h |

**canary 在 cs-1-1 上测到的 ~9 分/步是异常值**,原因是该节点当时被其他作业占满(mixed 状态)。
正式 job 落在负载较轻的节点上时,A100 与 H200 差距不大(4.4 vs 4.0),**12h 限时余量充足**,
不需要调整 `--time`。选节点时应避开 `mixed` 程度高的节点。

### R3. 第二种挂起模式:训练全部完成后在**收尾阶段**静默挂起(无任何错误签名)
**job 13621183(dsv2 kpop s3 @ m13h-2-2)**,时间线:
```
11:28:50  最后一步(global step 86 = 3 epoch × 29 步)的 ppo update 完成
11:32:42  [RemoteInfEngine] Loading weights from disk done in 231.87s
11:33:42  第 3 个 epoch checkpoint 落盘(epoch2epochstep28globalstep86,30G)
11:35:43  最后的 vLLM engine 日志
之后 41 分钟完全静默;job 仍为 RUNNING,剩余 5h28m
```
**与 R1 的区别**:R1 是训练中途挂死、伴随 12 次 `Read timed out` 和 stale session 清理;
本次 `Read timed out` **0 次**、无 stale session、**无任何 Traceback/错误输出**,
且**三个 epoch checkpoint 全部完整落盘**——训练产物是有效的,卡住的只是 python 进程退出,
导致 `cells.sbatch` 末尾的评测链代码执行不到(`Submitted batch job` 出现 0 次)。

**处置**:取消该 job(训练已完成,再占 8 张 H200 五个半小时无意义),
用与 `cells.sbatch` **逐字相同**的命令手动接上评测:
```bash
CK=$RLROOT/experiments/checkpoints/$USER/kt-dsv2/kpopvllm-s3/default
E=$(ls "$CK" | grep "^epoch" | tail -1)      # → epoch2epochstep28globalstep86
sbatch --export=ALL,EVAL_PATH="$CK/$E",EVAL_TAG="suite_dsv2_kpopvllm_s3" rl/eval_suite.sbatch
```
提交为 13627007。**结果有效性不受影响**:评的是同一个 epoch* 目录,tag 格式不变。

**值得注意的相关性**:R1 与 R3 两次挂起**都在 m13h(H200)节点**上
(m13h-2-1 与 m13h-2-2),而同期 dw(A100)上的 13621177 / 13621178 一直正常推进。
样本量太小,不足以下结论,但**后续排在 m13h 的 job(13626596 icepop s3、13621184 q30b seqtis s1)
需要重点盯**。

**给 kzhao2 的建议**:`cells.sbatch` 的评测链依赖 python 正常退出,这在收尾挂起时会整个丢掉。
建议改成训练结束即写一个 sentinel 文件,或把评测提交挪到 `trap EXIT` 里,
这样即使 python 挂死或 job 被 TIMEOUT 杀掉,已完成的 checkpoint 仍能自动进入评测。

**监控要点**:识别这种情况的判据不是 job 状态(一直是 RUNNING),而是
**日志 mtime 停滞 + checkpoint 已有 3 个 epoch 目录**——满足这两条就可以直接取消并手动接评测。

### R4. 重提 job 前必须清掉上一次留下的陈旧 checkpoint
`examples/math/gsm8k_icepop.yaml` 里 `recover.mode: disabled`,所以**重提的 job 不会 resume**,
而是从头跑、覆盖同名的 `epoch0*`/`epoch1*` 目录。

**陷阱**:如果重提的 job 在中途失败(比如只写完 epoch0),checkpoint 目录里就会是
**新的 epoch0 + 上一次运行遗留的 epoch1**,而 `cells.sbatch` 末尾取的是
`ls "$CK" | grep "^epoch" | tail -1`,会选中那个**陈旧的 epoch1** 去评测,
产出一行数值看似正常、实际来自上一次运行的结果——属于 HANDOFF §6 里"评测行数值异常"那一类,
而且**比塌缩更难发现**,因为数字不会明显离谱。

**处置**:重提之前(job 仍为 PENDING 时)清空该 trial 的 checkpoint 目录:
```bash
CK=$RLROOT/experiments/checkpoints/$USER/kt-<cell>/<method>vllm-s<seed>/default
squeue -j <newjob> -h -o "%T" | grep -q RUNNING || rm -rf "$CK"/epoch*
```
本次对 13626596(dsv2 icepop s3)执行了此清理,删掉了 13621182 留下的
`epoch0epochstep28globalstep28`(07:10)和 `epoch1epochstep28globalstep57`(08:36)。

**给 kzhao2 的建议**:在 `cells.sbatch` 训练开始前加一句清理,或者把评测链改成按
**本次 job 的 globalstep** 选目录而不是 `tail -1`,可以从根上消除这个风险。

### R5.(重要)评测全部失败于 FIPS —— gap venv 被 uv 重新装回了 opencv
**症状**:两个评测 job(13627007、13632168)都在**加载完 5 个数据集之后**立刻崩,
`sacct` 显示 `FAILED / ExitCode 6:0`,日志末尾:
```
crypto/fips/fips.c:154: OpenSSL internal error: FATAL FIPS SELFTEST FAILURE
slurm_script: line 14: ... Aborted (core dumped) OPENSSL_CONF=/dev/null python rl/eval_suite.py ...
```
崩的位置是 `eval_suite.py` 里数据集加载之后紧接着的
`from math_verify ... / from transformers ... / from vllm import LLM`。

**根因(我的疏漏)**:两个 venv 的安装口径不一致。
- AReaL venv 我用了 `--no-deps`(当初为绕开 outlines/outlines-core 的自相矛盾),所以严格按 freeze 装;
- **gap venv 我用了正常依赖解析** `uv pip install -r env/requirements-gap.txt`,
  于是 uv 按 vllm 0.26 的依赖把 **`opencv-python-headless==5.0.0.93` 拉了进来**。

`env/requirements-gap.txt` 里**没有** opencv(kzhao2 的 freeze 是卸掉之后导出的),
所以逐包核对时它表现为"多出 1 个包"——我当时只看了"缺失=0、版本不符=0",
**没看多出项**,漏掉了。

**注意 `OPENSSL_CONF=/dev/null` 拦不住它**:`eval_suite.sbatch` 第 14 行已经带了这个环境变量,
仍然崩溃——opencv 自带的 libcrypto 会做自己的 FIPS 自检,和 OPENSSL_CONF 无关。

**HANDOFF §2 预检覆盖不到这个**:第 2 条 opencv 检查是在 **AReaL venv** 里做的
(`cd ~/AReaL && python -c "import cv2"`),而**评测跑在 gap venv**(`rl/eval_suite.sbatch`
里 `cd $HOME/gap_measurement && source .venv/bin/activate`)。两个 venv 是独立的,
AReaL 干净不代表 gap 干净。

**修法**:
```bash
cd /tmp && uv pip uninstall --no-config --python ~/gap_measurement/.venv/bin/python opencv-python-headless
```
修完两个 venv 均与各自 freeze 逐包一致(缺失 0 / 多出 0 / 版本不符 0)。
两个评测已重提(13633843 kpop s3、13633844 kpopfix s1),立即开始运行。

**给 kzhao2 的建议**(两条,都很便宜):
1. §2 预检的 opencv 检查**要在两个 venv 里各做一次**,建议改成:
   ```bash
   for v in ~/AReaL/.venv ~/gap_measurement/.venv; do
     $v/bin/python -c "import cv2" 2>&1 | grep -q ModuleNotFound \
       && echo "$v opencv absent OK" || echo "!! $v 需卸载 opencv"
   done
   ```
2. TRANSFER §7 应写明 **gap venv 也要用 `--no-deps`**,否则 uv 会按 vllm 的依赖把 opencv 装回来。
   并且核对时**必须同时看"多出项"**,不能只看缺失和版本。

### R6. 排队:m13h 拥堵时可迁到 dw(HANDOFF §4 允许)
13626596(dsv2 icepop s3)与 13621184(q30b seqtis s1)在 m13h 的预计启动是
**2026-09-13 05:18 / 09-12 18:00**(两三天后)。用
`scontrol update JobId=<id> Partition=dw QOS=dw87` 迁到 dw 后,预计启动变成 **当天 19:59**。

依据:dw MaxTime 7 天(容得下 q30b 的 30h);SIGMA_TIS 记录过 **q30b 在 dw/A100 上跑过**
(`q30b kpop s2,dw/A100`,~9 min/step → 87 步 ≈13h);且**迄今两次挂起都在 m13h,dw 零次**,
迁过去对可靠性反而更好。

### R7. 并发过多导致 disk 权重同步被 I/O 拖垮 → 必然超时(需取消重提)
**背景**:`weight_update_mode: disk` 下,每个 job **每一步**都要把约 30GB 权重落盘再由 4 个
推理 rank 重载。5 个 dsv2/q30b job 同时在 dw 上跑时,NFS 带宽成为瓶颈。

**实测(2026-09-10 18:18,5 个 job 并发)**,`Loading weights from disk done in Xs`:

| job | 节点 | 单次重载 |
|---|---|---|
| fp16 s1 | dw-1-2 | **152s** |
| ours s1 | dw-1-1 | **259s** |
| q30b seqtis s1 | dw-1-3 | **408s** |
| **gspo s1** | **dw-2-3** | **1354–1446s** |

对照:单 job 独占时(13621183)是 **231s**。所以并发确实普遍拖慢,但 **gspo 所在的 dw-2-3
严重得多(9 倍)**,而且是**双峰**的——同一个 job 里既有 122–141s 也有 1270–1446s,
从第 13 步起持续落在慢峰。

**后果**:gspo 的步速从 3 分钟/步恶化到 **19–24 分钟/步**。按此推算,剩余 72 步需要约 23 小时,
而 TimeLimit 只到 02:16;即使其他 job 在 22:00 前后结束、争用缓解,它也要到 02:45 才完成,
**仍然会 TIMEOUT**。

**注意:用户无法延长 TimeLimit**——`scontrol update JobId=... TimeLimit=20:00:00` 返回
`Access/permission denied`(只能调小,不能调大)。所以**发现会超时时,唯一的选择是取消重提**,
拖到 TIMEOUT 只会白烧更多机时。

**处置**:取消 13621179(已烧 4h04m,0 个 epoch),重提为 13639468 并
`--exclude=dw-2-4,dw-2-3` 避开慢节点。释放的 dw-2-3 立刻被排队的评测 job 用上。

**识别方法**(比等 TIMEOUT 早得多):
```bash
# 步速异常时,直接看权重重载耗时
grep -oE "Loading weights from disk done in [0-9.]+s" $LOG | tail -4
```
正常 ~130–260s;超过 **1000s** 基本可判定该 job 会超时。

**给 kzhao2 的建议**:
1. 同一分区上**并发的 dsv2/q30b job 不宜超过 3–4 个**,否则 disk 同步互相拖累;
   race 式抢占多分区比在单分区堆满更划算。
2. `cells.sbatch` 可考虑加一个早期自检:若前 5 步的平均重载耗时 × 剩余步数 > 剩余墙钟时间,
   直接主动退出并打印提示,避免白跑到 TIMEOUT。

### R8. 第三次挂起:CIS 臂在 epoch 边界静默停止(dw 节点,推翻"只有 m13h"的猜测)
**job 13621181(dsv2 ours/CIS s1 @ dw-1-1)**,时间线:
```
18:12:53  权重重载完成(258-259s,正常量级)
18:14:45  epoch0 checkpoint 落盘(30G,完整)
18:17:33  vLLM engine 最后一条日志(还在生成)
18:17:53  最后一次文件写入
之后 42 分钟完全静止;job 仍为 RUNNING,剩余 8h33m
```
**故障签名全为 0**:Read timed out 0、stale session 0、EngineCallError 0、Traceback 0。

**决定性判据不是日志静默,而是"是否还在写文件"**(日志静默会被 R7 的慢重载混淆——
单次重载可达 23 分钟):
```bash
# 挂死的 job:18:15 之后零写入
find $EXPROOT -path "*oursvllm-s1*" -newermt "18:15" -printf "%TH:%TM  %p\n"
# 正常的 job:持续在写 weight_update_vN/model.safetensors
find $EXPROOT -path "*fp16vllm-s1*" -newermt "18:50" -printf "%TH:%TM  %p\n"
# 且 name_resolve/<trial>/update_weights_from_disk/<N> 会不断出现新序号
```
本次对照结果:CIS job 18:17:53 之后零写入、name_resolve 无任何条目;
同期 fp16 正在写 `weight_update_v40/model.safetensors`(19:00:03),
icepop / q30b 的 name_resolve 也都有新条目。三个兄弟 job 全部正常,只有它停了。

**修正之前的观察**:R1、R3 两次挂起都在 m13h,我当时记的"两次挂起都在 m13h,dw 零次"
**已被本次推翻**——本次在 **dw-1-1**。所以挂起**与分区无关**,是 AReaL + vLLM + disk 权重同步
这条链路本身的偶发问题。

**发生率值得注意**:到目前为止 8 个 job 里出现 3 次挂起(13621182 训练中、13621183 收尾、
13621181 epoch 边界),约 37%。三次的共同点是:**无任何错误输出、job 状态一直是 RUNNING**,
只能靠外部探测发现。

**处置**:取消(已烧 3h27m,1/3 epoch),按 R4 清掉 `epoch0*` 与 `weight_update_*`
(否则重提后若中途失败会评到陈旧 checkpoint),重提为 13639641,LAMP=15.0 已核对。

**给 kzhao2 的建议**:这条链路需要一个**看门狗**。最简单的实现是在 trainer 侧每步更新一个
心跳文件,外部脚本发现心跳超过 N 分钟未更新就自动 scancel + 重提;
只靠 Slurm 状态和日志都发现不了(状态恒为 RUNNING,日志静默与慢 I/O 无法区分)。

### R9. 第四次挂起(fp16 s1,又是收尾形态)—— 挂起率已达 8 个 job 中 4 次
**job 13621180(dsv2 fp16 s1 @ dw-1-2)**:
```
18:06:06  epoch0 落盘
20:35:23  epoch1 落盘
23:09:59  epoch2 落盘(globalstep86,与成功完成的 job 完全一致)
23:12:44  最后一次日志写入
之后 45 分钟零写入;job 仍 RUNNING,剩余 3h09m;评测链触发 0 次;故障签名 0
```
训练奖励 0.772 → 0.825,86 步,产物完整有效。

**处置**:取消 + 手动接评测(13643164,评 `epoch2epochstep28globalstep86`,
tag `suite_dsv2_fp16vllm_s1`)。释放的 dw-1-2 立即被排队的 CIS 重提 job 接管。

**汇总:到此 8 个 job 出现 4 次挂起(50%)**,分两类:

| 类型 | 例子 | checkpoint | 处置 | 损失 |
|---|---|---|---|---|
| **训练中挂起** | 13621182(中途)、13621181(epoch 边界) | 不足 3 个 | 清残留 + 重提 | 整个 job 的已用机时 |
| **收尾挂起** | 13621183、13621180 | **3 个齐全,有效** | 取消 + 手动 `sbatch eval_suite.sbatch` | 仅自动化链条,**结果不受影响** |

**关键区分方法**(两者的日志表现完全一样,都是静默 + 无错误):
```bash
ls $CK | grep -c "^epoch"   # =3 → 收尾挂起,手动接评测即可;<3 → 真挂,需重提
```

**这条提高了看门狗的优先级**:按 50% 的发生率,不做自动化的话平均每个 job 都要人工介入一次,
而且"收尾挂起"若没被发现,一个已经跑完 9 小时、结果完全有效的 job 会白白作废。
建议的最小实现:训练结束写 sentinel 文件,外部脚本轮询
「sentinel 存在 或 epoch 目录=3」且「评测链未触发」→ 自动 scancel + 补提评测。

## 2026-09-11 收尾:8 个组合全部完成,两个 cell 均 11/11

### 最终产出
dsv2 与 q30b 的 11 个算子**全部有 held-out 结果**。我提交的 8 个组合最终都产出了有效数据:
dsv2 的 kpopfix s1 / seqmis s1 / gspo s1 / fp16 s1 / kpop s3 / icepop s3 / ours(CIS) s1,
以及 q30b 的 seqtis s1。

### R10. 挂起最终统计:8 个组合、13 次提交、6 次挂起
| 组合 | 提交次数 | 挂起 | 说明 |
|---|---|---|---|
| dsv2 kpopfix s1 | 1 | 0 | 一次跑通 |
| dsv2 seqmis s1 | 1 | 0 | 一次跑通 |
| dsv2 icepop s3 | 2 | 1(训练中,R1) | 12× rollout_complete 30s 超时后挂死 |
| dsv2 kpop s3 | 1 | 1(收尾,R3) | 训练有效,手动接评测 |
| dsv2 fp16 s1 | 1 | 1(收尾,R9) | 训练有效,手动接评测 |
| dsv2 gspo s1 | 2 | 0 | 首次非挂起,是 I/O 拖垮必然超时(R7),主动取消 |
| dsv2 ours(CIS) s1 | 2 | 1(epoch 边界,R8) | 重提后 7h19m 跑通 |
| **q30b seqtis s1** | **3** | **2(均在 epoch 边界)** | 前两次共烧 12h14m 零产出,第三次 **16h00m** 跑通 |

**挂起位置高度集中在"刚写完 epoch checkpoint 之后"**:R8(CIS,epoch0 后)、
q30b 两次(epoch1 后、epoch0 后)。q30b 的 checkpoint 是 57GB(dsv2 的近两倍),
这个窗口更长,可能是它 2/3 次都中招的原因。

**q30b seqtis 的真实成本**:7h52m + 4h22m + 16h00m = **28h14m 的 8 卡机时**,
产出一行结果。若有看门狗自动重提,前两次的 12h14m 可以省下大半。

### R11. q30b 第三次为何跑了 16 小时(而非预估的 9.4)
第三次(13643927)`COMPLETED`,Elapsed **16:00:35**,无任何故障签名,评测链正常触发。
比按前期步速外推的 9.4 小时长了约 70%。原因是后期 dw 上其他用户的负载上来,
disk 权重同步变慢——这与 R7 是同一机制,只是没有严重到会超时(30h 限时留了足够余量)。
**教训**:q30b 这类 87 步 × 57GB 同步的 job,墙钟时间对 I/O 争用极其敏感,
30h 限时是必要的,不能按空闲时的步速去压缩。

## 2026-09-13 E3 dose–response:两条 sglang 后端约束(新签名)

### R12. `fp8_e4m3` 在 Ampere 上必须显式指定 `attention_backend=triton`
提交 `+sglang.kv_cache_dtype=fp8_e4m3` 后,job 在 **CUDA graph 捕获**阶段失败
(`FAILED`,ExitCode 1:0,约 12 分钟):
```
Scheduler hit an exception: Traceback ...
  sglang/srt/model_executor/cuda_graph_runner.py:1024 capture_one_batch_size
  sglang/srt/models/qwen2_moe.py:766 forward
RuntimeError: FlashAttention on Ampere/Ada cards only supports fp16 and bf16 data type
```

**根因**在 sglang `ServerArgs` 里:
```python
if self.attention_backend == "fa3" and self.kv_cache_dtype == "fp8_e5m2":
    logger.warning("FlashAttention3 only supports fp8_e4m3 if using FP8; "
                   "Setting attention backend to triton.")
    self.attention_backend = "triton"
```
它**只对 `fp8_e5m2` 自动切换到 triton,对 `fp8_e4m3` 不切**;而 yaml 默认
`attention_backend: fa3`(`SGLangConfig` 第 2067 行),于是 e4m3 保持 fa3 → 撞 Ampere 限制。

**重要推论(对 §5.4 的可比性有利)**:原有的 fp8_e5m2 实验(`rl/fp8.sbatch`)并没有设置
`attention_backend`,它能跑通正是因为**被 sglang 静默切成了 triton**。所以新增档位显式使用
triton **与既有数据点口径一致**,不构成后端混淆。
(仍需注意:bf16 基线档用的是 fa3,与 fp8 各档的后端不同——这是既有数据自带的差异。)

### R13. `fp4_e2m1` 同样只是后端问题,**不是硬件限制**
fp4 探针同样 `FAILED`(10:53),根异常:
```
AssertionError: KV4 MHA expects attention_backend to be one of
['triton', 'torch_native', 'flex_attention', 'trtllm_mha'], but got fa3
```
之前担心的"mxfp4 需要 Blackwell(sm100)"**不成立**——sglang 自己的断言把 `triton` 列为合法选择,
A100(sm80)可用。所以 **4 档(bf16 / fp8_e4m3 / fp8_e5m2 / fp4_e2m1)在现有硬件上全部可跑**,
唯一要求是显式 `++sglang.attention_backend=triton`。

**修法**(已加入 `env_local/dose.sbatch`,未改仓库):
```bash
+sglang.kv_cache_dtype=${KVDTYPE} \
++sglang.attention_backend=${ATTN:-triton} \
```
参考:`rl/cells.sbatch` 对 dsv2 早就有 `++sglang.attention_backend=triton`,同一条经验。

### R14.(操作陷阱)`env_local/big_files_env.sh` 会覆盖调用方的变量 `S`
该脚本里有 `S=$HOME/nobackup/autodelete`(未 export,但 `source` 会污染当前 shell)。
若调用方也用 `S` 存 sbatch 路径,`source` 之后 `sbatch "$S"` 会指向一个目录,
报 `sbatch: error: Batch script is empty!`。**排查时容易误以为脚本被写坏**。
建议把该文件里的局部变量改名(如 `_KTS`)或加 `local`/`unset`。

### R12/R13 更正(2026-09-13):`attention_backend=triton` **不足以**在 Ampere 上跑 fp8 KV
上一节我写"显式设 triton 即可,4 档在 A100 全部可跑"——**这一点被后续实验证伪,特此更正**,
以免按原结论去排实验。

带 `++sglang.attention_backend=triton` 重提后(job 13671215),`config.yaml` 里确认
`attention_backend: triton` 与 `kv_cache_dtype: fp8_e4m3` **都已生效**,但仍然失败:
```
Exception: Capture cuda graph failed: FlashAttention on Ampere/Ada cards only
                                      supports fp16 and bf16 data type
```

**为什么后端开关管不了它**:
- 该错误串在 `site-packages` 里(含 `.so` 扫描)**完全搜不到**,`triton_backend.py` 内也没有
  任何 fp8 相关代码;
- traceback 的末端是 `torch/_ops.py:841 __call__`,`cuda_graph_runner.py:655` 只是把
  `self.capture()` 抛出的 `RuntimeError` 包装成 "Capture cuda graph failed";
- 说明这是**编译扩展(FA3 / sgl-kernel)在内核层面拒绝 fp8 张量**,与 Python 层选哪个
  attention backend 无关。`get_attention_backends()` 确实会把 `attention_backend` 传播到
  prefill/decode 两侧,所以不是传播漏掉的问题。

**仍然成立的部分**:R12 里"sglang 只对 `fp8_e5m2` 自动把 fa3 切成 triton、对 `fp8_e4m3` 不切"
这条源码事实无误;R13 里 fp4 的 `KV4 MHA expects attention_backend to be one of [...]` 也确实
是后端断言。只是**修掉后端之后还有第二道内核层门槛**。

**正在验证的两条路**(各投一个探针,均为 nocorr / fp8_e4m3):
1. `13671239` A100 + triton + `++sglang.disable_cuda_graph=true` —— 错误发生在 graph 捕获,
   跳过捕获是否能绕开;
2. `13671240` **H200(sm90)** + triton —— 错误信息字面只点名 "Ampere/Ada",
   Hopper 上 FA3 支持 fp8,大概率可用。

**若确认是硬件门槛,对 E3 的影响**:KV-cache 路线必须在 **Hopper(m13h H200 / cs2 H100)**上跑,
不能用 dw 的 A100。这同时带出一个**必须向 kzhao2 确认的可比性问题**:
既有的 §5.4 `fp8_e5m2` 数据点当初跑在什么硬件上?若它在 Hopper 上,新档位也应在 Hopper;
若它在 A100 上,那它当时能跑通的机制需要重新解释(可能是 e5m2 被自动切 triton 后走了不同内核)。

### R15. `disable_cuda_graph=true` 确实绕过了 Ampere 的内核门槛(但随后 OOM)
承接上一节的更正。第三轮探针 `13671239`(A100 + `attention_backend=triton`
+ `++sglang.disable_cuda_graph=true`)结果:

- **`FlashAttention on Ampere/Ada` 错误出现 0 次** —— 内核层门槛确实被绕过了,
  说明该拒绝只发生在 **CUDA graph 捕获路径**上,并非 fp8 KV 在 sm80 上完全不可用;
- 但 job 仍 `FAILED`(10:58),死因变成
  `Inference server process exited with code -9 before becoming healthy`
  —— **-9 = SIGKILL,典型的 OOM 被杀**,而非断言失败;
- 当时 `mem_fraction_static: 0.8`(yaml 默认)。关掉 CUDA graph 会失去 graph 复用带来的
  显存与调度优化,0.8 的静态占比不再合适。

**处置**:重提 `13671272`,同配置但 `++sglang.mem_fraction_static=0.6`。

**方法论要点**:失败信息从"断言拒绝"变成"被 SIGKILL",是**路线可行性的正向信号**,
不应与前两轮的失败等同看待。判断一次重试是否有进展,要看**失败模式是否改变**,
而不只看是否仍为 FAILED。

**同时在验证的另一条路**:`13671240`(**H200 / sm90** + triton,CUDA graph 保持开启)
—— 目前零错误,已推进到 `RolloutController INFO: Proxy servers initialized`,
比任何一次 A100 尝试都远。**尚未出第一个训练步,结论待定**,不在此下断言。

两条路若都可行,优先用 A100(dw 分区容量远大于 m13h,E3 需要 9+ 个 run);
若仅 H200 可行,则 E3 的 KV-cache 路线必须排在 m13h,吞吐会成为瓶颈。

### R16. ✅ E3 跑通:**H200(sm90)+ `attention_backend=triton`**,CUDA graph 保持开启
`13671240`(cell A / Qwen1.5-MoE / `kv_cache_dtype=fp8_e4m3`)在 m13h-1-2 上
**`Train step 1/87 done`**,首步 `task_reward/avg=0.6982`、`seq_len/avg=298.2`,
零错误。这是 E3 第一次真正进入训练。

**三条路的最终对照**(全部 nocorr / fp8_e4m3):

| Job | 硬件 | 配置 | 结果 |
|---|---|---|---|
| 13664168 | A100 | 默认 fa3 | ✗ `FlashAttention on Ampere/Ada only supports fp16/bf16` |
| 13671215 | A100 | triton | ✗ 同上(triton 已生效但内核层仍拒绝) |
| 13671239 | A100 | triton + 关 cuda graph(mem 0.8) | ✗ 门槛已过(该错 0 次)但 `code -9` OOM |
| 13671272 | A100 | triton + 关 cuda graph(mem 0.6) | 验证中 |
| **13671240** | **H200** | **triton(graph 开)** | **✅ 训练正常** |

**结论**:fp8 KV 的拒绝只发生在 **Ampere 的 CUDA graph 捕获路径**上;Hopper 无此限制,
且无需关闭 CUDA graph、无需降显存,是最干净的路线。

**据此的决定**:新增的 dose 档位**全部排在 m13h(H200)**,不与 A100 混用。
理由是跨档位可比性——dose-response 曲线要求只有 `kv_cache_dtype` 一个变量在变,
若各档跑在不同硬件/不同 graph 设置上,曲线就不可解释。
代价是吞吐:m13h 只有 4 个节点(dw 有 7),E3 的 9 个 run 会排得比较久。

已提交(均 H200 + triton):`fullis`/`ours` 的 fp8_e4m3 两臂,以及 fp4_e2m1 的 nocorr 探针。

**遗留的可比性问题(需 kzhao2 回答)**:既有的 bf16 基线档与 §5.4 的 fp8_e5m2 档
当初跑在什么硬件、什么 attention backend 上?若与 H200+triton 不一致,
这条 dose 曲线的前两个点就不能直接并入。

### 附:一个读数陷阱
用 `grep` 跨该 trial 目录下的多个日志文件(`main.log` / `merged.log` / `rollout.log`)
取指标时,同一行会被重复计入,表现为"连续几步数值完全相同"。
**取训练指标只应从 `main.log` 单读**。我第一次读 H200 的结果时就被这个骗过,
误以为是 3 步、且怀疑训练卡住。

### R17. A100 路线放弃:降显存无效,且即使跑通也不可用于同一条曲线
承接 R15。`13671272`(A100 + triton + `disable_cuda_graph=true`
+ `++sglang.mem_fraction_static=0.6`)结果:**仍然 `code -9` OOM**,
`FAILED` 于 **10:04**,完成 0 步;`FlashAttention on Ampere` 仍为 0 次(门槛确实已过)。

两次对照:

| mem_fraction_static | 结果 | 用时 |
|---|---|---|
| 0.8(yaml 默认) | code -9 OOM | 10:58 |
| **0.6** | **code -9 OOM** | **10:04(更早)** |

**降显存没有改善、反而更快失败**,说明瓶颈不是 GPU 静态占比——更可能是主机内存,
或关闭 CUDA graph 后内存压力转移到了别处。继续二分 mem_fraction 属于盲试。

**决定:放弃 A100,E3 全部在 H200 上跑。** 除了上面的失败,还有一个更根本的理由:
A100 必须带 `disable_cuda_graph=true` 才能过内核门槛,而 H200 各档不需要。
若混用,dose 曲线上不同档位的**推理配置本身就不一致**,而这条曲线的前提恰恰是
"只有 `kv_cache_dtype` 一个变量在变"。所以 A100 从一开始就只在"全部档位都能跑通"
时才有价值,这个前提现在不成立。

**代价**:m13h 只有 4 个节点(dw 有 7),E3 剩余的 run 会排得较久,这是已知且可接受的。

**可复用的判据**:排查一条新硬件/新精度路线时,除了"能不能跑通",还要问
**"跑通所需的变通是否会污染要做的对比"**。本例中关闭 CUDA graph 就是这样一个变通——
它能解决报错,却会让该档位与其他档位不可比。

### R18. 把 E3 的排队范围从 m13h 扩到 Hopper 全体(含 H100),但**不**回到 A100
只排 m13h 时,三个待跑 job 的预计启动是 **12–23 小时之后**
(fullis 2026-09-14 01:31、ours 10:47、fp4 探针 12:56),而 fp4 探针还门控着另外两臂,
按此推算 fp4 那一档要到两天后才有结果。

**可用的 Hopper 通道**(均 sm90,与 H200 同代):

| 分区 | 硬件 | 节点数 | 我可用的 QOS | 可抢占 |
|---|---|---|---|---|
| m13h | H200 | 4 | `gpu` | 否 |
| **cs2** | **H100** | **2** | **`cs`** | **否** |
| eng | H200 | 1 | `standby` | **是** |

**决定**:把最晚的两个(fp4 探针 13671278、ours 13671277)迁到 **cs2 / QOS=cs**,
fullis 留在 m13h。**不使用 eng**,因为那里只能用 `standby`,可被抢占——
一个 6 小时的 run 被抢占等于全损,而本项目已经在挂起/重提上损失了大量机时。

**为什么这不同于被否掉的 A100**:A100 必须加 `disable_cuda_graph=true` 才能过内核门槛,
那是**推理配置上的差异**,会让该档位与其他档位不可比。H100 与 H200 同为 sm90,
走同一套内核、**用完全相同的配置**(triton + CUDA graph 开启),不引入任何配置差异。
两者的区别在显存容量与带宽,影响的是**速度而非数值**,而这条曲线比较的是 held-out 精度。

**仍需记录**:若最终有 arm 落在 H100、有 arm 落在 H200,会在结果里**逐 arm 标注实际硬件**,
由 kzhao2 判断是否需要在 caption 中说明。我不擅自认定两者完全等价。

**操作提醒**:`scontrol update Partition=` 会把 TimeLimit 重置为新分区的默认值
(本次又一次重置为 3 天),迁移后必须重新 `scontrol update TimeLimit=12:00:00`。
另:单个 job 只能带一个 QOS,而 m13h 不接受 `cs`、cs2 不接受 `gpu`,
**因此无法让一个 job 在 m13h 与 cs2 之间浮动**——只能逐个指定。

### R19. E3 评测链的前提核查(含一个待验的 tokenizer 隐患)
E3 走的是 `eval_tp.sbatch → rl/eval_gsm8k.py`,**与主表的 `rl/eval_suite.py` 是两条不同的评测路径**,
因此主表上做过的修复未必覆盖到它。逐项核查:

| 前提 | 结果 |
|---|---|
| gap venv 离线加载 `openai/gsm8k`(main/test) | ✅ n=1319 |
| gap venv 已卸 opencv(否则 FIPS 崩,见 R5) | ✅ |
| 评测资源 4×A100 / bf16 推理(不涉及 fp8) | ✅ dw 有空闲节点 |
| **cell A checkpoint tokenizer 是否可用** | ⏳ 待验 |

**待验的隐患**:`eval_gsm8k.py` 第 14 行是
```python
tok = AutoTokenizer.from_pretrained(path)     # path = checkpoint
```
即**直接用 checkpoint 自带的 tokenizer**;而 `eval_suite.py` 为 SIGMA_TIS ⑭ 那个 bug
(AReaL 保存的 checkpoint tokenizer 在 eval venv 下 decode 出 `Ġ`、分词不同 → 评测塌到 20%)
改成了一律用 base 模型快照的 tokenizer。

**但 `eval_suite.py` 的修复只覆盖两个 cell**:
```python
_base = {"kt-dsv2": "deepseek-ai--DeepSeek-V2-Lite-Chat", "kt-q30b": "Qwen--Qwen3-30B-A3B"}
```
**没有 cell A / Qwen1.5-MoE**,说明当初判断该 bug 是 DeepSeek 特有的。旁证是 cell A 原有的
33 个 run 本来就用 `eval_gsm8k.py` 评,结果正常(nocorr 55.55 等),没有 20% 那种塌陷。

**为什么仍要验**:cell A 的历史结果产生于更早的环境,而本账号的 gap venv 是
transformers 5.14.1 重建的。`eval_suite.py` 注释里写的正是 "byte-level BPE wrongly under
transformers 5.x",**触发条件是 transformers 版本而非模型**。所以必须拿本次 job 真实存下的
checkpoint 去比对,不能用历史结果推断。

已挂后台任务:`13671240` 的第一个 epoch 落盘后自动比对 base 与 checkpoint tokenizer 的
`tokenizer_class`、词表大小、编码 id、解码文本(查 `Ġ` 伪影)与 chat template。
若不一致,则在提交评测前把 `eval_gsm8k.py` 的 tokenizer 改为 base 快照
(与 `eval_suite.py` 同样的做法),否则 E3 四个档位会产出一批约 20% 的废数字。

**给 kzhao2 的建议**:`eval_suite.py` 的 `_base` 映射应补上 `kt-dose`/`kt-fp8` 等使用
Qwen1.5-MoE 的实验,或者干脆把 `eval_gsm8k.py` 也改成同一套 base-tokenizer 逻辑,
避免两条评测路径的修复状态不一致。

### R19 结论:cell A 的 tokenizer 隐患**已排除,无需修改 `eval_gsm8k.py`**
`13671240` 的第一个 epoch 落盘后(`epoch0epochstep28globalstep28`,15:49)实测比对:

| 项 | base 快照 | checkpoint |
|---|---|---|
| tokenizer_class | `Qwen2Tokenizer` | `Qwen2Tokenizer` |
| 词表大小 | 151646 | 151646 |
| 编码 id | — | **一致** |
| 解码文本 | — | 正常,**无 `Ġ` 伪影** |
| chat template | — | **一致** |

**结论**:SIGMA_TIS ⑭ 那个 bug **确实是 DeepSeek 特有的**,不影响 Qwen1.5-MoE。
`eval_gsm8k.py` 第 14 行直接用 checkpoint tokenizer 是安全的,E3 的评测链**不需要改动**。
这也解释了为什么 `eval_suite.py` 的 `_base` 映射当初只列 `kt-dsv2` 与 `kt-q30b`——
判断是对的,不是遗漏。

**因此撤回 R19 里给 kzhao2 的那条建议**(不必把 `_base` 映射扩到 Qwen1.5-MoE 实验,
也不必改 `eval_gsm8k.py`)。当时以为触发条件是 transformers 5.x 版本而非模型,
实测表明模型侧的 byte-level BPE 实现才是关键:Qwen2Tokenizer 在 transformers 5.14.1 下
往返正常,DeepSeek 的则不然。

### R20. 读日志的分工:训练指标与引擎指标在不同文件
做健康检查时我用 `main.log` 一把抓,结果权重重载耗时样本数为 0,**等于静默失去了 R7 那道
超时预警**。实际分工是:

| 指标 | 所在文件 |
|---|---|
| `task_reward/avg`、`seq_len/avg`、`Train step N/M done` | **`main.log`** |
| `Loading weights from disk done in Xs` | **`rollout.log`**(与 `merged.log`) |

与 R16 附录那条(跨文件 grep 会重复计数)合起来才完整:**该分文件读的不能合并,
该单文件读的不能跨文件**。`13671240` 用正确文件复测的结果:重载中位 **260s**、最慢 283s,
远低于 1000s 预警线;步速 4.4 分钟/步无劣化,87 步约 6.6 小时,12h 限时余量充足。

### R21. E3 第一个数据点:`fp8_e4m3 / nocorr = 62.17`,但**与既有 bf16 行不可比**
`13671240`(cell A / Qwen1.5-MoE / H200 / triton / CUDA graph 开)训练 `COMPLETED` 6:36:52,
3/3 epoch、零故障签名,评测 `13673240` 用时 9:15:

```
RESULT dose_fp8_e4m3_nocorr acc=0.6217 (820/1319)
训练终值 reward=0.8398(step 87),seq_len 241.7(从首步 298.2 收敛)
```

填入曲线后的现状:

| 档位 | nocorr | fullis | ours | 来源 |
|---|---|---|---|---|
| bf16 | 55.55 | 59.72 | 64.29 | 既有,硬件/后端未知 |
| **fp8_e4m3** | **62.17** | — | — | **本次 H200+triton** |
| fp8_e5m2 | 54.36 | 57.62 | 60.12 | 既有,硬件/后端未知 |
| fp4_e2m1 | — | — | — | 未跑 |

**必须正视的异常**:`nocorr` 在 fp8_e4m3 下(62.17)比在 **bf16 下(55.55)高 6.6 分**。
KV 量化只会增加噪声,不该让 held-out 变好。**这不是"量化提升了性能",而是既有 bf16 行
与本次结果不可比的直接证据。**

相对 e5m2 的方向倒是自洽的:e4m3 是 4 位指数/3 位尾数,e5m2 是 5 位指数/2 位尾数,
**e4m3 尾数精度更高、量化噪声更小**,故 62.17 > 54.36 符合预期。

**分歧来源(至少三重,均非单一 KV dtype 效应)**:
1. **硬件**:本次 H200(sm90),既有两档硬件未知(若为 A100 则差异更大);
2. **attention backend**:本次显式 `triton`,既有 bf16 档大概率是默认 `fa3`
   (sglang 只对 `fp8_e5m2` 自动切 triton,见 R12);
3. **训练轨迹本身**:rollout 引擎配置不同 → 采样不同 → 训练数据不同。
   这一条最容易被忽略——**它不是评测差异,而是两次训练本就不是同一个 run**。

**结论与处置**:在 kzhao2 确认既有两档的硬件与后端之前,**这条 dose 曲线只能用本次自跑的档位
(fp8_e4m3 / fp4_e2m1)内部比较**,既有 bf16 / fp8_e5m2 两行不得并入。
若最终无法对齐,正确做法是**把 bf16 基线也在 H200+triton 下重跑一遍**(3 个臂,约 20 小时),
让整条曲线的四个档位来自同一套配置——这才是 E3"只有 kv_cache_dtype 一个变量在变"的前提。

**对 E3 结论的影响**:老师要的是"把 μ 推大、看 CIS 相对基线的优势如何变化",
这是**同一档位内不同算子之间**的比较(nocorr vs fullis vs ours),
**不依赖跨档位的绝对值对齐**。所以即使既有两档并不进来,只要本次自跑的档位把三个臂都补齐,
E3 的核心论证(Theorem 4.6 的指数 2→1)仍然成立。这一点是好消息。

### R22. 各 arm 的实际硬件(目录名不可信,以 `sacct NodeList` 为准)
提交时统一用了 `TAGSFX=-h200`,但 `ours` 与 fp4 探针后来被迁到 cs2,**实际跑在 H100 上**,
目录名中的 `-h200` 与事实不符。固化如下(趁 sacct 记录仍在):

| JobID | 档位 | 方法 | 实际节点 | 硬件 | trial 目录名(⚠ 误导) |
|---|---|---|---|---|---|
| 13671240 | fp8_e4m3 | nocorr | m13h-1-2 | **H200** | `nocorr-fp8_e4m3-h200` |
| 13671276 | fp8_e4m3 | fullis | m13h-1-2 | **H200** | `fullis-fp8_e4m3-h200` |
| 13671277 | fp8_e4m3 | **ours** | **cs-2-1** | **H100** | `ours-fp8_e4m3-h200` ← 名不符实 |
| 13671278 | fp4_e2m1 | nocorr | 待分配 | cs2 → 预计 H100 | `nocorr-fp4_e2m1-h200` |

**影响评估**:
- **结果行标识不受影响** —— `EVAL_TAG="dose_${KVDTYPE}_${METHOD}"` 不含 `TAGSFX`,
  故 `eval_suite.tsv` / `RESULT` 行里的 tag 是干净的;
- **受影响的是 checkpoint 与日志目录名**,读者会据此误判硬件。

**处置**:`env_local/dose_curve.sh` 已改为**每次汇总时从 `sacct NodeList` 现取实际节点并映射硬件**,
不再依赖目录名。硬件信息有时效性——job 被清出 sacct 后 NodeList 就查不到,所以趁早固化。

**同档内出现 H200/H100 混用**:fp8_e4m3 档的 nocorr/fullis 在 H200、ours 在 H100。
按 R18 的判断两者同为 sm90、配置完全相同(triton + CUDA graph 开),不引入配置差异,
差别在显存容量与带宽(影响速度而非数值)。**但这是判断而非实测**,故按承诺逐 arm 标注,
由 kzhao2 决定是否需要在 caption 说明,或是否要求同档同硬件重跑。

**给后续的操作建议**:`TAGSFX` 不应编码硬件(提交时并不知道会落到哪),
应改为编码**用途**(如 `-probe`、`-nocg`),硬件一律事后从 `sacct` 读。

### R23. 取消 fp4 的"探针门控",三臂一起投
先前的做法是先投一个 fp4 探针、验证通过再补另外两臂(见 R16)。**现在撤销该做法**,理由是
排队成本变了:

- 探针 `13671278` 的预计启动从 03:10 退到 **11:30**(`Reason=Resources`),而当前
  **没有任何 Hopper 节点有 8 张空闲卡**(cs-2-2 空 4、eng-1-1 空 7、m13h 各空 0–2),
  两个分区各约 50 个 PENDING。迁分区无济于事;`eng` 只能用可抢占的 `standby`,
  不适合 6 小时的 run。
- 门控意味着:**11 小时排队 + 15 分钟验证,之后另外两臂再排一轮**,fp4 档要拖到 09-15 之后。
- 而 fp4 失败的风险已显著下降:A100 上它的报错是
  `KV4 MHA expects attention_backend ∈ ['triton', ...]` —— 一个**后端断言**,triton 已修掉;
  sglang 帮助文本只要求 CUDA 12.8+/PyTorch 2.8+(均满足)、**未提硬件门槛**;
  同样配置的 fp8_e4m3 已在 **H200 与 H100 上双双跑通**。
- 若判断错误,多投的两臂会在引擎启动阶段 **10–12 分钟内失败**(前几轮实测),代价很小。

**结论**:预期收益约一天,风险约 20 分钟机时。已提交 `13675133`(fullis)、`13675134`(ours),
`TAGSFX=-hop`(按 R22 的建议,不再在后缀里编码硬件)。

**可复用的判据**:探针门控在"失败概率高 + 失败代价大"时值得;当排队时间远超验证时间、
且失败模式已被前几轮实验充分刻画时,**门控本身的等待成本会超过它规避的风险**,应当取消。

### R24. H100 可用性从"判断"升级为"实测"
R18 里我根据 H100/H200 同为 sm90 推断两者行为一致,并据此把 `ours` 臂迁到 cs2。
现已实测确认:`13671277`(cell A / fp8_e4m3 / triton / CUDA graph 开)在 **cs-2-1(H100)**
上 `Train step 1/87 done`,首步 `reward=0.6748`、`seq_len=317.4`,
与同档 nocorr 在 **H200** 上的首步(0.6982 / 298.2)同量级,无任何错误。

**因此可以确定**:Ampere 上那道 fp8 内核门槛在**整个 Hopper 世代**(H100 与 H200)都不存在,
且两者使用**完全相同的配置**(`attention_backend=triton`,CUDA graph 保持开启),
不需要任何硬件专属的变通。这与 A100 必须 `disable_cuda_graph=true` 的情况有本质区别
(见 R17)。

**仍保留的谨慎**:首步数值同量级不等于整条训练轨迹等价。按 R22 的承诺,
各 arm 的实际硬件已逐一标注(nocorr/fullis 在 H200、ours 在 H100),
是否需要在 caption 说明、或要求同档同硬件重跑,由 kzhao2 判断。

### R25. fp4 首跑 OOM —— 但**不是架构不支持**,是 colocate 下的显存挤压
`13671278`(fp4_e2m1 / nocorr / cs-2-2 / H100)`FAILED` 于 21:26,0 步。
**关键是要看清失败发生在哪一侧**:

```
RLTrainer ERROR: Training failed with exception: CUDA error: out of memory
  areal/utils/stats_tracker.py:331 export → :388 _aggregate → :233 _placeholder_scalar
torch.AcceleratorError: CUDA error: out of memory
```
—— traceback 落在 **训练侧**(RLTrainer / stats_tracker),不是 sglang 引擎。
且 `KV4 MHA expects ...` 断言 **0 次**,说明 **fp4 的后端路径本身是通的**,
R23 里"fp4 架构可用"的判断没有被推翻。

**显存证据**:
```
ppo update: memory allocated 53.40 GB, reserved 74.55 GB, device used/total 78.68 (H100 = 80GB)
```
已经打满。而**同为 H100、同样 `mem_fraction_static: 0.8` 的 `ours-fp8_e4m3` 臂正常运行**
(当时 8/87 步),所以差异来自 dtype 本身。

**机制**:AReaL 是推理与训练 **colocate 在同一组 8 卡**上。mxfp4 需要额外的 scale 张量与
反量化缓冲,sglang 侧占得比 fp8 更多,于是**挤压了训练侧 FSDP 可用的显存**。
`mem_fraction_static` 是**推理引擎的静态占比**,降低它等于给训练让出空间。

**与 A100 那次 OOM(R15/R17)的区别——两种形态必须分开**:

| | A100(R15) | 本次 fp4(R25) |
|---|---|---|
| 死的是谁 | **推理服务器进程** | **训练进程** |
| 症状 | `exited with code -9`(SIGKILL,server never healthy) | `torch.AcceleratorError: CUDA OOM` 于 stats 聚合 |
| 阶段 | 引擎启动前 | 已完成 ppo update,收尾统计时 |
| 降 `mem_fraction` 是否有效 | **无效**(0.8→0.6 更早失败) | 待验 |

**处置**:取消 `13675133`/`13675134`(同参数必然同样失败),按 R4 清理残留,
以 `++sglang.mem_fraction_static=0.6` 重投一个验证臂(`TAGSFX=-lowmem`,按 R22 编码用途而非硬件)。
通过再铺开另外两臂。

**修正我先前的预设判据**:R23 取消探针门控时,我把"fp4 失败"等同于"fp4 不可用、曲线停在 3 档"。
这个等号是错的——**失败模式的归属(推理侧 vs 训练侧、断言 vs OOM)决定了它是硬门槛还是可调参数**。
按原判据我会直接放弃 fp4 档,而实际上它很可能只需要一个显存参数。

### R26. fp4 档成立 —— 降 `mem_fraction_static` 即可,R25 的诊断得证
`13675244`(fp4_e2m1 / nocorr / cs-2-2 / H100 / `++sglang.mem_fraction_static=0.6`)
**`Train step 1/87 done`**,首步 `reward=0.6484`、`seq_len=288.4`,
与同一方法在 fp8_e4m3 下的首步(0.6982 / 298.2)同量级。

**R25 的判断得到证实**:fp4 的 OOM 是 **colocate 下 mxfp4 挤压训练侧显存**,
把推理引擎的静态占比从 0.8 降到 0.6、给训练让出空间即可,**不是架构门槛**。
若按 R23 原来的判据("fp4 失败 = fp4 不可用"),整个 fp4 档会被错误放弃。

**但余量仍然很窄**:0.6 下 `ppo update` 实测
`memory reserved 72.61 GB, device used/total 77.34/79.18` —— 距上限不到 2GB。
而 `ours` 臂要额外运行 σ-TIS 算子(`KT_SIGMA_*`),开销更高。
故补提的 `fullis` / `ours` 两臂改用更保守的 **0.55**。

**这不改变实验语义**:`mem_fraction_static` 只控制 sglang 推理引擎的显存预留,
不涉及算子选择、训练超参或 `kv_cache_dtype`。同档三臂之间的差异仍然只有 METHOD 一项,
档内对比(E3 的核心)不受影响。**但不同档位之间该参数不一致(fp8_e4m3 用 0.8、fp4 用 0.55–0.6),
需在结果中标注**,虽然它影响的是显存调度而非数值。

**已提交**:fp4_e2m1 的 fullis 与 ours(`TAGSFX=-lowmem`,`mem_fraction_static=0.55`)。
至此 E3 的两个新档位各三臂全部在途。

### R27. `Invalid reward type` 是**被捕获的单轨迹异常**,不是致命失败(第 4 次监控误报)
监控在 `13671276`(fullis / fp8_e4m3)第 80 步报"致命错误",实际核查:

| 证据 | 结果 |
|---|---|
| job 状态 | **RUNNING** 6:14:37,告警后步数从 80 → **81/87**,仍在推进 |
| 异常归属 | `[RemoteInfEngine Rank 2] ERROR: Workflow execution failed: Invalid reward type: <class 'int'>`,落在 `areal/infra/workflow_executor.py:1147` —— **单个 rollout worker 的一条轨迹**,被该 worker 捕获 |
| 出现次数 | **6 次**,且**只在这一个 arm**(nocorr / ours / fp4-nocorr 均为 0 次) |
| 显存 | 79.09/139.80 GB(H200 140GB),**宽裕**,与 fp4 那次 77.34/79.18 的紧张状况无关 |

**结论**:这是 rollout 侧偶发的奖励类型异常(某条轨迹的 reward 返回了 `int` 而非期望类型),
被 workflow executor 捕获后**丢弃该条轨迹并继续**,不影响训练推进,也不影响已完成步的有效性。
6/87 步中出现、单 arm 独有,属偶发。

**监控判据的第 4 次修正**:先前只要匹配到 `Traceback` 就报致命错误,导致
c10d 的 IPv6 警告(R12 前)、`NameResolve INFO: No such path`、同名目录里上一次的残留、
以及本次被捕获的 workflow 异常,**四次误报**。每次误报都会消耗该 job 唯一的告警名额、
**掩盖之后的真错误**。现已改为三重约束:
1. 排除含 ` INFO: ` / ` WARNING: ` 的行;
2. 排除 `Workflow execution failed`(worker 已捕获);
3. **只在 job 不再 RUNNING 时才判定致命** —— 仍在推进的 job 不可能是致命失败。

**可复用的判据**:判断异常是否致命,**不能只看日志里有没有 Traceback,要看 job 是否仍在推进**。
异常被捕获与否,比异常本身的措辞更有决定性。

### R28. fp8_e4m3 档第二个点:`fullis = 60.20`,与 `nocorr` **统计上不可分**

| arm | acc | 正确数 | 二项 SE |
|---|---|---|---|
| `dose_fp8_e4m3_nocorr` | **0.6217** | 820/1319 | 1.34 pp |
| `dose_fp8_e4m3_fullis` | **0.6020** | 794/1319 | 1.35 pp |

差值 **1.97 pp**,配对差的 `SE_diff = 1.90 pp`,**z = 1.04**。
**不可分。** 这两个数**不能**读成"exact-ratio 修正在 fp8_e4m3 下劣于不修正"。

**这一对是干净对照**:`nocorr` 与 `fullis` 同为 `m13h-1-2`(H200)+ `triton` +
`mem_fraction_static=0.8`,唯一差异就是 METHOD。但**干净不等于有分辨力**——
GSM8K 1319 题在 acc≈0.6 处,单臂 SE 已有 1.34 pp,要在 0.8 power 下检出差异需要 **≥ 约 3.7 pp**。

**对 E3 的直接影响(需提前告诉老师)**:E3 想看的是"把 μ 推大、CIS 相对基线的优势如何变化"。
但若各档位只有单 seed × GSM8K,**档内根本分辨不出算子差异**,曲线上的起伏会全部落在噪声里。
要么补 seed,要么把评测扩到五 benchmark 合并(n≈3065,SE 降到约 0.9 pp)。
这与主表 cell D 的困境同源(NOTES 前文:全块跨度 1.20 分 vs SE 0.79)。

**注意第三臂不同源**:`ours`(`13671277`)跑在 **cs-2-1(H100)**,不是 H200。
同为 Hopper sm90、同 `triton`、同 `kv_cache_dtype`,数值路径一致,
但它与前两臂的对比**多带一项硬件差异**,结论时要标注。

### R29. 一次被查证拦下的迁移:"分区有 idle 节点" ≠ "我的 8 卡作业能起"

fp4 的 `fullis`/`ours`(`13675697`/`13675698`)在 cs2 排到 **11:30 / 12:36**。
`sinfo -p m13h` 当时显示 **3 个节点 idle**,看上去迁过去就能立刻开跑。

**但两分钟后复查,4 个节点全变 `mixed`**:`fslcollab499` 投了约 18 个**单卡** gstandby 作业,
把 m13h-1-1 / 1-2 填到 8/8,剩余两节点只有 7 和 6 张空卡 ——
**没有任何一个节点能满足 `--gres=gpu:8`**。eng-1-1 同样 8/8。

**可复用的判据**:分区级的 `idle` 计数对**整节点独占**作业没有意义。
一批单卡作业可以在不占满分区的前提下**把每个节点都变成碎片**,
使 8 卡作业无处可落。要判断能否起,必须逐节点算
`CfgTRES.gres/gpu - AllocTRES.gres/gpu >= 8`,而不是看 `%T` 是否为 `idle`。

**为什么最终选择原地不动**(而非 `scontrol update Partition=cs2,m13h` 扩大候选):
1. 本 NOTES 前文已记录 `scontrol update Partition=` 会**重置 TimeLimit**;
   m13h 的 `DefaultTime=00:30:00`,一旦静默生效,作业会在 30 分钟后被杀,
   而这类失败**没有报错、只表现为跑不完**,正是最难发现的一类。
2. 两个分区 QOS 不同(cs2 用 `cs`,m13h 的 `AllowQos` 不含 `cs`),
   多分区 + 单 QOS 的组合可能让作业进入不可调度状态,**赔上已排到的名次**。
3. 收益本就为零 —— m13h 现在也起不来。

**净结论:查证顺序救了这一次。** 若按"看到 idle 就取消重投",会同时丢掉
11:30/12:36 两个位置**并且**在 m13h 上照样排队。
**先验证目标可用、再释放已有资源**,顺序不能反。

### R30. E3 的 `ours` 臂与主表 CIS 是**同一算子**(逐字核对);σ 参数定为 b=2.3

**先说风险**:`ours` 与 `fullis` 的 hydra 覆盖 `OVR` **完全相同**
(都是 `action=clamp lower=1e-6 upper=1e6`),两者唯一的差别是五个 `KT_SIGMA_*` **环境变量**。
也就是说 **一旦 export 没生效,`ours` 会静默退化成 `fullis`**,
而且**日志不打印 σ 参数、`config.yaml` 不含该字段、ktdump 也只有
`old_logp/prox_logp/cur_logp/version` 四列**——三条常规取证路径全都看不出来。
这是 E3 目前最危险的一种静默失败。

**已排除(用产物反推控制流)**:trial 目录名 `ours-fp8_e4m3-h200` 是
`trial_name=${METHOD}-${KVDTYPE}${TAGSFX}` 拼出来的,该目录存在即证明
`case $METHOD in ours)` 命中;`export` 与 `OVR=` 在**同一分支同一行段**内,
故 export 必然已执行。**比去日志里找打印更可靠——因为它根本不打印。**

**逐字核对**:`dose.sbatch:23-24` 与 `rl/cells.sbatch:43-44`、`rl/fp8.sbatch:24-25` 三处一致:
```
OVR="actor.rejection_sampling.action=clamp actor.rejection_sampling.lower=1e-6 actor.rejection_sampling.upper=1e6"
export KT_SIGMA_TIS=1 KT_SIGMA_C1=1.0 KT_SIGMA_B=${LAMP:-2.3} KT_SIGMA_A=1e9 KT_SIGMA_FLOOR=5e-3
```
→ **E3 的 CIS 与主表的 CIS 是同一算子**,两边结果可以互相引用。

**`LAMP` 未传,故 b=2.3**:`LAMP` 仅在 `submit_batch.sh:34-39` 中对
`dsv2`(15.0)/ `q30b`(0.95)赋值;E3 是 **cell A**(Qwen1.5-MoE),不在该规则内,
`dose.sbatch` 在任何一次提交里都没有接收过 `LAMP`。三个 `ours` 臂一律取默认 **2.3**。

**实际生效的 CIS 形式**(`areal/utils/functional/functional.py:423-439`):
`sig = max(c1*(1-p), floor)`,`log M = clip(log k, -a*sig, +b*sig)`,`p = exp(prox_logp)`。
代入 `c1=1.0, b=2.3, a=1e9, floor=5e-3`:
- 上界 `= 2.3*(1-p)`;下界 `= -1e9*(1-p)` → **实际是单边裁剪**,只截上尾。
- `floor=5e-3`:即使 `p→1`,带宽也不低于 `2.3*5e-3 = 0.0115`。

**一个容易读错的地方,必须讲清楚**:源码注释说 `c1=0.195` 是 Qwen1.5-MoE **实测**的
scale law(`sigma(log k | p) = c1*(1-p)`)。部署配置把 `c1` 改成 **1.0**,
这意味着 `sig` **不再是实测 σ**,而就是 `(1-p)` 本身。
所以部署的 `b=2.3` **不能读成"2.3 个 σ"**——换算成实测 σ 应是 `2.3/0.195 ≈ 11.8 σ`,
与源码默认的 `b=12`(即 12 σ)**几乎一致**。两套参数化只是 `c1*b` 的不同拆法。
**真正的配置差异不在带宽,而在 `a`**:源码默认 `a=20`(即 ±3.9 的双边裁剪),
部署用 `a=1e9`(单边)。**单边 vs 双边才是 CIS 的实质选择,带宽两者等价。**

### R31. 两套评测协议的差异,以及为什么要补一个 bf16 对照臂

仓库里同为 **cell A / bf16 / nocorr / GSM8K n=1319**,却存在两个相差 5.6 分的数:
- `SIGMA_TIS.md:172` 三种子 56.10 / 54.44 / 56.10 → 均值 **55.55**(曲线里当作 bf16 基线)
- `eval_suite.tsv` 的 `suite_nocorr_s1/s2/s3` = 60.42 / 60.65 / 62.55 → 均值 **61.2**

**先记一个差点犯的错**:我一度怀疑 55.55 其实是**未训练的 base**——
因为 `suite_base` 的 gsm8k = **55.50**,与之只差 0.05。
逐行核对后否定:55.55 是 SIGMA_TIS 三种子的均值,与 `suite_base` 接近纯属巧合。
**0.05 的接近度足以骗过直觉,但核对只花了一分钟**;在写进 NOTES 之前先查,是对的。

真正的原因是**两套评测协议**(`SIGMA_TIS.md:206` 把新的一套称为"统一 \boxed + math-verify"):

| | 旧:`rl/eval_gsm8k.py` | 新:`rl/eval_suite.py` |
|---|---|---|
| prompt | 裸问题(:15-17) | 问题 + `INSTR`(:14)`"...put your final answer within \boxed{}."` |
| 判分 | 取文本最后一个数字的正则(:24-25) | `math_verify.parse/verify`(:58) |
| `max_tokens` | 1024 | 1536 |
| `max_model_len` | 2048 | 4096(默认) |
| `enable_thinking` | 不传 | `False` |

**对 E3 的两点影响**:
1. E3 的 dose 数字走 `eval_gsm8k.py`,与既有 bf16 / fp8_e5m2 两行**同协议**。
   所以"评测口径"**不是**两者不可比的原因 —— 不可比只剩 **硬件** 与 **训练轨迹** 两项。
   (此前 R21 列的三项里没有协议,结论未受影响;这里是把范围**收窄**并给出依据。)
2. 反过来,E3 的数字**不能**与主表的 `suite_*` 行并读。
   已提交 `13676235` 对 `nocorr-fp8_e4m3-h200` 的最终 epoch 做五 benchmark 复评
   (tag `dose5_fp8_e4m3_nocorr`),**它与既有 `dose_fp8_e4m3_nocorr = 0.6217` 之差,
   就是两套协议之差在同一 checkpoint 上的直接测量值** —— 无需重训。

**因此补提 `13676237`(`kv_cache_dtype=bf16` / nocorr / `TAGSFX=-ctrl`)**:
既然协议已经一致,只要让 bf16 档也走完全相同的 dose 路径
(同脚本、同 Hopper + `triton`、同 `mem_fraction_static=0.8`),
E3 就**不再依赖 2026-08 那批旧数据**,硬件与训练轨迹两项差异一并消除。

这是一次**高信息量的单点实验**:结果落在 55.55 附近 → 旧行可并入曲线;
落在 62 附近 → 旧行作废,整条曲线改用自跑数据。**两种结果都推进结论,不存在白跑。**
先只投 nocorr 一个臂,理由是并发上限(前文记过同分区不宜超过 3–4 个),
且 nocorr 是每一档的基线锚点,单臂即可判定旧行去留。

### R32. 老师的 `CIS_待补实验.md` 已不可恢复,E5 无法执行

该文档是**在对话里粘贴的,从未落盘**:仓库内没有,`find ~` 没有,
会话记录 `8f1a70a4-…jsonl` 里也只剩上下文压缩后的摘要引用,原文已丢失。

**注意不要拿 `问题.md` 顶替**:它虽然也有 E1–E5,但那是**另一套编号**
(E1 均值分解验收 / E2 标准化重拟合(s 坐标)/ E3 √pos 律去混杂 /
E4 逐位置恒等式 / E5 序列级三个数),内容是**分析任务**而非训练实验,
与老师那份的 E3(受控噪声注入 dose–response)对不上。混用会跑错实验。

**需要用户重新提供该文档**,建议这次存进仓库(如 `docs/CIS_待补实验.md`)以免再丢。
E1/E2/E4 本就被 `TODO_kzhao2_from_tianruny.md` 的 A–C 阻塞;E5 现在是**连要求都拿不到**。
