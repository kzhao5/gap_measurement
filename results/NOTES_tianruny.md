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
