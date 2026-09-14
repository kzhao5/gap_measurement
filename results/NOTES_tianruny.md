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

### R33. 两套评测协议的差实测为 **+6.06 分**,并由此解释掉仓库里 5.66 分的矛盾

canary `13676235`(`COMPLETED 00:10:20`)对**同一个 checkpoint**
(`nocorr-fp8_e4m3-h200` 的 `epoch2epochstep28globalstep86`)跑了 `eval_suite.py`,
与该 checkpoint 已有的 `eval_gsm8k.py` 结果合起来,得到一个**三点分解**
(GSM8K,同一批 n=1319 题,同一份 gold):

| | prompt | 判分器 | acc |
|---|---|---|---|
| `eval_gsm8k.py`(旧协议) | 裸问题 | 末位数字正则 | **62.17** |
| `eval_suite.py` 的 `gsm8k_em` | + `\boxed` 指令,`max_tokens` 1536 | **同一个**正则 | **65.50** |
| `eval_suite.py` 的 `gsm8k` | + `\boxed` 指令,`max_tokens` 1536 | math-verify | **68.23** |

- prompt + 生成长度(1024→1536)贡献 **+3.33**
- 判分器(末位数字正则 → math-verify)贡献 **+2.73**
- **协议总效应 = +6.06**

能做这个分解,靠的是 `eval_suite.py` 对 gsm8k **同时输出两个判分**:
`judge_em`(:68-70)与 `eval_gsm8k.py`(:24-25)**逐字同构**
(同样的 `re.findall(r"-?\d+\.?\d*", ...)`、同样的 `nums[-1].rstrip(".") == gold`),
gold 提取也都是 `a.split("####")[-1]`。**所以 `gsm8k_em` 正好是"只换 prompt、不换判分"的中间态。**

**用它去对 R31 里悬着的那个矛盾**:

| | |
|---|---|
| `SIGMA_TIS.md:172` bf16 nocorr(旧协议)三种子均值 | **55.55** |
| `eval_suite.tsv` `suite_nocorr` 的 gsm8k(新协议)三种子均值 | **61.21** |
| 两者之差 | **+5.66** |
| 本次实测的协议效应 | **+6.06** |
| **残差** | **0.40 分** |

**结论:那 5.6 分完全是评测协议造成的,不是两批训练跑出了不同质量的模型。**
仓库里这两组数可以互相换算(约 ±6 分),R31 里"55.55 可能其实是未训练 base"的怀疑就此排除。

**但要写清适用边界**:`+6.06` 是在**一个** checkpoint 上测的,而 `+5.66` 是从**另外三个**
checkpoint 推出来的。协议效应未必对所有模型质量都是常数——`\boxed` 指令对弱模型的
增益通常更大。**0.40 分的吻合是强证据,不是证明**;要坐实需在第二个 checkpoint 上重复。
作为换算系数使用已经足够,作为定律引用则不够。

### R34. 五 benchmark 口径下的第一个 E3 数字,以及为什么现在**还不能**下结论

`dose5_fp8_e4m3_nocorr` 等权五 benchmark 均值 = **34.14**
(gsm8k 68.23 / math500 25.80 / svamp 65.00 / minerva 6.62 / olympiad 5.04)。

**聚合方式必须等权,这一点先核对过**:等权可精确复现已发表的
CIS **34.47** / nocorr **30.99** / base **28.93**;若按题数加权则得 41.57 / 37.41 / 34.55,
整体高约 7 分(被 n=1319 的 GSM8K 主导)。
**用错口径会让所有数字系统性偏高、且从单个数字上完全看不出来。**

放进主表 cell A 的刻度:

| | 五 benchmark 等权均值 |
|---|---|
| base | 28.93 |
| bf16 nocorr(3 seed) | **30.99**(样本 SD 0.22) |
| **本次 dose5 fp8_e4m3 nocorr** | **34.14** |
| bf16 CIS / ours(3 seed) | 34.47 |

**一个未加任何修正的 fp8_e4m3 运行,打平了 bf16 下的 CIS(34.14 vs 34.47)。**
这个数字很扎眼,但**现在不能读成"量化有益"、更不能读成"CIS 的优势不成立"**:
它与主表那三个 seed 之间,除了 `kv_cache_dtype` 还差着**硬件、代码库日期
(2026-08 vs 2026-09)以及整条训练轨迹**三项。

`13676237`(bf16 / nocorr / 走完全相同的 dose 路径)正是为排除这三项而提的:
- 落在 **31** 附近 → 那 +3.15 来自 dose 管线本身,与量化无关;
- 落在 **34** 附近 → 主表三个 seed 与 dose 管线之间存在系统差异,E3 必须整条曲线自跑,
  既有 bf16 / fp8_e5m2 两行一律作废。

**在它落地之前,E3 的任何跨档结论都不成立。**

### R35. 第 5 次监控误报:把监控从"单作业"泛化到"多作业"时,文件 glob 悄悄扫进了历史

`b70xeung6` 报 `FATAL FIPS SELFTEST FAILURE`。查证:命中该串的是
`suite_13627007.out` 与 `suite_13632168.out`,**mtime 都是 2026-09-10 15:53**,
即四天前被 `opencv-python-headless` 打死的那两次评测(见前文 FIPS 条目)。
当时正在跑的 `13676298` 命中 **0 次**,`RUNNING @ dw-1-4`,五个数据集全部正常加载;
十分钟前成功完成的 canary `13676235` 也是 0 次。**纯误报。**

**成因是一次"无害的泛化"**:
- 第一版监控盯单个作业,写的是 `$LD/suite_$J.out` —— 天然免疫;
- 第二版要盯"以后所有 kt-suite 作业",我把它改成 `$LD/suite_*.out` ——
  **同一行代码的语义从"这个作业的日志"变成了"这个目录下的全部历史日志"**。

日志目录是**累积**的。只要作业名模式不变,glob 就会一直向后兼容地扫到几个月前的失败。
**按作业 ID 命名的日志,其 glob 形式天然带有跨时间的副作用。**

**危害不在于这一次噪声,而在于闩锁**:该监控用 `fatal=1` 保证同一错误只报一次
(这本身是 R27 的正确做法),于是这次误报**吃掉了它唯一的告警名额,
此后对真正的致命错误失明**。R27 已经写过"每次误报都会掩盖之后的真错误",
这次是我自己再犯——**去重与误报叠加时,误报的代价从"噪声"升级为"失明"**。

**修法:不要用名字锚定,要用时间锚定。**
```bash
# 错:扫全部历史
grep -hE "$PAT" $LD/suite_*.out
# 对:只扫最近修改过的日志
find $LD -name 'suite_*.out' -mmin -30 -exec grep -hE "$PAT" {} +   # 注意:本机不能用 -newermt,见 R41
```
**可复用的判据**:任何"盯一类作业"而非"盯一个作业"的监控,
其文件选择都必须带**时间下界**;否则它报的是历史,不是现状。

### R36. 同一个迁移,30 分钟前是错的、现在是对的 —— 区别在于两次都先查了

R29 记录过:我本想把排队的 arm 迁到 m13h,因为 `sinfo` 显示 3 个节点 `idle`;
逐节点核算后发现 `fslcollab499` 的约 18 个**单卡**作业把每个节点都切碎,
**没有一个节点能容下 `--gres=gpu:8`**,遂放弃迁移。

30 分钟后,同样的核算给出**相反**的结果:m13h-1-1 / 1-2 / 2-1 **各 0/8 已用、空闲 8**
(那批单卡作业是短作业,已全部退出)。于是把 `13676237` 从 cs2 迁到 m13h:
**它在 cs2 的预计起始是 `2026-09-15T00:00:00`(约 20 小时后),迁过去后立即 `RUNNING`。**

**两次的差别不在判断力,而在于两次都做了同一件事**:动手前逐节点算
`CfgTRES.gres/gpu − AllocTRES.gres/gpu >= 8`。
第一次若凭 `idle` 就迁,会白丢队列位次;第二次若凭"上次不行"就不查,会白等 20 小时。
**集群的空闲状态在分钟级翻转,任何基于上一次观察的结论都必须重新验证**;
"上次查过了"和"没查过"在决策上等价。

**另外记一个容易误读的现象**:cs2 只有 2 个节点,且**都被我自己的两个作业占满**
(`13671277` / `13675244`)。我的三个 PENDING 优先级同为 100311,
所以 `13676237` 实际是在**排我自己的队**,不是集群繁忙。
`Reason=Priority` 这个字段不区分"别人挤占"和"自己挤占"。

### R37. bf16 整档一次投满,因为 `ours` 臂不是可选项

同时提交 bf16 档三臂(`13676300` nocorr / `13676301` fullis / `13676302` ours)。
不只投 nocorr 的理由:**bf16 = 零注入噪声,是整条 dose 曲线的锚点**。
E3 要回答的是"CIS 相对基线的优势**随注入噪声如何变化**",
**没有 bf16 的 `ours`,这条曲线连起点都没有**,后面 fp8 / fp4 档的 CIS 数字无从参照。
先前只投 nocorr 是受 cs2 队列拥堵所限,不是判断它够用。

硬件上 m13h 反而更合适:fp8_e4m3 的 nocorr / fullis 两臂正是跑在 m13h-1-2(H200),
bf16 档同在 m13h,可把档间硬件差异进一步收窄
(仍余 `ours-fp8_e4m3` 在 cs-2-1 / H100 这一处例外,见 R28)。

代价是并发从 2 个训练作业升到 5 个。**已据此把重载耗时纳入自动监控**
(>500s 预警;实测健康区间 107–240s),不再手动抽查 —— 这是对 R7 早期预警的复用:
并发翻倍之后,靠人工抽查等于没查。

### R38. 协议差在第二个 checkpoint 上复现(+6.03,极差 0.07),R33 的保留条件解除

R33 当时写明:`+6.06` 只在一个 checkpoint 上测过,**"是强证据但不是证明,
要坐实需在第二个 checkpoint 上重复"**。现已重复:

| 档位/方法 | 旧协议 | `gsm8k_em`(新 prompt,同一正则) | `gsm8k`(新 prompt + math-verify) | prompt 效应 | 判分效应 | **合计** |
|---|---|---|---|---|---|---|
| fp8_e4m3 / nocorr | 62.17 | 65.50 | 68.23 | +3.33 | +2.73 | **+6.06** |
| fp8_e4m3 / fullis | 60.20 | 64.06 | 66.19 | +3.86 | +2.13 | **+5.99** |

**两个 checkpoint 的合计相差仅 0.07 分**,均值 **+6.03**;
与仓库里 SIGMA_TIS(55.55)和主表(61.21)那 **+5.66** 的差,残差 **0.37**。
**~~协议差在这个模型尺度上是可复现的常数,R33 的保留条件解除。~~**
**⚠ 此句已被 R46 推翻**:第三个 checkpoint(`ours`)给出 +4.70,三点极差 1.36,
它不是常数而是一个分布。核心解释(协议差解释了仓库那 5.66)反而更强了,但"常数"的说法是错的。

**但要注意一个细节,它影响怎么引用这个数**:合计很稳(0.07),
**分解却不稳**——prompt 效应 +3.33 vs +3.86、判分效应 +2.73 vs +2.13,各自偏移约 0.5。
两项此消彼长,恰好抵消。
**所以可以放心引用"协议总差约 +6",但不要单独引用"`\boxed` 值 +3.3"**——
后者在两个 checkpoint 上就已经不稳定了。

已把这套计算固化为 `env_local/dose5_curve.sh`(不在仓库内),
其中写死了两条容易出错的口径:五 benchmark 均值必须**等权**;
`gsm8k_em` **不计入**均值(它是同一批题的第二种判分,计入等于给 GSM8K 双倍权重)。

### R39. `fullis` 的种子方差是 `nocorr` 的十倍 —— R28 的结论不变,但理由要换更强的

用同一套等权口径把主表 cell A 的参照值算全,发现:

| bf16 参照(主表,3 seed) | 等权均值 | 种子 SD |
|---|---|---|
| `nocorr` | 30.99 | **0.22** |
| `ours`(CIS) | 34.47 | **0.48** |
| `fullis` | 31.86 | **2.23** |
| `base`(1 seed) | 28.93 | — |

**`fullis` 的种子 SD 是 `nocorr` 的十倍。** 这与 `SIGMA_TIS.md` 的记载一致
(GSM8K 上 full-IS 三种子 64.59 / 55.50 / 59.06,极差 9.1,而它把这归因于"未截断 IS 脆弱")。

**对 R28 的修正**:R28 判定 fp8_e4m3 的 `nocorr`(62.17)与 `fullis`(60.20)
统计上不可分,依据是**二项 SE**(1.90pp,z=1.04)。这个依据**太弱了**——
单 seed 运行的真实不确定性由**种子方差**主导,而 `fullis` 的种子方差远大于评测抽样噪声。
**结论(不可分)不变,但强度应当更高**:不是"差异约 1 个标准误",
而是"`fullis` 这一臂本身就带着约 ±2 分的种子噪声,单 seed 根本定不住它的位置"。

**可复用的判据**:判断两个单 seed 结果能否分开,**二项 SE 只是下界**;
必须再问"这个方法的种子方差有多大"。对种子稳定的方法(nocorr SD 0.22)二项 SE 够用,
对脆弱的方法(fullis SD 2.23)则严重低估。**方法本身的脆弱性是误差预算的一部分。**

### R40. 五 benchmark 口径下 fp8_e4m3 两臂的位置(仍不构成跨档结论)

| | 等权五 benchmark |
|---|---|
| `dose5_fp8_e4m3_nocorr` | **34.14** |
| 主表 bf16 `nocorr`(3 seed) | 30.99 ± 0.22 |
| `dose5_fp8_e4m3_fullis` | **32.56** |
| 主表 bf16 `fullis`(3 seed) | 31.86 ± 2.23 |

**两者的错位方式不同,这本身是信息**:
- `nocorr` 高出 **+3.15**,而 bf16 `nocorr` 的种子 SD 只有 0.22 —— 相当于 **14 个种子标准差**。
  任何真实的量化效应都不会有这个量级;**这几乎必然来自管线/硬件/代码库差异,而非 `kv_cache_dtype`。**
- `fullis` 只高出 **+0.70**,完全落在它自己 ±2.23 的种子噪声里,**无法分辨**。

`13676300/01/02`(bf16 三臂,已在 m13h 上运行)正是为定住这一点而提的。
**在它们落地之前,以上仍不构成任何跨档结论**,只说明"既有主表值与 dose 管线之间存在
一个远大于量化效应的系统差",与 R34 的判断一致。

### R41. 订正 R35 的修法:本机 `find` 是 `bfs`,`-newermt` 会直接报错

**R35 给出的修法是错的,已在上文就地订正,此处说明原委。**

R35 主张用 `find ... -newermt '-30 minutes'` 给日志扫描加时间下界。
实测该语法在本机**不可用**:

```
$ find $LD -name 'dose_*.out' -newermt '-40 minutes'
bfs: error: bfs -S dfs -regextype findutils-default ... -newermt "-40 minutes"
bfs: error: Invalid timestamp.
```

**本机的 `find` 是 `bfs`(一个广度优先的 find 替代品),不是 GNU findutils。**
`bfs` 的 `-newermt` **不接受相对时间字符串**,遇到就报错退出、不输出任何文件。
可用的写法是 `-mmin -N`(实测匹配到 5 个活跃日志,GNU/bfs 都支持)。

| 写法 | 匹配数 |
|---|---|
| `-newermt '-40 minutes'` | **0(报错)** |
| `-newermt '40 minutes ago'` | **0(报错)** |
| `-mmin -40` | **5** |
| 无时间条件 | 14 |

**后果有三层,第二层最值得记**:

1. 按 R35 建的重载耗时监控 `brgfb1su9`,**从 armed 那一刻起就在扫描零个文件、全程失明** ——
   正是 R35 声称修好的那种故障,换了个原因又发生一次。
2. **我对该修法做的"正向验证"是无效的。** 当时我跑
   `find ... -newermt '-30 min' -exec grep -lE 'FATAL FIPS' {} + | wc -l` 得 **0**,
   并把它读作"时间窗成功排除了四天前的日志"。
   但 **0 也正是 find 报错、什么都没扫时的输出** —— 这个测试**根本区分不开
   "过滤器正确工作"与"过滤器整个失效"**,两种情况给出完全相同的观测。
   我在 R35 里刚写下"沉默不等于健康",随即就用一个无法证伪的测试去验证它。
3. R35 的修法已经推送进仓库。照抄的人会造出同样失明的监控,故必须就地订正。

**可复用的判据(这条比前两条都重要)**:
**验证一个过滤器时,"期望为空"的测试没有证据价值。**
必须先用一个**必然命中**的条件(如把阈值降到 0、或换一个一定存在的字符串)
证明管道端到端能出数,再去看真实条件下是否为空。
否则"没有输出"永远同时兼容"健康"与"坏掉",而你会默认选择前者。

本次正是靠"把阈值从 500 降到 0、发现仍无输出"才暴露的 ——
**这个降阈值动作应当成为每个新监控 armed 后的标准动作。**
新监控已改用 `-mmin`,并在启动时打印窗口内的文件数(`ARMED ... N 个活跃日志`),
使"选择器坏了"在第一条事件里就能看出来,而不必等到需要它报警时。

**顺带记录一个被预判到的现象**:三个 bf16 臂同时起跑后,
`13671277` 的重载耗时从 107s 升至 **191s**、`13675244` 从 117s 升至 **224s**,约翻倍;
步频也从 3.4 min/step 降到约 4.6。是我提高并发的直接代价,
仍远低于 500s 阈值与各自的时限余量(数小时),属可接受范围。

### R42. 一条只存在于监控定义里的决策规则,差点随监控一起消失

按 R41 的判据清理监控时,我停掉了 `bvlqtb24v`(fp4 两臂在 `mem_fraction_static=0.55`
下的验证):它 3 小时只输出 1 行,且那行是自己的横幅而非事件,**我无法把"健康"与"坏掉"区分开**,
且职责已被 `bghyqpwvb` 覆盖。

**停掉后看到它的完整定义,才发现它并没有坏**——它是个 `seq 1 700` × `sleep 60`
(约 11.6 小时覆盖)的完整循环,只因两个臂仍 PENDING 才没有事件可发。
**判断方法没错**(不可验证就不该信任),**但我为此付出了一个没预料到的代价**:

它的命令文本里写着一条**别处都没有的决策规则**:

> 若 `fullis-fp4_e2m1` / `ours-fp4_e2m1` 在 `mem_fraction_static=0.55` 下**仍然 OOM**,
> **下一步应降训练侧峰值 `actor.mb_spec.max_tokens_per_mb` 4096→2048,
> 而不是继续降 `mem_fraction_static`。**
>
> 理由(承 R25/R26):`mem_fraction_static` 只是把显存在推理与训练之间**重新分配**;
> 当 OOM 发生在**训练侧**时,继续把显存让给训练会**同时**把 sglang 的 KV 压到不够用,
> 迟早从"训练 OOM"翻转成"推理引擎起不来"(即 R15/R17 在 A100 上见过的那种形态)。
> 降 `max_tokens_per_mb` 则是直接**降低训练侧的峰值需求**,不动分配比例。

这条规则现在才落进 NOTES。**教训不是"不要停无法验证的监控",而是:
监控是易失的,决策规则不该只活在监控的定义里。**
一个只在特定条件下才会触发的分支,如果只写在 watcher 的 `echo` 里,
那么 watcher 一停、会话一换,它就不存在了 —— 而它恰恰是**将来出问题时最需要的那句话**。

**附:`find` 实现确认**(承 R41)。本机 `find` 为 **`bfs 4.1.1`**。
已无歧义复查 `env_local` 下全部 11 个脚本:`-newermt` **0 处**
(正向对照:同目录 `sglang` 命中 10 行,证明检索本身有效);
`find` 调用仅 `archive_migrate.sh:18` 一处,为 `find "$src" -type f`,纯 POSIX,不受影响。

### R43. 预先登记 bf16 对照臂的判读规则(在数据到达之前)

**为什么现在写**:`13676300/01/02` 完成后会自动接 `eval_tp.sbatch → eval_gsm8k.py`,
产出的是**旧协议 GSM8K**。而 R34 写的判读规则("落在 31 附近 / 34 附近")
用的是**五 benchmark 等权刻度** —— **两者不是同一把尺子**。
若不先换算,等数字出来很容易不自觉地去迁就它。本会话已有两次发布后撤回的记录,故前置。

**把主表(新协议)减去协议效应 6.03,与 SIGMA_TIS(旧协议)对照**:

| 方法 | 主表新协议 | 减 6.03 | SIGMA_TIS 旧协议 | 残差 | 旧协议种子 SD |
|---|---|---|---|---|---|
| `nocorr` | 61.21 | **55.18** | **55.55** | **+0.37** | 0.96 |
| `fullis` | 62.07 | 56.04 | 59.72 | +3.68 | 4.58 |
| `ours` | 67.88 | 61.85 | 64.29 | +2.44 | 0.92 |

**只有 `nocorr` 这一行是紧的**(残差 0.37,远小于其 0.96 的种子 SD)——
而 `nocorr` 恰好就是我要测的那个臂。`fullis` 残差 3.68 但其种子 SD 高达 4.58,
本就分不开;`ours` 残差 2.44 对其 0.92 的 SD 偏大,**是一个尚未解释的缺口**。

**预先登记的判读(针对 `13676300` 的旧协议 GSM8K 数字)**:

- 落在 **54–57**(两个独立历史来源一致指向 55.2–55.6)
  → dose 管线 ≡ 历史管线。既有 bf16 / fp8_e5m2 两行**可并入曲线**。
  同时意味着 **fp8_e4m3 的 62.17 比 bf16 高约 +6.8**,这个量级不可能是量化本身带来的,
  必须另找原因(最可能是 `attention_backend=triton` 与历史默认后端不同)。
- 落在 **60–64**
  → dose 管线比历史管线整体高约 +6~+7。既有两行**作废**,E3 整条曲线只用自跑数据;
  同时 R40 里 `dose5_nocorr` 高出主表 3.15 分(14 个种子 SD)也随之得到解释。
- 落在 **57–60**(两者之间)
  → 无法判定,需再看 `13676301/02` 两臂是否同向偏移;若三臂同向,按第二种处理。

**这条规则在数字到达前写下,不得事后修改。** 若三种情形都不符(如 <54 或 >64),
则说明存在第三种我未预料的因素,届时**必须明确记为"预登记失败"**,而不是补一条新规则去圆。

**一个附带发现,留给 kzhao2 核对**:`SIGMA_TIS.md:172` 的 `nocorr` 三种子是
`56.10 / 54.44 / 56.10` —— **s1 与 s3 完全相同**(0.5610×1319 = 740.0,即同为 740/1319)。
贪心解码下不同 seed 得到逐题完全一致的正确数,虽非不可能,但值得确认
是否为**误填或复制**。若其中一个有误,则上表 `nocorr` 行 0.37 的残差需重算。

### R44. 订正 R28 的功效门槛(3.7 → 5.3pp),并预先登记 `ours` 臂的判读

**先订正一个已推送的算错。** R28 写"GSM8K n=1319 在 0.8 power 下需 ≥ 约 3.7pp 才能检出差异"。
复核发现 **3.74pp 是 `(1.96+0.84) × SE_单臂`**,即"**单臂对一个已知常数**"的公式;
而 `nocorr` 与 `fullis` 是**两个都带误差的测量值**,必须用 `SE_diff`:

```
SE_单臂  = sqrt(p(1-p)/1319)            = 1.34 pp
SE_diff  = sqrt(p1q1/n + p2q2/n)        = 1.90 pp
门槛     = (1.96+0.84) × SE_diff        = 5.32 pp   ← 正确值
(我写的 3.74 = (1.96+0.84) × 1.34,用错了公式)
```
**我把门槛低估了约 42%。** 方向上这个错误**加强**而非削弱 R28 的结论
(分辨率比我报告的更差),但数字必须改对。

**顺带收窄 R39 的表述**(它说"二项 SE 只是下界",过宽了)。把主表 cell A 旧协议 GSM8K
的实测种子 SD 与二项 SE 并排看:

| 方法 | 实测种子 SD | 二项 SE | 判断 |
|---|---|---|---|
| `nocorr` | 0.96 | 1.34 | 种子 SD **低于**二项 SE → 二项 SE 已够用 |
| `ours` | 0.93 | 1.34 | 同上 |
| `fullis` | **4.58** | 1.34 | 远超 → 二项 SE **严重低估**,R39 的警告只对这一类成立 |

即:**"二项 SE 是下界"这句话只对种子脆弱的方法成立**;对种子稳定的方法,
观测到的种子离散度本来就落在二项噪声之内,再叠加会重复计数。

**预先登记(针对 `13671277` 即将产出的 `dose_fp8_e4m3_ours`,旧协议 GSM8K)**:

档内已有 `nocorr = 62.17`、`fullis = 60.20`。按 5.32pp 门槛:

| 若 ours 落在 | 判读 |
|---|---|
| **≥ 70.9** | CIS 的优势在注入噪声下**完整保持甚至放大**(cell A 在 bf16 下 CIS 对 nocorr 为 64.29 vs 55.55 = **+8.74**;同样幅度加在 62.17 上即 70.91) |
| **67.5 – 70.9** | 优势**保持但被削弱**——仍与 `nocorr` 统计可分(≥67.49),但小于 bf16 下的 +8.74 |
| **56.9 – 67.5** | **与 `nocorr` 不可分**。注意这不等于"CIS 无效",只等于"本档单 seed 分不出来" |
| **≤ 56.9** | CIS 在该噪声档下**显著劣于**不修正 —— 若出现,应优先怀疑运行故障而非算子失效 |

**这个档位是有判别力的,尽管它分不开小差异** —— 这一点值得单独指出:
R28 说 GSM8K 单档分不出算子,容易被读成"E3 白做"。但门槛 5.32pp 是针对**任意小差异**;
而 E3 真正要检验的效应量是 **+8.74**,**远在门槛之上**。
所以本档**无法**分辨 `nocorr` 与 `fullis` 那种 ~2pp 的差距,
却**足以**分辨"CIS 的优势是否还在"。**功效不足是相对于效应量而言的,不是绝对属性。**

**一个必须同时声明的前提**:上表的 70.9 建立在"把 bf16 的 +8.74 绝对值搬到 62.17 之上"。
但 62.17 本身是否与 bf16 的 55.55 同一把尺子,**正是 R43 要判定的事**。
若 `13676300` 显示 dose 管线整体高出历史约 +6~7,则本表的绝对值需随之平移,
**相对判读(±5.32pp 门槛与 +8.74 效应量)不受影响**。
**本表在数字到达前写下,不得事后修改。**

### R45. fp8_e4m3 档三臂齐全:预登记判读落在「与 nocorr 不可分」,但这**不是**无结论

`13671277` 干净收尾(87/87 步,epoch 3/3,评测链自动触发),**未发生收尾挂起**。

| fp8_e4m3 档(旧协议 GSM8K,n=1319) | acc | 正确数 |
|---|---|---|
| `nocorr` | 62.17 | 820 |
| `fullis` | 60.20 | 794 |
| **`ours`(CIS)** | **63.61** | **839** |

**R44 预登记的判读:63.61 落在 56.9–67.5,即「与 `nocorr` 不可分」。** 按原样记录,不作事后调整。

档内三组两两比较(门槛 5.32pp):

| 对比 | 差 | SE_diff | z | 判定 |
|---|---|---|---|---|
| ours − nocorr | +1.44 | 1.88 | 0.77 | 不可分 |
| ours − fullis | +3.41 | 1.89 | 1.80 | 不可分 |
| nocorr − fullis | +1.97 | 1.90 | 1.04 | 不可分 |

**整档跨度仅 3.41 分,三个算子互相都分不开。**

#### 但"不可分"不等于"没信息" —— 这一档排除了 bf16 量级的优势

R44 事先论证过:本档**分不开 ~2pp 的小差异,却足以分辨 +8.74 这个 E3 真正关心的效应量**。
既然有判别力却没测到,这个否定结果是有内容的:

| | CIS 相对 nocorr 的优势 | 95% CI |
|---|---|---|
| bf16(SIGMA_TIS,3 seed) | **+8.74** | [+7.23, +10.25] |
| fp8_e4m3(本次,单 seed) | **+1.44** | [−2.25, +5.13] |

**两个区间不相交。** 若真实优势仍为 +8.74,观测到 +1.44 相当于偏离 **3.59 个 SE**
(已并入 bf16 侧的种子不确定性)。**"CIS 在 fp8_e4m3 下仍保有 bf16 量级的优势"这一假设被数据排除。**

#### 然而有一个同样有力的替代解释,在 bf16 对照臂落地前无法排除

**注意两个档位之间各臂的移动方式极不对称**:

| | bf16(历史) | fp8_e4m3(本次) | 移动 |
|---|---|---|---|
| `ours` | 64.29 | 63.61 | **−0.68(几乎没动)** |
| `nocorr` | 55.55 | 62.17 | **+6.62(大幅上移)** |

所谓"CIS 的优势收缩",**完全可以由"管线把 `nocorr` 抬高了"来解释,而不需要"噪声削弱了 CIS"**。
而 R40 已经独立发现:本管线的 `dose5_nocorr` 比主表 bf16 `nocorr` 高出 **+3.15 分,
相当于 14 个种子 SD** —— 这个量级的管线偏移**是真实存在的**,不是假设。

**两种解释对同一组数据同样自洽**:
1. **噪声假说**:fp8_e4m3 的失配噪声削弱了 CIS 的相对优势 → E3 想要的结论;
2. **管线假说**:dose 管线(2026-09 代码库 / Hopper / `attention_backend=triton`)
   系统性地抬高了弱基线,把差距压平 → 与噪声无关,是测量伪像。

**`13676300`(bf16 nocorr)与 `13676302`(bf16 ours)正在 m13h 上运行,它们直接判决这一点**:
- 若 bf16 档在本管线下仍给出 `ours − nocorr ≈ +8.7` → **噪声假说成立**,E3 的 dose–response 有了第一段真实斜率;
- 若 bf16 档在本管线下也只给出 ≈ +1~2 → **管线假说成立**,上面的"收缩"是伪像,
  且既有 bf16 / fp8_e5m2 两行一并作废。

**在它们落地之前,不得把 R45 的区间不相交解读为"CIS 在噪声下失效"。**
这一条现在写下,同样不得事后修改。

#### 三项必须同时声明的混杂
1. **单 seed**。cell A 上 `ours` 的种子 SD 0.93、`nocorr` 0.96,各约 ±1 分。
2. **档内存在硬件差**:`ours` 跑在 cs-2-1(H100),`nocorr`/`fullis` 跑在 m13h-1-2(H200)(R28)。
3. 上述比较只用了**旧协议 GSM8K 单一 benchmark**(n=1319),
   五 benchmark 口径的 `dose5_fp8_e4m3_ours` 正在评测中,届时可在主表等权刻度上复核。

### R46. 第三个 checkpoint 推翻 R38 的"常数"说;同时在五 benchmark 口径上印证 R45

#### 一、R38 的"极差 0.07、可复现的常数"是错的

R38 依据**两个** checkpoint(+6.06 / +5.99)断言协议差是"可复现的常数"。第三个推翻了它:

| 方法 | 旧协议 | `gsm8k_em` | `gsm8k`(mv) | prompt 效应 | 判分效应 | **合计** |
|---|---|---|---|---|---|---|
| `nocorr` | 62.17 | 65.50 | 68.23 | +3.33 | +2.73 | +6.06 |
| `fullis` | 60.20 | 64.06 | 66.19 | +3.86 | +2.13 | +5.99 |
| **`ours`** | 63.61 | 67.70 | 68.31 | **+4.09** | **+0.61** | **+4.70** |

| | 均值 | 极差 | SD |
|---|---|---|---|
| 前两个(R38 所依据) | +6.02 | **0.07** | — |
| 三个全部 | **+5.58** | **1.36** | 0.77 |

**离散几乎全部来自判分效应**:prompt 效应 +3.33/+3.86/+4.09(极差 0.76,稳);
判分效应 +2.73/+2.13/**+0.61**(极差 **2.12**)。
R38 里我写过"合计稳、分解不稳,可引用总差不要引用某一半"——**方向说反了**:
不稳的那一半(判分)的离散最终把合计也带得不稳了,只是在前两个样本上恰好互相抵消。

**但核心解释反而更强**:仓库待解释的差是 **+5.66**;
两点均值 6.02 的残差是 0.36,**三点均值 5.58 的残差只有 0.08**。
即:协议差解释仓库那 5.66 分**这个结论更牢固了**,被推翻的只是"它是个常数"。

**可复用的判据**:**n=2 的一致不构成可复现性**,它只是"还没见到差异"。
两点必然落在某条直线上、必然有一个极差,而那个极差本身没有自由度去暴露离散。
R33 当时要求"第二个 checkpoint 才能坐实"是对的,但**门槛定低了**;
真正该问的是"离散有多大",而这至少需要三点。

#### 二、`ours` 的判分效应异常小,反过来强化了 R45

`ours` 的判分效应只有 +0.61,而 nocorr 是 +2.73。判分效应度量的是
**末位数字正则相对 math-verify 的漏判量**——它小,说明 `ours` 的输出更"正则友好"。

**推论:旧协议偏袒 `ours`。** 换用更准的 math-verify 后:

| GSM8K,ours − nocorr | 差 |
|---|---|
| 旧协议(末位数字正则) | **+1.44** |
| 新协议(math-verify) | **+0.08** |

**R45 报告的 +1.44,有约 1.4 分是判分器造成的假象。** 用更准的判分器看,
CIS 与不修正在 fp8_e4m3 下**几乎完全相同**(+0.08)。这使 R45 的结论**更强而非更弱**。

#### 三、五 benchmark 等权口径独立印证 R45

| fp8_e4m3 档 | 等权五 benchmark |
|---|---|
| `nocorr` | 34.14 |
| **`ours`** | **33.95** |
| `fullis` | 32.56 |

`ours − nocorr = **−0.19**`。**事先算出的预测区间是 [33.90, 35.00],实测 33.95 落在其中**;
也落在预登记的"33.5–35.5 → 与 R45 一致"带内。
**两种评测口径、两种判分器,一致给出"CIS 在 fp8_e4m3 下相对不修正没有可测优势"。**
R45 的结论不是 benchmark 选择或判分器造成的 —— 这三种可能已逐一排除。

**未被触动的仍然只有一个问题**:这是噪声削弱了 CIS,还是 dose 管线抬高了 nocorr?
仍由 m13h 上的 bf16 三臂判决(R45 第二节)。

### R47. 降 `mem_fraction_static` **不创造余量,只是把余量交给分配器吃掉** —— 我的"更保守"判断是反的

`13675697`(fp4 / fullis)在从未实测过的 `mem_fraction_static=0.55` 下**跑通首步、零 OOM**。
但它报出的显存数字比"跑通"这个结论更重要。**以下两行直接取自原始日志,不是 NOTES 转述**:

| 臂 | `mem_fraction` | allocated | reserved | device used/total | **余量** |
|---|---|---|---|---|---|
| `nocorr`(13675244) | **0.6** | 53.41 | 72.61 | 77.34/79.18 | **1.84 GB** |
| `fullis`(13675697) | **0.55** | 53.41 | **74.16** | **78.89**/79.18 | **0.29 GB** |

**`allocated` 两者完全相同(53.41)** —— 真实张量需求一模一样;
变的只有缓存分配器的 `reserved`(+1.55 GB),而余量恰好少了 1.55 GB(1.84 → 0.29)。**分毫对得上。**

**机制**:`mem_fraction_static` 是**推理引擎的静态占比**。把它调低,腾出的显存并没有变成"安全余量",
而是立刻被训练侧 PyTorch 的**缓存分配器**吸收(它按可用量膨胀,而非按需求)。
于是**总占用反而更逼近上限**。

**这推翻了我在 R26 里的处置理由。** 当时我写"`ours` 臂要额外运行 σ-TIS,开销更高,
故补提的两臂改用更保守的 **0.55**"——**方向是反的**:0.55 比 0.6 **更危险**,余量只有它的六分之一。

**同时,这给 R42 抢救下来的那条规则提供了机制层面的证明**(此前只是论证,现在是实测):

> 若仍 OOM,应降 `actor.mb_spec.max_tokens_per_mb`,**而不是**继续降 `mem_fraction_static`。

理由现在很具体:降 `mem_fraction` 只是**重新分配**余量,分配器会把腾出的部分吃掉,
继续降只会让总占用更贴近上限,直到从"训练 OOM"翻转成"推理引擎起不来"(R15/R17 在 A100 上见过的形态)。
**只有降 `max_tokens_per_mb` 才真正降低训练侧的峰值需求**,即缩小 `allocated` 本身(53.41 这个数)。

**处置**:`13675698`(fp4 / `ours`,仍 PENDING @ 0.55,且要多跑 σ-TIS)**已取消并按 0.6 重投**。
代价不对称:现在重投只丢一个队列位次;若 OOM 则同样丢位次,还要多赔一次调度与 R4 的残留清理。
**`13675697` 维持 0.55 不动** —— 它已跑通且在推进,为参数一致性打断一个 6.5 小时的运行不划算;
该参数只影响显存调度,不涉及算子、训练超参或 `kv_cache_dtype`,不改变实验语义(承 R26)。

**可复用的判据**:在**推理与训练 colocate** 的框架里,
**分配比例参数重新分配余量,不产生余量**;被让出的部分会被缓存分配器立即占据。
要真正制造安全边际,必须**降低峰值需求**,而不是调整分配比例。
**判断一个参数是"缓解"还是"转移",看 `allocated` 变不变**:它不变,就只是转移。

### R48. 一个可复现的算子级现象:`ours` 比 `nocorr`/`fullis` 跑得明显更快

先记下来,以免将来被误读成故障信号:

| 档位 | `nocorr` | `fullis` | `ours` |
|---|---|---|---|
| fp8_e4m3(总用时) | 6:36:52 | 6:41:56 | **4:55:23** |
| bf16(同一时刻步频) | 5.2 分/步 | 5.2 分/步 | **3.6 分/步** |

σ-TIS 额外做裁剪计算,直觉上 `ours` 应当**更慢**,实际**快 25–30%**,且**在两个档位上都成立**。
最可能的原因是裁剪改变了训练动态进而改变 rollout 的生成长度(rollout 占单步耗时的大头),
而非计算量差异。**这是算子的真实性质,不是故障。**
实际影响:bf16 档的 `ours` 预计约 08:44 完成,而 `nocorr`/`fullis` 约 11:07 ——
**判决需要 `ours − nocorr` 这一对,故能下结论的时刻由较慢的那个决定,不是 `ours` 的完成时刻。**
(⚠ 此处原写"约 08:44 / 约 11:07",两个数都偏晚,订正见 R51:它们用 `elapsed/steps` 算,
而 elapsed 含约 17 分钟启动加载。按边际步频重算为 `ours` **08:07**、`nocorr` **09:36**。)

### R49. 预先登记 bf16 判决的**测量工具与阈值**(在三臂落地之前)

R45 留下的唯一悬案是:fp8_e4m3 下 CIS 优势的消失,是**噪声削弱**还是**管线抬高了 nocorr**。
判决依据是 bf16 档的 `ours − nocorr`。此处先定死**用什么量**和**怎么读**。

#### 一、不能用自动触发的旧协议评测做这个判决

bf16 三臂完成后,`dose.sbatch` 自动接的是 `eval_tp.sbatch → eval_gsm8k.py`(**旧协议**)。
但 R46 已实测:旧协议的末位数字正则**对不同 checkpoint 的低估程度不同**
(判分效应 `nocorr` +2.73 / `fullis` +2.13 / `ours` **+0.61**),
它**系统性偏袒输出格式规整的模型**。同一对比在旧协议下是 +1.44,换 math-verify 后只剩 +0.08。

**跨算子比较必须用 math-verify。** 因此 bf16 的 `nocorr` 与 `ours` 落地后,
我会像 fp8_e4m3 那样**补投 `eval_suite.sbatch`**(tag `dose5_bf16_*`),用它做判决;
自动链产出的旧协议数字只用于 R43(**单个算子**的管线位移判定 —— 那里历史数据同为旧协议,同尺可比,不受此偏倚影响)。

#### 二、参照值与阈值(均从 `eval_suite.tsv` 逐 seed 重算,非转述)

| 历史管线 bf16(cell A,3 seed) | `ours − nocorr` | 种子 SE | 95% CI |
|---|---|---|---|
| 五 benchmark 等权 | **+3.48** | 0.30 | [+2.88, +4.08] |
| GSM8K(math-verify) | **+6.67** | 1.16 | [+4.40, +8.95] |

| 本管线单 seed 的测量精度 | SE_diff | 0.8 power 门槛 |
|---|---|---|
| 五 benchmark 等权 | 1.13pp | 3.17pp |
| GSM8K(math-verify) | 1.81pp | 5.07pp |

已测的 fp8_e4m3(本管线):五 benchmark **−0.19**,GSM8K(mv) **+0.08**。

#### 三、预先登记的判读规则

两个假设相距 3.48/1.13 ≈ **3.1 个 SE**(五 benchmark 刻度),单 seed 足以区分但**不是决定性的**。
取中点为界:

| bf16 `ours − nocorr`(五 benchmark) | 判读 |
|---|---|
| **≥ +1.74** | **噪声假说**:CIS 的优势在 bf16 下存在、在 fp8_e4m3 下消失 → E3 拿到第一段真实斜率 |
| **< +1.74** | **管线假说**:本管线下 bf16 也没有优势 → R45/R46 的"收缩"是伪像,既有 bf16 / fp8_e5m2 两行作废 |

GSM8K(math-verify)刻度以 **+3.34** 为同一中点,**两把尺子必须给出同向结论**;
若方向相反,则记为**判决失败**,需补种子而不是择一采信。

**同时声明三条限制**,避免事后把它读得过强:
1. **单 seed**。历史参照的种子 SD 是 `ours` 0.48 / `nocorr` 0.22(五 benchmark 刻度),
   本管线各臂只有一个 seed,无法估计自身的种子方差。
2. **本判决只覆盖 cell A**。dsv2 上 CIS 本就未复现 cell A 的优势(见前文),
   此处无论哪个假说成立,都不改变那个结论。
3. **两个假说并非互斥**。若观测值落在 +1.74 附近,更可能是两种效应各占一部分,
   届时应记为"无法分解",而不是强行归因给其中之一。

**本规则在 bf16 三臂落地前写下,不得事后修改。**

### R50. E3 的答案对象已脚本化(`e3_answer.sh`),并订正 R49 里一个附带数字

#### 一、为什么要脚本化

E3 的交付物是"CIS 相对基线的优势**随注入噪声档位如何变化**"。此前我有两个脚本
(`dose_curve.sh` 出旧协议 GSM8K 曲线、`dose5_curve.sh` 出五 benchmark 表),
**但没有任何脚本直接产出这个答案对象**;更要紧的是,R49 的判读规则只写在 NOTES 里,
等数据到达时由我手工套用 —— **这正是事后调整的风险敞口**。

`env_local/e3_answer.sh`(不在仓库内)把三条规则写死为机械执行:
1. **跨算子比较一律用 math-verify / 五 benchmark 等权**,旧协议值降为附注
   (R46:旧协议正则对不同 checkpoint 低估程度不同,同一对比 +1.44 vs +0.08);
2. 五 benchmark 均值**等权**,且 `gsm8k_em` **不计入**(同批题的第二种判分,计入即双倍权重);
3. bf16 档自动套用 **R49 预先登记的判决**,中点阈值 **+1.74** 写死在脚本里,不随数据调整;
   并在观测值距边界不足 1 个 SE 时**主动打印"应记为无法分解"**的提示。

#### 二、在决胜数据到达之前就验证它(而不是到时候第一次运行)

按 R41 的教训,**未经运行的工具与坏掉的工具无法区分**。此刻恰好有已知正确答案可对照,
是验证成本最低的时刻;若拖到 11:07 判决时首次运行,一个 bug 会正好卡在最需要它的节点。

| 对照项 | 手算已知值 | 脚本输出 | |
|---|---|---|---|
| fp8_e4m3 `ours−nocorr` | −0.19 | **−0.19** | ✓ |
| fp8_e4m3 `ours−fullis` | +1.39 | **+1.39** | ✓ |
| 历史管线参照 | +3.48(SE 0.30,CI [+2.88,+4.08]) | **同左** | ✓ |
| bf16 判决 | 数据未齐 | **正确报告未齐** | ✓ |

#### 三、订正 R49 的"门槛 3.17pp"

脚本给出该档 SE_diff = **1.10**、门槛 **3.09**,而 R49 写的是 1.13 / 3.17。差异已查明:

- R49 的算法:用 `nocorr` 的比例估**两个臂**的方差,再乘 √2 → `√2 × 0.80 = 1.13`;
- 脚本的算法:对**每个臂用其自身比例**分别算方差再合成 → `√(0.755² + 0.80²) = 1.10`。

`ours` 在 math500 / minerva / olympiad 上的比例更低,而 `p < 0.5` 时 `p(1−p)` 随 p 减小而减小,
故 `se5(ours) = 0.755 < se5(nocorr) = 0.80`。**脚本的算法正确,R49 那个数是偏保守的近似。**

**对 R49 的判决边界没有影响**:+1.74 是 0 与 +3.48 的中点,与该 SE 无关;
受影响的只是 R49 表格里"0.8 power 门槛"这个**附带说明数字**,
且它本就随各档实测比例而变(每一档会略有不同),不应被当成固定常数引用。
**预登记的判决规则本身不变,继续有效。**

### R51. `elapsed / steps` 会系统性低估吞吐 —— 我此前所有完成时刻推算都偏晚

**现象**:06:39 时 `13676300`(bf16/nocorr)已到 step 40/87,而我在 05:13 报的完成时刻是 11:07。
按实际推进倒推,真实速率远快于我的推算。

**根因**:我用 `squeue` 的 `elapsed` 除以已完成步数当作步频。但 `elapsed` **包含启动段**——
模型加载、sglang 引擎初始化、首次权重同步,实测约 **17 分钟**(对照:`13675244` 从
`00:36:32` 启动到 `00:53:11` 打出首步)。这段固定开销被平摊进每一步,
**步数越少、失真越大**,而我恰恰是在步数还少的时候做的推算。

**改用日志时间戳算边际速率**(取最近 10 步的时间戳差):

| 作业 | 步数 | 平均(含启动) | **边际** | 修正后完成时刻 |
|---|---|---|---|---|
| `13676302` bf16/ours | 58/87 | 3.19 | **3.01** | **08:07** |
| `13676300` bf16/nocorr | 40/87 | 4.62 | **3.77** | **09:36** |
| `13676301` bf16/fullis | 38/87 | 4.87 | **4.76** | 10:32 |
| `13675244` fp4/nocorr | 78/87 | 4.66 | **4.72** | 07:22 |
| `13675697` fp4/fullis | 38/87 | 3.55 | **3.69** | 09:40 |

**对计划的实际影响**:R45/R49 的判决需要 `ours − nocorr` 配对,
故判决点由 `nocorr` 决定 —— 从我先前反复引用的 **11:07 提前到 09:36**,约早 1.5 小时。
R48 中的"08:44 / 11:07"已就地订正。

**可复用的判据**:**带固定启动开销的作业,平均速率不等于边际速率。**
要预测完成时刻必须用**最近若干步的时间戳差**,而不是 `elapsed / steps`;
后者在作业早期会显著偏悲观,且偏差随进度自行收敛——这使它**看起来在"变准"**,
容易被误读为"作业在加速"。上表中 `13675244`(已 78/87)的平均与边际已几乎相等,
正是这种收敛的表现。

### R52. 「epoch 目录存在」≠「checkpoint 可用」—— 逐文件校验拦下一次错误提交

`13675244`(fp4/nocorr)的守卫报出 `EPOCH3 第三个 epoch 已落盘`。按 `13671277` 的先例,
我随即准备对该 checkpoint 提交五 benchmark 复评(checkpoint 一旦在盘,评测就不必等训练作业干净收尾)。
**逐文件字节校验把它拦下了**:

```
--- 参照(已知完好的 fp8_e4m3 nocorr)---
  chat_template.jinja 328 / config.json 1581 / generation_config.json 242
  model.safetensors 28632153464 / tokenizer_config.json 417 / tokenizer.json 11418262
--- 新(nocorr-fp4_e2m1-lowmem/epoch2epochstep28globalstep86)---
  (空 —— 0 个文件)
```

**目录已创建,里面一个文件都没有。** AReaL 是**先建目录、再写 27G 权重**,
而 cs-2-2 上并发 5 个作业争 I/O,写入明显更慢。
若没有这道校验,就会对着一个空目录提交 4 卡 A100 的评测。

**语义边界必须讲清楚**:我这套收尾守卫(`bx9xg1w9e` / `brqyl000n`)的 `EPOCH3` 判据是
`ls "$CK" | grep -c '^epoch' >= 3`,它断言的是**训练已进入第三个 epoch 并建了目录**,
**不是** checkpoint 已可用。`13671277` 那次没出问题是**运气**——我在数十秒后检查,文件恰好已写完。

**受影响与不受影响的分别**:
- `dose.sbatch` 自带的评测链**不受影响**:它在 python 进程退出后才执行,那时写入必然已完成;
- **我手工提交的评测受影响**,必须自己把关。

**判据**:判断"产物可用"不能看**目录是否存在**,要看**内容是否与已知完好的同类产物逐文件一致**。
同模型同 dtype 的 checkpoint,各文件字节数应完全相同 —— 这给了一个**廉价且决定性**的完整性判据,
比"大小接近 27G"或"文件数为 6"都强(两者在写入过程中都可能短暂成立)。

**处置**:已挂一个后台门控,每分钟比对一次逐文件字节数,
**写完整即自动提交**(并在训练作业已离队却仍不完整时报警),不再由我轮询。

### R53. fp4_e2m1 + 不修正 = 模型被训坏(比不训练还差 16 分),且它推翻了"协议差"的适用范围

#### 一、崩溃本身:四条独立证据

`13675244`(fp4_e2m1 / `nocorr`)训练完成 87/87、epoch 3/3,held-out 结果:

| benchmark | fp4/nocorr | 对照 fp8_e4m3/nocorr | 对照 base(未训练) |
|---|---|---|---|
| gsm8k(mv) | **15.77** | 68.23 | 55.50 |
| math500 | **7.60** | 25.80 | 21.20 |
| svamp | **36.00** | 65.00 | 59.00 |
| minerva | **2.21** | 6.62 | 4.04 |
| olympiad | **2.08** | 5.04 | 4.90 |
| **等权均值** | **12.73** | 34.14 | **28.93** |

**比完全不训练还低 16.20 分。** 这不是"没有提升",是**方向性的损害**。

四条证据互相独立、指向一致:
1. **五个 benchmark 全线低于 base**,不是单一 benchmark 的偶然;
2. **训练奖励单调塌陷**:0.648 → 0.566 → 0.417 → 0.249 → 0.141 → 0.085 → 终值 **0.06**
   (对照 fp8_e4m3/nocorr 同期 0.698 → 0.850 单调**上升**);
3. **评测端零 math-verify 超时**(`13677043` 计数 0;fp8_e4m3 三次评测为 0/2/0)——
   **排除了"判分器超时压低分数"这一量具解释**;
4. **样例输出是"流畅但算错"**,不是乱码:
   `"Janet's ducks lay 16 eggs per day. So, Janet has 16-3=13 eggs left. So, Janet makes 13*2=26$"`
   (应为 16−3−4=9)。**丢的是算术能力,不是语言能力。**

#### 二、`Timeout during comparison`:退化输出的伴随标记,兼新的挂起签名

该串出自 **`math_verify/grader.py`**(训练期算奖励用的判分库,不是 AReaL),
在本臂日志中出现 **12 次**,**其余所有 dose 作业为 0 次**。

含义:退化输出让判分器解析超时 → 该条奖励按失败计 → **训练奖励被进一步压低**。
它既是崩溃的**伴随标记**,也可作为**早期预警信号**(单臂独有、其他臂为零)。
好在 held-out 的 `gsm8k_em` 用末位数字正则、与该库无关,**崩溃结论不依赖它**。

**同时这是一个新的挂起签名**。前文记录挂起"无任何错误输出",本次不然:
收尾挂起前的最后一行正是 `Timeout during comparison`,且发生在最后一次权重重载
(210.91s)之后。**这次挂起很可能是崩溃诱发的,而非那种通用的收尾挂起。**
处置:三项条件核对齐备(RUNNING + epoch 3/3 + 静默 754s)后人工补投评测(`13677051`)并取消作业,
**释放的 cs-2-2 立即被 `13676652`(fp4/ours)接手** —— 取消挂起作业有直接的调度收益。

#### 三、最重要的方法论后果:协议差对崩溃模型**连符号都相反**

同一个 checkpoint,**同一判分器**(末位数字正则),只换 prompt:

| checkpoint | 旧协议(裸 prompt) | 新协议 `gsm8k_em`(+`\boxed`) | prompt 效应 |
|---|---|---|---|
| fp8_e4m3 / nocorr | 62.17 | 65.50 | **+3.33** |
| fp8_e4m3 / fullis | 60.20 | 64.06 | **+3.86** |
| fp8_e4m3 / ours | 63.61 | 67.70 | **+4.09** |
| base(未训练) | 52.31 | 53.60 | **+1.29** |
| **fp4 / nocorr(崩溃)** | **36.09** | **13.65** | **−22.44** |

**健康模型一律小幅加分,崩溃模型重挫 22 分。**
机制上讲得通:`\boxed{}` 是一条**格式指令**,模型要先能遵循指令才谈得上受益;
已丧失该能力的模型,被指令进一步干扰。

**对前文的约束**:R38→R46 用 +5.58 的协议差解释了仓库里 SIGMA_TIS 与主表的矛盾——
那涉及的全是健康模型,**结论不受影响**。但必须补一条边界:
**协议差换算绝不可施加于退化模型**;对它们,两种协议测的实际上是不同的东西。

**另记一个被查证拦下的假模式**:我一度想写"两种协议下跌幅都约 −16 分,互相印证"。
实际那是拿**旧协议的 GSM8K 跌幅**(52.31→36.09 = −16.22)去比**新协议的五 benchmark 均值跌幅**
(28.93→12.73 = −16.20)——**两者不是同一把尺**。同口径(GSM8K)对比是
**−16.22(旧) vs −39.73(新)**,相差 2.4 倍。数值巧合极具迷惑性,落笔前必须核对量纲。

#### 四、fp4 档尚未定论:`fullis` 目前没有崩

`13675697`(fp4/fullis)已跑 59 个奖励采样点,**最新 0.683、峰值 0.723,健康且在高位**;
而 `nocorr` 在约第 37 步后就已决定性转折。fullis **已越过那个转折点仍未崩**。
`13676652`(fp4/ours)刚启动。

**因此现在只能说"fp4 下不修正会崩",不能说"fp4 下所有方法都崩"。**
E3 在这一档的核心问题——**CIS 能否在这个噪声水平下存活**——要等 `13676652`。
奖励崩溃监控(`bdomvewcv`)已覆盖这两臂,阈值为跌破峰值 50% 或低于 0.30。

### R54. 预先算出 `dose5_bf16_ours` 的区间(写在最后两个 benchmark 到达之前)

`13677094` 已出三项:`math500 26.60`、`svamp 71.00`、`minerva 4.41`(小计 102.01),
另有 `gsm8k_em 69.52`。尚缺 `olympiad` 与 math-verify 口径的 `gsm8k`。

把缺的两项在合理范围内扫一遍(`gsm8k(mv)` 由 `em=69.52` 加判分效应推得,
前四例为 +2.73 / +2.13 / +0.61 / +2.12;`olympiad` 取 3.5–6.5):

**预测区间:`dose5_bf16_ours` 等权均值 ∈ [35.00, 36.16]。**

参照:历史管线 bf16 `ours` **34.47**、bf16 `nocorr` **30.99**;
本管线 fp8_e4m3 `ours` **33.95**、`nocorr` **34.14**;base **28.93**。

#### 一个在判决前就该记下的张力:管线位移**可能是随刻度变化的**

同一个 bf16 `ours` 臂:

| 刻度 | 本管线 | 历史管线 | 位移 |
|---|---|---|---|
| 旧协议 GSM8K | **68.99** | 64.29 | **+4.70** |
| 五 benchmark 等权 | 预测 ~35.4 | 34.47 | **约 +0.9** |

若最终落在预测区间内,则**同一个管线差异在两把尺子上相差约 5 倍**。
这对 **R43** 有直接影响:R43 用**旧协议 GSM8K** 判定"本管线是否与历史同尺",
其三个区间(54–57 / 57–60 / 60–64)是在那把尺子上定的。
**若位移本身依赖刻度,那么"管线是否同尺"就不是一个二元问题**,
而要分刻度回答;R43 的判定结论只对旧协议 GSM8K 成立,不能自动推广到五 benchmark 刻度。

**但这仍然不是判决。** R49 的判决量是 **bf16 的 `ours − nocorr`**,
需要 `13676300`(约 09:36)。`ours` 单独一个数在任何刻度上都区分不了噪声假说与管线假说——
因为两个假说对 `ours` 的预测几乎相同,差异全在 `nocorr` 上。

**本条写于最后两个 benchmark 落地之前,区间不得事后调整。**

### R55. `dose5_bf16_ours = 35.77`:R54 区间命中;管线位移确认随刻度变化;并订正协议差的用法

#### 一、R54 的预登记区间命中

| benchmark | 值 |
|---|---|
| gsm8k(mv) | 71.80 |
| math500 | 26.60 |
| svamp | 71.00 |
| minerva | 4.41 |
| olympiad | 5.04 |
| **等权均值** | **35.77** |

R54 在最后两项到达前算出的区间是 **[35.00, 36.16]**,实测 **35.77**,**命中**。

#### 二、管线位移确实随刻度变化(R54 的猜测得证)

同一个 bf16 / `ours` 臂,与历史管线相比:

| 刻度 | 本管线 | 历史管线 | 位移 |
|---|---|---|---|
| 旧协议 GSM8K | 68.99 | 64.29 | **+4.70** |
| 五 benchmark 等权 | 35.77 | 34.47 | **+1.30** |

**两把尺子上的位移相差 3.6 倍。**

**对 R43 的约束(重申并落实)**:R43 用**旧协议 GSM8K** 判定"本管线是否与历史同尺",
其三个区间是在那把尺子上定的。既然位移本身依赖刻度,
**R43 的判定结论只对旧协议 GSM8K 成立,不能推广到五 benchmark 刻度**。
需要在五 benchmark 刻度上作同类判定时,必须另行设定阈值,而不是沿用 R43 的。

#### 三、协议差的离散继续扩大 —— 并且我用错了统计量

五个**健康** checkpoint 的协议总差:

| checkpoint | 总差 |
|---|---|
| fp8_e4m3 / nocorr | +6.06 |
| fp8_e4m3 / fullis | +5.99 |
| fp8_e4m3 / ours | +4.70 |
| base(未训练) | +3.19 |
| **bf16 / ours** | **+2.81** |

**均值 +4.55,极差 3.25,SD 1.52**(R46 时基于 3 点为均值 +5.58、极差 1.36)。
样本每增加一个,离散就扩大一截 —— 这是"它不是常数"的第三次确认。

**更要紧的是我此前用错了统计量。** R38/R46/R53 都在用**跨 checkpoint 的均值**
去解释仓库里那 5.66 分的矛盾。但那个矛盾是
`SIGMA_TIS 的 nocorr(55.55)` 对 `主表的 suite_nocorr(61.21)` ——
**一个 nocorr 对 nocorr 的比较**:

| 用哪个偏移 | 残差 |
|---|---|
| 五点大均值 4.55 | 1.11 |
| **nocorr 自身的偏移 +6.06** | **0.40** |

**协议差是随 checkpoint 变化的条件量,不是总体常数;
做同类 checkpoint 之间的换算时,应当用该类自己的偏移,而不是跨异质样本的平均。**
把条件量当总体量用,会在样本变得更异质时**让残差莫名其妙地变大**——
我正是这样先得到 0.08(三点)、又得到 1.11(五点)的。
**结论不变**(协议差确实解释了那个矛盾),**但理由要用对的那个数**。

#### 四、判决仍未到

本管线 bf16 `ours` = **35.77**(五 bench)/ **68.99**(旧协议)。
R49 的判决量是 **`ours − nocorr`**,另一半来自 `13676300`(约 09:36)。
**两个假说对 `ours` 的预测几乎相同,差异全在 `nocorr` 上** —— 故此处不作任何判读。

### R56. 预先算出 `dose5_fp4_e2m1_fullis` 的区间(写在最后两项到达之前)

已出三项(计入均值):`math500 21.00`、`svamp 65.33`、`minerva 5.15`(小计 91.48);
另有 `gsm8k_em 64.82`(不计入均值)与旧协议 `60.35`。

缺 `gsm8k(mv)` 与 `olympiad`。二者均有五个先例可约束:
判分效应 +0.61 ~ +2.73(故 `gsm8k(mv)` ≈ 65.4–67.6),`olympiad` 实测 2.08–5.04。

**预测区间:`dose5_fp4_e2m1_fullis` 等权均值 ∈ [31.88, 32.91]。**

区间只有 1.03 分宽,因为三项已知、两项被先例夹紧。参照:
fp8_e4m3 / `fullis` **32.56**、历史 bf16 / `fullis` **31.86**、base **28.93**、fp4 / `nocorr` **12.73**。

#### 预先登记的三分支判读

| 若落在 | 判读 |
|---|---|
| **30–34** | `fullis` 对 fp4 噪声**基本免疫**(与 fp8_e4m3 的 32.56 相差 ±2 内),与逐 benchmark 观察一致;**fp4 档"修正 vs 不修正"的差距 ≈ 18–21 分** |
| 25–30 | 有可测损伤但远未崩溃;阈值效应成立,程度减弱 |
| **< 20** | 与逐 benchmark 观察**矛盾**(math500 / svamp / minerva 三项均正常)。此时应**优先怀疑剩余两项异常或我的计算有误**,而不是推翻免疫性结论 |

最后一条是刻意写的:三个已知 benchmark 都正常时,若均值突然极低,**更可能是我算错或某项异常,而非模型真的崩了**。
预先声明这一点,是为了防止自己在看到反常数字时倒过来修改解释。

**本条写于最后两项落地之前,区间与判读不得事后调整。**

#### 它为什么重要

fp4 是 E3 唯一出现**大效应量**的档位。bf16 与 fp8_e4m3 上各算子统计不可分(跨度 ~3 分,门槛 5.32),
而 fp4 上 `nocorr` 已崩至 12.73。若 `fullis` 落在 30–34,则该档"修正与否"相差约 20 分 ——
**远超一切噪声下限**,是 E3 目前最坚实的一个结论:
**"失配修正有没有用"取决于失配有多大;在低噪声下无法分辨,在高噪声下决定成败。**

### R57. fp4 档 `fullis = 32.32`(R56 区间命中):剂量-反应是**阶跃函数**,不是斜坡

#### 一、R56 的预登记区间与分支判读双双命中

| benchmark | fullis | (对照 nocorr) |
|---|---|---|
| gsm8k(mv) | 65.20 | 15.77 |
| math500 | 21.00 | 7.60 |
| svamp | 65.33 | 36.00 |
| minerva | 5.15 | 2.21 |
| olympiad | 4.90 | 2.08 |
| **等权均值** | **32.32** | **12.73** |

R56 预登记区间 **[31.88, 32.91]** → 实测 **32.32**,**命中**;落在"**30–34 = 基本免疫**"分支。

#### 二、`fullis` 跨三个噪声档几乎不动

| 档位 | `fullis` 等权均值 |
|---|---|
| bf16(历史 3 seed) | 31.86 |
| fp8_e4m3(本管线) | 32.56 |
| fp4_e2m1(本管线) | **32.32** |

**跨三档极差 0.70 分。** 注入噪声从 bf16 一路加到 fp4(足以把 `nocorr` 从 34.14 打到 12.73),
而 exact-ratio 修正下的结果**几乎与噪声水平无关**。

#### 三、因此剂量-反应是**阶跃**,不是斜坡

同一个量(`fullis − nocorr`,五 benchmark 等权):

| 档位 | `fullis − nocorr` |
|---|---|
| bf16(历史) | **+0.87** |
| fp8_e4m3(本管线) | **−1.58** |
| **fp4_e2m1(本管线)** | **+19.58** |

前两档都在噪声里(门槛 3.17),第三档**跳出一个数量级**。
换到别的刻度量级不同但结论一致:旧协议 GSM8K **+24.26**,`gsm8k(mv)` **+49.43**。

**这是 E3 迄今最坚实的结论**,因为它是唯一效应量远超一切噪声下限的观测:
**"训推失配修正有没有用"取决于失配有多大 —— 低噪声下无法分辨,高噪声下决定成败。**
而且转折是**突变的**:fp8_e4m3 与 fp4 之间没有中间状态,
`nocorr` 不是"逐渐变差",而是从"与修正无异"直接跳到"比不训练还差 16 分"。

**给老师的一句话**:E3 问的是 dose–response,答案是**这条响应曲线高度非线性**;
若只在 bf16 / fp8 两档之间取点,会得出"修正无用"的错误结论。

#### 四、必须同时声明的限制

1. **本条只涉及 `fullis`(exact-ratio),不涉及 CIS。** `13676652`(fp4 / `ours`)仍在跑
   (约 12:35),**CIS 能否在同一噪声下存活尚无数据** —— 这才是 E3 真正要回答的那一问。
2. **单 seed**。但此处效应量(19.58)是种子噪声(`fullis` 的种子 SD 2.23)的 **8.8 倍**,
   与 fp8_e4m3 档那种 1~2 分的差异性质完全不同。
3. **bf16 行用的是历史管线的值**;本管线的 bf16 `fullis` 约 10:32 落地,届时可换成同管线对照。

#### 五、协议差:第六个样本,判分效应再创新低

`fp4/fullis` 的总差 **+4.85**(判分效应 **+0.38**,为迄今最小;此前最小为 +0.61)。
六个健康 checkpoint:`+6.06 / +5.99 / +4.70 / +3.19 / +2.81 / +4.85`,
**均值 +4.60,极差 3.25,SD 1.37**。样本增至六个,离散未收敛 —— 再次确认它不是常数。

### R58. 监控的**寿命**本身是一个覆盖参数 —— 差点让判决臂的评测无人提交

#### 险情

08:18 我挂了三臂合并门控 `b80aena2f`,循环上限 **90 次 × 60s**,随后在多轮汇报里
反复把它算作"已覆盖"。09:20 复核时才发现:

| | 时刻 |
|---|---|
| 门控 armed | 08:18 |
| **门控超时** | **约 09:48** |
| `13676300`(bf16/nocorr,**R49 判决臂**)ETA | **09:56** |
| `13676301`(bf16/fullis)ETA | **10:04** |

**门控会在判决臂完成前 8 分钟静默退出**,而两个 epoch2 目录当时都还没出现 ——
也就是说,**这两臂的五 benchmark 评测一个都不会被自动提交**,
而我会一直以为有覆盖,直到很久以后发现数据缺失。

#### 为什么会发生

我把"覆盖"当成了**布尔量**(挂了 = 覆盖了),但它其实是**区间量**:
监控只在 `[armed, armed + 寿命]` 这段时间内有效。
挂上时 fp4/fullis 即将完成,90 分钟看着绰绰有余;
但另外两臂的 ETA 后来因并发变化而后移,**超出了当初隐含假设的窗口**。

**armed 那一刻的 ETA 不等于到期时刻的 ETA。** 被监视对象的完成时间会漂移,
监控的到期时间却是写死的 —— 两者是独立变量,必须显式比较。

#### 判据(与 R41 / R52 同族,但维度不同)

- **R41**:监控的**选择器**可能匹配不到东西(扫零个文件);
- **R52**:监控的**信号**可能名不副实(epoch 目录存在 ≠ checkpoint 可用);
- **R58**:监控的**寿命**可能短于它要等的事件。

三者共同点:**都表现为"看起来有覆盖,实际没有",且都只在主动核对时才暴露**。
故**每挂一个有限寿命的监控,必须当场记下它的到期时刻,并与被监视事件的 ETA 相比**;
ETA 后移时要重新评估,而不是沿用当初的判断。

#### 处置

停掉 `b80aena2f`(避免与新门控在边界时刻**重复提交**、产生重复 tsv 行),
改挂 `bmt3a66e7`,寿命 **150 分钟**(覆盖到约 11:50,两臂 ETA 09:56 / 10:04,余量充裕),
并在 ARMED 行里显式打印"旧门控约 09:48 到期、两臂 ETA 09:56/10:04,故重挂"的理由。

`13676652`(fp4/ours)ETA **12:52**,**仍超出 150 分钟窗口,故此刻不挂** ——
这一次是明确判断过寿命不够才推迟,而不是默认它会被覆盖。

### R59. R43 的预登记判定揭晓:**本管线比历史高 +6.39,既有 bf16 / fp8_e5m2 两行作废**

`dose_bf16_nocorr = **61.94**`(817/1319,旧协议 GSM8K)。

R43 在数据到达前写死的三个区间:

| 区间 | 含义 | 是否命中 |
|---|---|---|
| 54–57 | 管线与历史同尺,既有两行可并入曲线 | — |
| **60–64** | **管线整体高出约 6–7 分,既有两行作废** | **✓ 61.94** |
| 57–60 | 无法判定,需看另两臂是否同向 | — |

历史管线 bf16 `nocorr` = 55.55,本管线 = 61.94,**位移 +6.39**。

#### 后果:六个继承来的数字不可再用

```
bf16     nocorr 55.55 / fullis 59.72 / ours 64.29   → 作废
fp8_e5m2 nocorr 54.36 / fullis 57.62 / ours 60.12   → 作废
```

它们来自 2026-08 的另一批运行,与本管线在旧协议刻度上差约 6.4 分,**并入曲线会制造虚假的剂量效应**。
这也回溯性地解释了 R40 的困惑:本管线 `dose5_nocorr` 比主表高 3.15 分(14 个种子 SD)——
那不是量化的功劳,是管线差异。

**E3 的曲线因此必须自给自足。** 好消息是三档我都自己跑了:

| 档位(旧协议 GSM8K) | nocorr | fullis | ours |
|---|---|---|---|
| bf16 | **61.94** | 待测(约 10:04) | **68.99** |
| fp8_e4m3 | 62.17 | 60.20 | 63.61 |
| fp4_e2m1 | 36.09 | 60.35 | 待测(约 12:52) |

#### 一处必须限定的范围:**"作废"是针对旧协议刻度的**

R55 已实测到管线位移**随刻度变化**:同一个 bf16 `ours` 臂,
旧协议 GSM8K 上位移 **+4.70**,五 benchmark 上仅 **+1.30**。
R43 的区间是在**旧协议刻度**上设定的,故其结论严格来说是
"**在旧协议 GSM8K 上**,本管线与历史不同尺"。
五 benchmark 刻度上两管线可能接近得多 —— 本管线 bf16 `nocorr` 的五 benchmark 值即将落地,
届时可直接比对历史的 30.99。**在那之前,不要把"作废"推广到五 benchmark 刻度。**

#### 不在此处下的结论

bf16 档旧协议配对现已可算:`ours − nocorr = 68.99 − 61.94 = **+7.05**`,
而本管线 fp8_e4m3 同一量是 **+1.44**。这看起来支持噪声假说,
**但 R49 明确规定判决用 math-verify / 五 benchmark 刻度**(因 R46 证明旧协议判分器
对不同 checkpoint 偏袒程度不同,差异可达 2 分以上)。
**故此处不作判决**,等 `dose5_bf16_nocorr` 齐全后由 `e3_answer.sh` 按写死的阈值机械输出。

### R60. 把 R49 的判决边界翻译成对待测数字的阈值(写在最后两项到达之前)

`dose5_bf16_nocorr` 已出三项(计入均值):`math500 23.60`、`svamp 71.00`、`minerva 5.88`
(小计 100.48),另有 `gsm8k_em 64.97`。缺 `gsm8k(mv)` 与 `olympiad`。

判分效应已观测六例(+2.73 / +2.13 / +0.61 / +2.12 / +2.28 / +0.38)→ `gsm8k(mv)` ≈ 65.35–67.70;
`olympiad` 已观测 2.08–5.04。

**预测区间:`dose5_bf16_nocorr` 等权均值 ∈ [33.67, 34.74]。**

#### 判决阈值(直接写成对待测数字的不等式)

已知本管线 bf16 `ours`:五 benchmark **35.77**、`gsm8k(mv)` **71.80**。
代入 R49 写死的两个中点:

| 尺子 | 中点 | 判据 |
|---|---|---|
| 五 benchmark 等权 | +1.74 | `nocorr` **< 34.03** → 噪声假说;**> 34.03** → 管线假说 |
| `gsm8k` math-verify | +3.34 | `nocorr` **< 68.46** → 噪声假说;**> 68.46** → 管线假说 |

#### 必须在数据前声明的缺陷:两把尺子的信息量**不对等**

- **尺子一(五 benchmark)**:预测区间 **[33.67, 34.74] 跨越阈值 34.03**。
  → 这把尺子的判读**真正取决于实测值**,是一个有风险的检验。
- **尺子二(`gsm8k` math-verify)**:预测范围 **[65.35, 67.70] 整体低于阈值 68.46**。
  → 这把尺子**几乎必然判"噪声假说"**,无论真相如何。

**这是 R49 设计上的一个缺陷,我现在才发现,并在看到数据前记下。**
R49 要求"两把尺子同向"作为稳健性检查,但若其中一把在事前就已被锁定,
**它的"同意"不构成独立证据** —— 只有尺子一的结论是有信息量的。
因此:

1. 若两把尺子都判噪声假说,**权重应几乎全部落在尺子一**上,不可宣称"两把尺子互相印证";
2. 若尺子一判管线假说而尺子二判噪声假说,按 R49 属**判决失败**,需补种子;
   但考虑到尺子二事前已被锁定,这种不一致**更可能反映尺子二无效,而非真正的矛盾**;
3. 无论结果如何,**单 seed 的判决只是初步的**,三档九臂都是单 seed。

#### 顺带可检验的一项

历史管线 bf16 `nocorr` 的五 benchmark 值是 **30.99**。若本管线落在 [33.67, 34.74],
则该刻度上的管线位移为 **+2.68 ~ +3.75**,而旧协议刻度上是 **+6.39**。
这将是 R55/R59"位移随刻度变化"的第二个独立验证点
(第一个是 bf16 `ours`:旧协议 +4.70 vs 五 benchmark +1.30)。

**本条写于最后两项落地之前,阈值与上述限定不得事后调整。**

### R61. R49 判决揭晓:**登记结论是『无法分解』**;并订正我自己的三处错误

`dose5_bf16_nocorr` 五项齐全:`gsm8k(mv) 68.69 / math500 23.60 / svamp 71.00 / minerva 5.88 / olympiad 5.79`
→ **等权均值 34.99**。

#### 一、按 R49 写死的规则,判决是『无法分解』

| 尺子 | ours − nocorr | 中点 | 方向 | 距边界 |
|---|---|---|---|---|
| 五 benchmark 等权 | **+0.78** | +1.74 | 管线假说 | **0.89 个 SE** |
| `gsm8k` math-verify | **+3.11** | +3.34 | 管线假说 | **0.13 个 SE** |

两把尺子**方向一致**(都指向管线假说),但 R49 另有一条附加条款:
「**观测值距边界不足 1 个 SE,应记为『无法分解』,不得强行归因**」。
**两把尺子都触发了这一条**(0.89 与 0.13)。
`e3_answer.sh` 也机械地打印了这条警告。

**故登记判决是:无法在噪声假说与管线假说之间分解。** 不是"管线假说成立"。
我先前写下那条附加条款,正是为了防止此刻把一个擦边的结果读成定论。

#### 二、一个功效更高的事后检验(**非预登记**,须如此标注)

R49 的中点检验把两个假说的距离(3.48)一分为二,再拿 SE 1.09 去分辨 —— 功效本就勉强。
直接问"本管线的 bf16 优势是否显著低于历史管线"更有力:

| | ours − nocorr | SE |
|---|---|---|
| 本管线(单 seed) | **+0.78** | 1.09 |
| 历史管线(3 seed) | **+3.48** | 0.30 |

差 **2.70**,合成 SE **1.13**,**z = 2.39(显著)**。
**这支持管线假说**:本管线**在 bf16 上就没有复现历史的 CIS 优势**。

**但这是事后构造的检验,不在 R49 的登记范围内**,故只能作为佐证,不能替代登记判决。
两者结论方向一致(都指向管线),差别在于把握程度。

**另一个角度**:本管线两个低噪声档的 CIS 优势都与 0 不可分 ——
bf16 `+0.78`(z=0.72)、fp8_e4m3 `−0.19`(z=−0.17)。

#### 三、订正我自己的三处错误

**(1) R60 的预测区间失手 —— 四次预登记中的第一次。**
预测 `[33.67, 34.74]`,实测 **34.99**,偏高 0.25。

**(2) 失手原因:我把未收敛的经验范围当成了硬上界。**
我用六次观测到的判分效应范围(+0.38 ~ +2.73)去夹 `gsm8k(mv)`,
而第七例是 **+3.72**,超出我的上限 0.99。判分效应七例现为 **+0.38 ~ +3.72,极差 3.34**。

**这正是我自己在 R46 与 R55 里写过两次的错误**:
R46 说"n=2 的一致不构成可复现性,两点的极差没有自由度暴露离散";
R55 说"样本每增加一个,离散就扩大一截"。
**我记下了这条教训,却在下一次构造区间时照样用经验极差当硬边界。**
**判据:用经验范围做预测区间时,必须留出外推余量(如按 SD 而非极差),
尤其在样本量小且离散尚未收敛时** —— 否则区间会系统性偏窄。

**(3) R60 宣称"尺子二事前已被锁定"—— 这个说法是错的,错因同源。**
R60 据 `gsm8k(mv) ∈ [65.35, 67.70]` 全部低于阈值 68.46,断言尺子二必判噪声假说,
并据此声称 R49 的双尺子设计有缺陷。实测 **68.69 高于阈值**,尺子二实际判了**管线假说**。
**所以 R49 的双尺子设计并无我所说的那个缺陷;有缺陷的是我对其中一把尺子的范围估计。**
这一点必须讲清:**我先前对一份预登记方案提出的批评,本身建立在一个错误的边界上。**

#### 四、顺带:管线位移随刻度变化的第二个验证点

| 臂 | 旧协议 GSM8K | 五 benchmark | 比值 |
|---|---|---|---|
| bf16 / `ours` | +4.70 | +1.30 | 3.6× |
| bf16 / `nocorr` | **+6.39** | **+4.00** | **1.6×** |

**位移确实随刻度变化(R55/R59 的判断成立),但比值本身不稳定**(3.6× vs 1.6×),
故不能用一个固定系数在两把尺子之间换算。

#### 五、对 E3 的意义,以及为什么 fp4/ours 变得更重要

本管线在**两个低噪声档**上,CIS 相对不修正都**没有可测优势**;
历史管线 bf16 的 +3.48 未被复现(z=2.39 显著低于)。
结合 R57:**唯一出现大效应的档位是 fp4**(修正 vs 不修正相差 19.58 分)。

因此 `13676652`(fp4 / `ours`,约 12:52)**不是收尾,而是 E3 现在最关键的一个数**:
低噪声档已确认分辨不出算子;**若 CIS 在 fp4 下也像 `fullis` 那样存活,
则"修正有用"成立而"CIS 特别有用"仍无证据;若 CIS 在 fp4 下崩溃,则是对 CIS 的负面证据。**
两种结果都有明确含义,且效应量足够大到能被这套测量分辨。


### R62. 预算 `dose5_bf16_fullis` 区间 —— 首次改用 SD 外推(承 R61 的订正)

已出三项(计入均值):`math500 25.00`、`svamp 68.33`、`minerva 6.25`(小计 99.58);
另有 `gsm8k_em 68.92`、旧协议 `68.01`。缺 `gsm8k(mv)` 与 `olympiad`。

**这次改了构造方法。** R61 记录:我上一个区间失手,是因为把七例中前六例的经验极差
(+0.38 ~ +2.73)当成硬上界,而第七例 +3.72 超了出去。故本次按 **mean ± 2SD** 外推:

| 量 | 7 例 mean | SD | ±2SD 带 | 旧法(min/max) |
|---|---|---|---|---|
| 判分效应 | 2.00 | 1.17 | [-0.34, +4.33] | [+0.38, +3.72] |
| `olympiad` | 4.45 | 1.20 | [2.05, 6.85] | [2.08, 5.79] |

→ `gsm8k(mv)` 推定 **[68.58, 73.25]**

**预测区间:`dose5_bf16_fullis` 等权均值 ∈ [34.04, 35.94]**(宽度 1.89)。

**刻意比旧法更宽** —— 这是为承认离散尚未收敛而付出的代价:
七个样本的判分效应 SD 高达 1.17,用极差夹只会重复 R61 的错误。
**区间变宽会降低它的信息量,但避免了"看似精确实则会被下一个样本打穿"。**

#### 参照与意义

本管线 bf16:`ours` **35.77**、`nocorr` **34.99**;
`fullis` 在别档:fp8_e4m3 **32.56**、fp4 **32.32**;历史管线 bf16 `fullis` **31.86**。

若本值落在 34–36,则**本管线 bf16 档三臂(nocorr 34.99 / fullis ? / ours 35.77)互相接近**,
与 R61 的判读一致:**本管线在低噪声档分辨不出算子**。
若显著低于 34(如 ≤32),则 `fullis` 在本管线 bf16 上明显劣于另两臂,
那会与 fp8_e4m3 档(fullis 32.56 < nocorr 34.14)同向,提示 `fullis` 有系统性劣势而非噪声。

**本条写于最后两项落地之前,区间与判读不得事后调整。**

### R63. bf16 档收官(跨度 0.88 分);**算子间差异随量具改善而单调缩小**

`dose5_bf16_fullis = **34.89**`,R62 预测区间 [34.04, 35.94] **命中**。

#### 一、先说两条对自己的限定,免得把没赚到的分算进账

**(1) 本例无法证明 R62 的方法改进有效。** R62 依 R61 的订正改用 mean±2SD 外推,
本次判分效应实测 **+0.91** —— 它**同时落在**新法的 ±2SD 带 [−0.34, +4.33] **和**
旧法的 min/max 带 [+0.38, +3.72] 内。**两种构造法在本例上给出相同判定**,
故这次命中**不能算作新法优于旧法的证据**。要检验该改进,需等到一个落在两带之间的样本。

**(2) 本次命中对"低噪声档算子不可分"这一结论**不构成独立证据**。**
我在数据到达前已声明:R62 的区间 [34.04, 35.94] 几乎完整落在它自己
"34–36 → 三臂接近"那一分支内,故**落在区间内只是没有反驳,不是印证**。
34.89 确实落在区间内 —— 按事前声明,**此处只能说"未被推翻"**。

#### 二、bf16 档三臂:跨度 0.88 分

| bf16(五 benchmark 等权) | 值 |
|---|---|
| `nocorr` | 34.99 |
| `fullis` | **34.89** |
| `ours`(CIS) | 35.77 |

两两差:`ours−nocorr` **+0.78**、`ours−fullis` **+0.88**、`nocorr−fullis` **+0.10**。
`SE_diff ≈ 1.09`,0.8 power 门槛 **3.05** → **三者互相不可分**。

#### 三、一个值得单独指出的模式:**量具越好,算子差异越小**

同一个 bf16 档,三种量具给出的档内跨度:

| 量具 | nocorr | fullis | ours | **跨度** |
|---|---|---|---|---|
| 旧协议 GSM8K(裸 prompt + 末位数字正则) | 61.94 | 68.01 | 68.99 | **7.05** |
| `gsm8k` math-verify(单 benchmark) | 68.69 | 69.83 | 71.80 | **3.11** |
| 五 benchmark 等权(math-verify) | 34.99 | 34.89 | 35.77 | **0.88** |

**跨度单调缩小:7.05 → 3.11 → 0.88。**

三者并非同一量的三次测量(benchmark 数与判分器都不同),故不能当作严格的
"信噪比提升"序列。但方向高度一致,且与 R46/R53 的机制吻合:
**末位数字正则对不同 checkpoint 的低估程度不同**(判分效应跨 checkpoint 从 +0.38 到 +3.72),
这种**与被测对象相关的偏倚**会被误读成算子差异。

**对论文的直接含义**:若只用旧协议单 benchmark,bf16 档会得出
"修正带来 +7 分"的结论;换用 math-verify 五 benchmark 后,**同一批 checkpoint 上差异只剩 0.88 分**。
**"算子有没有差别"这个问题的答案,在低噪声档几乎完全由量具决定。**

#### 四、E3 曲线现状(9 格填满 8 格)

| 档位 | `nocorr` | `fullis` | `ours` | `ours−nocorr` |
|---|---|---|---|---|
| bf16 | 34.99 | 34.89 | 35.77 | +0.78(不可分) |
| fp8_e4m3 | 34.14 | 32.56 | 33.95 | −0.19(不可分) |
| **fp4_e2m1** | **12.73** | 32.32 | **待测(约 12:58)** | 待测 |

**形状已经清楚**:bf16 与 fp8_e4m3 两档六个数全部落在 **[32.56, 35.77]**(跨度 3.21),
彼此难分;**唯一的阶跃发生在「不修正 × fp4」这一格**(12.73,比未训练的 base 低 16.20),
而同档的 `fullis` 仍有 32.32,与低噪声档持平。

**九格里有八格互相接近,一格塌陷。** 这就是 E3 的剂量-反应:
不是一条随噪声递降的斜坡,而是**一个只在"高噪声 × 无修正"处触发的开关**。

最后一格 `fp4 / ours` 将决定:这个开关是被**任何**修正关掉的,
还是 CIS 有其特殊之处 —— 目前它已越过 nocorr 的转折点且奖励为三臂最高,但**以 held-out 为准**。

### R64. CIS 全部结果汇总(截至 2026-09-14,九格填满八格)

应 kzhao2 与用户要求整理。**所有数字均从 `results/eval_suite.tsv` 重算**,不引用转述;
统计判据统一为五 benchmark 等权、math-verify 判分。

#### 一、主表三个 cell

| cell | CIS 值 | 名次 | 全块跨度 | 0.8 power 门槛 | 档内是否可分 |
|---|---|---|---|---|---|
| **A** = Qwen1.5-MoE(3 seed) | **34.47 ± 0.48** | **1 / 11** | 17.62 | — | **部分可分** |
| **D** = dsv2(单 seed) | 35.73 | 8 / 11 | **0.99** | **3.14** | **整块不可分** |
| **E** = q30b(单 seed) | 68.95 | 4 / 11 | **0.91** | **3.37** | **整块不可分** |

**cell A 是唯一有统计内容的 cell**,但"第一"要拆开看:

| 对比 | 差 | z | 判定 |
|---|---|---|---|
| CIS vs `nocorr` | **+3.48** | **11.46** | **真分离** |
| CIS vs `seqtis` | +2.19 | 3.84 | 真分离 |
| CIS vs `fullis` | +2.61 | 1.99 | 真分离(勉强) |
| CIS vs `kpopfix` | +1.48 | 1.47 | 不可分 |
| **CIS vs `IcePop`** | **+0.29** | **0.70** | **并列** |

→ **CIS 相对"不修正"的优势是真的;相对最强对手 IcePop 则是并列,"第一"是名次不是分离。**

**cell D / E 的名次没有统计内容**:两块跨度(0.99 / 0.91)都**远低于**各自门槛(3.14 / 3.37),
十一个方法互相都分不开。CIS 在 dsv2 的"第八"与第一名差 z=−0.84、与 `nocorr` 差 z=−0.54,
**都不可分**。根因是天花板:dsv2 最好方法仅高于 base **+1.46**,q30b 仅 **+1.32**。

#### 二、E3 剂量-反应(本管线自跑,既有 2026-08 行已按 R59 作废)

| 档位 | `nocorr` | `fullis` | `ours`(CIS) | CIS − nocorr |
|---|---|---|---|---|
| bf16 | 34.99 | 34.89 | 35.77 | **+0.78**(SE 1.09,不可分) |
| fp8_e4m3 | 34.14 | 32.56 | 33.95 | **−0.19**(不可分) |
| fp4_e2m1 | **12.73** | 32.32 | **运行中** | 待测 |

#### 三、经得起检验的 / 经不起检验的

**站得住**:
1. **cell A 上 CIS 显著优于不修正**(+3.48,z=11.46),且 kzhao2 已确认 cell A **未被
   `weight_update_*` 评测链 bug 污染**(xccl 权重同步不产生该目录,11 个 trial 全为 epoch 目录)。
2. **高噪声下"修正与否"决定成败**(R57):fp4 档 `nocorr` 12.73(比未训练 base 28.93 低 16.20),
   `fullis` 32.32,相差 **19.58 分**。但这一条**由 `fullis` 证明,不是 CIS 专属**。

**站不住 / 未被复现**:
3. **CIS 优于其他修正方法** —— cell A 上与 IcePop 并列(z=0.70),dsv2 / q30b 整块不可分。
4. **cell A 的 +3.48 在本管线未被复现**(R61):同模型、同算子配置
   (R30 已逐字核对 `dose.sbatch` 与 `cells.sbatch` 的 `ours` 分支一致,`KT_SIGMA_B=2.3`),
   本管线 bf16 只有 **+0.78**,与历史 +3.48 相比 **z = 2.39,显著更低**。
5. **低噪声档的"算子差异"高度依赖量具**(R63):同一批 bf16 checkpoint,
   档内跨度随量具改善从 **7.05 → 3.11 → 0.88** 单调缩小。

**综合**:CIS 目前唯一确凿的结论是 **cell A 上优于不修正**。
"优于其他修正方法"缺乏证据;"可跨模型规模推广"被 dsv2 / q30b 否证(虽因天花板而无判别力);
"在本管线可复现"被 R61 否证。**三次独立的复现尝试(dsv2、q30b、本管线 bf16)都没有再现那个 +3.48。**

#### 四、唯一未决的一格,以及它能回答什么

`13676652`(fp4 / `ours`)运行中。它**不能**挽回第 3–5 条,但能回答一个明确的问题:
**在已把 `nocorr` 打到比不训练还差 16 分的噪声下,CIS 是否至少与 `fullis` 一样守得住?**
- 若 CIS ≈ 32(与 `fullis` 相当)→ "修正有用"再添一例,但仍非 CIS 专属;
- 若 CIS 明显高于 32 → **这将是 CIS 相对其他修正方法的第一个真实分离**(该档效应量足够大到能被分辨);
- 若 CIS 崩溃 → 对 CIS 的负面证据。

#### 五、要让 CIS 的主张变强,需要什么

**补种子是唯一出路,且必须补在有判别力的地方**:
- cell A 的 CIS / IcePop 各补到 5–6 seed,才可能把 z=0.70 推到可分;
- dsv2 / q30b **不值得补**——天花板只有 1.3–1.5 分,补到 10 seed 也跨不过门槛;
- **fp4 档才是有判别力的地方**(效应量 19.58 分,是种子噪声的 8.8 倍),
  该档三臂各补 2–3 seed 的性价比远高于低噪声档。
