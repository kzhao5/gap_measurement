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
