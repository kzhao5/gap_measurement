# 迁移到其他服务器:代码组成与复现步骤

## 1. 代码组成(三部分)
| 部分 | 位置 | 作用 | 是否必需 |
|---|---|---|---|
| **AReaL(上游 RL 框架)+ 我们的补丁** | github.com/areal-project/AReaL @ `areal_patches/UPSTREAM_COMMIT`(v2.0.0-32-ge786869)+ `areal_patches/areal_local.patch`(15 个文件) | 全部 RL 训练:FSDP 训练、SGLang/vLLM rollout、算子实现(σ-TIS/CIS、KPop 修正、Seq-TIS/MIS、fp16 loss scaling) | **必需**,训练无法脱离它 |
| **gap_measurement(本仓库)** | `rl/`(sbatch、评测、诊断)、`src/`(k_t 静态测量:vLLM 采样 + HF 重打分)、`analysis/`(两通道拟合、编译链 λ₊=cτ*、论文图)、`results/paper`+csv/json(小结果)、`SIGMA_TIS.md`(全部实验记录) | 实验编排、评测、分析 | 必需 |
| 论文 | Overleaf/GitHub `kzhao5/ICLR_icepop` | tex | 独立 |

不上传:`.venv`、`core.*`(崩溃转储)、`results/tokens_*.parquet`(每个 430MB 的原始测量,另行传输或重新测)、模型/数据缓存。

## 2. 环境(两个 venv,版本已锁在 env/)
- **AReaL venv**(Python 3.11,`env/requirements-areal.txt`):torch 2.9.1+cu129、sglang 0.5.10.post1、vllm 0.16.0、flashinfer(-python/-cubin)0.6.3、megatron-core 0.16.1、transformers 5.3.0、math-verify 0.8.0。
  安装后**必须**:`uv pip uninstall opencv-python-headless`(其自带 libcrypto 在 FIPS 内核上崩;非 FIPS 机器可不管)。
- **gap venv**(Python 3.12,`env/requirements-gap.txt`):vllm 0.26、transformers 5.14.1、math-verify 0.9(评测/测量用)。
- 新机器上用 `uv venv --python 3.11 && uv pip install -r env/requirements-areal.txt` 复现;若 CUDA 版本不同需改 torch 的 cu 标签。

## 3. 部署 AReaL
最省事:直接 clone 已打好补丁的分支(上游 v2.0.0-32 + 我们的 15 文件改动):
```
git clone -b kt-patches https://github.com/kzhao5/AReaL-kt.git ~/AReaL && cd ~/AReaL
```
(等价的手动方式:clone 上游 https://github.com/areal-project/AReaL,checkout `areal_patches/UPSTREAM_COMMIT`,`git apply areal_patches/areal_local.patch`。)
```
uv venv --python 3.11 .venv && uv pip install -e . -r ../gap_measurement/env/requirements-areal.txt
uv pip uninstall opencv-python-headless
```
补丁内容概要:算子(`areal/utils/functional/functional.py`:KT_SIGMA_*、KT_KPOP_FIX、KT_SEQ_PROD)、fp16 静态 loss scaling 与 deepseek_v2 mask(`fsdp_engine.py`)、ktdump 钩子、vLLM 后端修复(FIPS 环境注入、build_app 签名、custom all-reduce 字段、权重更新超时)、DeepSeek tokenizer fast-backend、Qwen3 thinking 默认关闭(`client.py`/`math_agent.py`)、`examples/math/gsm8k_icepop.yaml`(disk 权重更新、超时、vllm 段)。

## 4. 与集群绑定、迁移时必须改的地方
- `rl/env.sh`:`AREAL`/`RLROOT` 路径、`HF_HOME/HF_HUB_CACHE`(离线缓存目录)、`OPENSSL_CONF=/dev/null`(仅 FIPS 集群需要)。
- 所有 `rl/*.sbatch`:`--partition/--qos/--gres/--exclude`、日志路径、`module load`。
- `src/common.py`:`DATA_ROOT`、`CODE_ROOT`、`MODELS` 注册表。
- `rl/eval_suite.py`、`rl/cells.sbatch` 里写死的 `/home/kzhao2/...` 路径。
- 计算节点若无外网:先在登录节点缓存模型(Qwen/Qwen1.5-MoE-A2.7B-Chat、deepseek-ai/DeepSeek-V2-Lite-Chat、Qwen/Qwen3-30B-A3B)与数据集(openai/gsm8k、HuggingFaceH4/MATH-500、ChilleD/SVAMP、math-ai/minervamath、OlympiadBench 文本子集)。

## 5. 怎么跑(与本集群相同)
- 训练一臂:`sbatch --export=ALL,CELL=dsv2,METHOD=ours,SEED=1,TAG=vllm,EXTRA="rollout.backend=vllm:d4p1t1",LAMP=15.0 rl/cells.sbatch`(cell A 用 `rl/seeds.sbatch`,METHOD ∈ nocorr/fullis/tis/icepop/kpop/kpopfix/seqtis/seqmis/gspo/fp16/ours)。
- 训完自动链 `rl/eval_suite.sbatch`(五 benchmark,math-verify),结果追加到 `results/eval_suite.tsv`。
- 静态测量与 λ₊ 编译:`rl/measure.sbatch`(ARCH=…)→ `analysis/build_dataset.py <arch>` → `analysis/paper_zb.py <arch>` → `results/paper/params_<arch>.json` 的 `lambda_from_tau_star`。
- 引擎/协议一致性检查:`rl/eval_parity.py`、`rl/diag_areal_path.py`。

## 6. 已知坑(见 SIGMA_TIS.md 末尾 12 层修复记录)
FIPS OpenSSL、opencv libcrypto、flashinfer 双包版本、vLLM custom all-reduce、awex 依赖 megatron、xccl 权重同步挂死(用 disk)、DeepSeek tokenizer(transformers 5.3)、Qwen3 thinking 默认开启(4 处调用点)。

## 7. 同集群换账号(BYU 另一账号)——最短路径
同一集群意味着 CUDA/驱动/FIPS/分区完全相同,venv 用 `env/` 的锁定版本重建即可精确复现;只有**路径与账号权限**两类差异。
1. `git clone https://github.com/kzhao5/gap_measurement.git ~/gap_measurement`,AReaL 用 `git clone -b kt-patches https://github.com/kzhao5/AReaL-kt.git ~/AReaL`;`rl/*.sbatch`、`rl/env.sh`、`rl/*.py`、`src/common.py` 已改为 `$HOME`/`%u`/`expanduser`,无需改;
   `analysis/*.py`(离线分析脚本)仍含绝对路径,一条命令处理:`grep -rl /home/kzhao2 analysis scripts | xargs sed -i "s|/home/kzhao2|$HOME|g"`。
2. 建目录:`mkdir -p ~/nobackup/autodelete/{areal_rl/{logs,experiments,name_resolve,ktdump},hf,uv_cache,gap_measurement}`(BYU 的 `~/nobackup` 是到 `/nobackup/autodelete/usr/<user>` 的标准链接)。
3. AReaL:按 §3 clone + checkout + `git apply areal_patches/areal_local.patch`;`uv venv --python 3.11 .venv && uv pip install -e . -r ~/gap_measurement/env/requirements-areal.txt && uv pip uninstall opencv-python-headless`。
   gap venv:`cd ~/gap_measurement && uv venv --python 3.12 .venv && uv pip install -r env/requirements-gap.txt`。
4. 缓存(登录节点有外网,计算节点没有;缓存根目录 = env.sh 的 `HF_HUB_CACHE=~/nobackup/autodelete/hf`):
   `HF_HUB_CACHE=~/nobackup/autodelete/hf hf download Qwen/Qwen1.5-MoE-A2.7B-Chat`(同样:deepseek-ai/DeepSeek-V2-Lite-Chat、Qwen/Qwen3-30B-A3B);
   数据集:openai/gsm8k、HuggingFaceH4/MATH-500、ChilleD/SVAMP、math-ai/minervamath、OlympiadBench 文本子集(用 gap venv 的 `datasets.load_dataset` 触发缓存)。
   **DeepSeek 必做**:删掉缓存里 `config.json` 的 `auto_map` 键(其 remote code 与 transformers 5.3 不兼容;删后走原生 deepseek_v2 实现):
   `python -c "import json,glob;p=glob.glob('$HOME/nobackup/autodelete/hf/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/*/config.json')[0];c=json.load(open(p));c.pop('auto_map',None);json.dump(c,open(p,'w'),indent=2)"`。
5. 权限检查:`sacctmgr show assoc user=<新账号> format=Account,QOS%40`——需要 `cs`(cs/cs2 分区)、`dw87`(dw)、`gpu`(m13h);缺哪个就把对应 sbatch 的 `--partition/--qos` 换成可用的。
6. 我们账号下的 checkpoint/结果不可读(home 700);若需要历史 checkpoint 或 430MB 的原始测量 parquet,走 login 节点 /tmp 中转(见 SIGMA_TIS.md)或让 ORC 建 file-sharing group。
