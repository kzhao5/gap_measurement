# MoE × 多 benchmark 扩展方案(v2,2026-08-19,用户定调:主打 MoE,≥4-5 个 benchmark)

## 总体结构:训练 cell 少而精,评测 benchmark 面广
数学 RL 论文标准做法(SimpleRL-Zoo 等):训练只在 1-2 个数据集,**每个 checkpoint
在全部 benchmark 上评测**。评测便宜,训练贵;benchmark 数量的扩张全部压在评测侧。

### 训练 cell(全是 MoE)
| Cell | 模型 | 训练集 | 臂 | 种子 |
|---|---|---|---|---|
| A(已完成) | Qwen1.5-MoE-A2.7B-Chat | GSM8K | 6 臂+消融+fp8 | 3 |
| B(新) | Qwen1.5-MoE-A2.7B-Chat | MATH 7.5k(gen 2048) | nocorr/TIS/IcePop/CIS | 3 |
| D(新,P2) | DeepSeek-V2-Lite-Chat(稳)或 Qwen3-30B-A3B(OOM 风险) | GSM8K | 4 臂 | 1-2 |
| (dense 对照降为可选,只在被审稿人要求时补) |

### 评测 benchmark(每个 final ckpt 全评;主表 5 个 + 附录 2 个)
| Benchmark | 题数 | 协议 | 单 ckpt SE | 备注 |
|---|---|---|---|---|
| GSM8K test | 1319 | greedy pass@1 | ~1.3 | 已有 |
| MATH500 | 500 | greedy pass@1 | ~2.2 | 已缓存 |
| SVAMP | 300 | greedy pass@1 | ~2.5 | 域外小学数学,弱模型有分辨力 |
| Minerva Math | 272 | greedy pass@1 | ~3.0 | math-ai/minervamath |
| OlympiadBench | 675(TO 文本子集) | greedy pass@1 | ~1.8 | hard |
| AMC23(附录) | 40 | avg@8, t=1 | 大 | math-ai/amc23 |
| AIME24(附录) | 30 | avg@8, t=1 | 大 | 已缓存;先探测是否全 0 |

判分统一 math-verify(GSM8K 保留 last-number EM 双报以衔接旧结果);
prompt 用 chat template + "put the final answer in \boxed{}";max 2048 tokens。

### 主表形态(TIGER 式)
大组 = 训练设置(GSM8K-trained / MATH-trained / 大 MoE),子组 = 5 benchmark,
行 = 方法(mean±std over 3 seeds),外加 Avg 宏平均列;CIS 行高亮。
诚实预期:方法间差距主要出现在训练分布近的集(GSM8K/MATH500/SVAMP);
hard 集(Minerva/Olympiad)大家都弱、SE 大,主要证明"不伤害 OOD"。

## 预算(8 卡 node-hours)
- A 重评:18 ckpt × 5 bench,TP4 批量(一个 job 评一个 ckpt 全部 bench ≈1.5h TP4)≈ 14
- B 训练:12 runs × 4-6h ≈ 60;B 评测 ≈ 9
- D 训练+评测 ≈ 15-20;冒烟+静态测量(c/ξ₊/ε̂→编译 λ₊)≈ 10(单卡为主)
- 总计 ≈ 110-120;race.sbatch 抢 dw/cs/cs2/m13h

## 需要准备
- [ ] 数据集离线缓存:SVAMP(ChilleD/SVAMP)、Minerva(math-ai/minervamath)、
      OlympiadBench(TO 文本子集)、AMC23、MATH train;已有:MATH500、AIME24、GSM8K
- [ ] math-verify 装进 eval venv
- [ ] eval_suite.py:MODEL/CKPT 参数,循环 5-7 个 bench,每个打印 RESULT 行,os._exit(0)
- [ ] eval_suite.sbatch(TP4,链在训练后)+ 老 ckpt 批量重评脚本
- [ ] seeds/race.sbatch 参数化 DATASET/GENLEN(MATH cell)
- [ ] D cell 权重下载 + FSDP 30B act-ckpt 配置(若选 Qwen3-30B-A3B)
- [ ] 预注册更新:评测集清单、协议、λ₊=编译值,先写后跑
- 风险:AMC/AIME 全 0(留附录);MATH 奖励格式 hacking(math-verify 严格模式);
  nobackup 90 天清除;fairshare

## 流水线(每个新训练 cell 不变)
离线缓存 → 8 卡冒烟(FA2+ktdump,median log k≈0)→ 单卡静态测量得 c/ξ₊/ε̂/κ
→ 解 equalizer 编译 λ₊ → 4 臂 RL → eval_suite 链评 → MANIFEST/SIGMA_TIS 记录。

## Baseline 缺口分析(v2 增补,2026-08-19;定调:第三模型 = Qwen3-30B-A3B)

按文献把修正方法分四层,我们的覆盖情况:

**A. token 级 ratio 算子(我们的算子类 M(k,p))——已齐**
nocorr / full-IS / TIS(cap 2, Yao&Yao 线) / IcePop(mask, Ant) / KPop(binary-KL mask, AReaL)
/ KPop†(数值修正) / CIS。这是主表,不缺。

**B. 序列级修正——缺,必补(P0)**
Seq-TIS(整条轨迹乘积 ratio 截断)与 Seq-MIS(乘积 ratio 出界整条 mask),出自
"When Speed Kills Stability"/Seq-MIS 线,是 2025H2 起最活跃的对手。对我们特别有利:
S0 闭包实验已证明序列级 best-response 收敛到常数 cap(λ*=0.07),理论预言它们
在长轨迹上退化 → 跑出来无论输赢都是内容。实现:loss 层按 cu_seqlens 聚合 log k,
AReaL 里 ~30 行。臂数:旗舰 3 种子,其他模型 1 种子。

**C. surrogate 级(改 PPO 目标而非修 k)——缺,必补(P0)**
GSPO(Qwen 团队,序列级几何均值 ratio 进 clip):官方叙事就是"MoE RL 不用 routing
replay 也稳",Qwen3-30B-A3B 上审稿人必问。AReaL 已内建
(_compute_sequence_level_ratio_and_advantages),纯配置臂。
CISPO(MiniMax M1)属同层但换掉了 PPO surrogate 本身,related work 引用即可,不跑。

**D. 系统级(从源头消灭失配)——一补一论**
- FP16 全链路(Defeating the Mismatch via FP16):便宜(SGLang dtype + FSDP 混精配置),
  必补一臂(旗舰 cell,1-3 种子)。附带红利:测 fp16 下的 c,理论预言 c 大幅缩小
  (尾数 10 位 vs bf16 7 位),是编译链 λ=cτ* 的又一个正向数据点;同时回应
  "为什么不直接换 fp16"——我们的答案:fp16 只压 scale 通道的 c,不解决版本陈旧的
  B_t,且大模型有量程风险。
- R3/PR2(routing replay,MoE 专用 SOTA):不属于算子类(改训练前向,把推理端路由
  回放进训练),AReaL/FSDP 实现工程量大(需 SGLang 导出逐 token 专家 + HF 前向 hook
  替换 top-k)。处理:related work + 讨论"正交可叠加"(R3 消掉 B_t 的路由来源,我们
  的算子处理剩余;我们已有 routing-flip enrichment 1.46× 的测量支撑这个话术)。
  Qwen3-30B-A3B cell 若时间富余再评估单跑一臂,优先级 P2,不进承诺范围。

**不补的及理由**:DAPO/Dr.GRPO(recipe 层,所有臂共享 recipe 已控制)、
VESPO/TOPR/Trust-the-Batch(preprint 小众,类内代表已有)、PR2(与 R3 同族取一)。

**新臂预算**(seq-tis / seq-mis / gspo / fp16 ×旗舰 3 种子 + 16B/30B 各 1 种子):
旗舰 12 runs×2h ≈ 24 node-h;16B/30B 每模型 4 runs ≈ 视步时另计。
预注册:Seq-TIS cap=2(对齐 TIS)、Seq-MIS 阈值按原文、GSPO clip 按 AReaL 默认、
FP16 loss-scale 默认;全部先写 contract 再跑。
