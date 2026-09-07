# 数据清单(Z+B 建模 §1 强制项)

## 静态测量(campaign,base 模型,单版本)
- 表:`results/tokens_moe.parquet`(12,293,115 行)、`results/tokens_dense.parquet`(12,389,115 行);float32 存储
- 字段:traj_id, prompt_id, pos, token_id, **logp_infer**(vLLM 0.26.0 采样步原始 logprob,temp=1/top-p=1,无重缩放/重归一化)、
  **logp_train**(HF bf16 teacher-forcing,同 checkpoint)、D=logp_train−logp_infer、
  MoE 另有 hard_flip / n_hard_flip_layers / flip_bits / margin_*(逐层专家翻转 + 门控 margin)、
  对照列 logp_train_fp32 / logp_c1(prefill 重打分)/ logp_c2a,c2b(两遍重算)
- 模型:Qwen/Qwen1.5-MoE-A2.7B-Chat 快照 `/home/kzhao2/nobackup/autodelete/hf/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be`(index md5 前缀 3a14da71222f);dense: Qwen1.5-14B-Chat
- 引擎:生成 vLLM 0.26.0 bf16(enforce_eager,无前缀缓存);重算 HF transformers,bf16,sdpa
- 原始 shard:`~/nobackup/autodelete/gap_measurement/{gen,recompute}/{moe,dense}/*.parquet`(会被清理;合并表已在 repo)
- log_m 从不写回:算子作用值仅在分析脚本内计算

## RL 训练逐 token dump(AReaL,带权重版本)
- champion (∞,2.3,ε₀=5e-3):`~/nobackup/autodelete/areal_rl/ktdump/lgrid_0/`(600 文件,KT_DUMP_EVERY=10;字段 old_logp(sglang rollout)/prox_logp(FSDP 训练器,更新前)/cur_logp/version;float32)
- 其余臂:recipe_{fullis,tis,icepop,kpop}(主表 baseline)、seeds_*_s{2,3}、fp8_*、v3_*、eps_0.1_s*;**recipe_sigmatis = v1 中心臂(崩溃),非 champion**
- 引擎:sglang(rollout,bf16;fp8 臂 kv_cache_dtype=fp8_e5m2)vs FSDP bf16 + FA2(训练器)
- 版本标签:version = 生成该 token 的策略版本;lag = 当步最大版本 − version

## 配对陈旧测量(同 token 多引擎版本)
- `~/nobackup/autodelete/gap_measurement/paired/`:trajs_s{0-3}(v57 生成 + 解码 logp)、pre_{v57,v86,v28}_s*(vLLM prefill 重打分)、hf_v86_s*(HF teacher-forcing);508,915 token
- checkpoint:kt-lgrid/lg0-e3/default/epoch{0,1,2}...globalstep{28,57,86}
