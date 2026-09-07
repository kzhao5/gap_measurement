"""Replicate AReaL's rollout+reward path offline with vLLM and isolate the gap."""
import os, sys, re
model, tag = sys.argv[1], sys.argv[2]
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from areal.utils.hf_utils import apply_chat_template
from areal.api.cli_args import GenerationHyperparameters
from areal.reward.gsm8k import gsm8k_reward_fn
from math_verify import parse, verify
g = GenerationHyperparameters()
ds = load_dataset("openai/gsm8k", "main", split="train").select(range(1000))
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
def main():
    prompts_ids=[]; raws=[]; golds=[]
    for ex in ds:
        msgs=[{"role":"user","content":ex["question"]+"\nPlease put your final answer within \\boxed{}."}]
        prompts_ids.append(apply_chat_template(tok, msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False))
        raws.append(ex["answer"]); golds.append(ex["answer"].split("####")[-1].strip())
    print("PROMPT_TAIL:", repr(tok.decode(prompts_ids[0])[-120:]), flush=True)
    llm=LLM(model=model, dtype="bfloat16", tensor_parallel_size=4, gpu_memory_utilization=0.85, max_model_len=4096, enforce_eager=True)
    sp=SamplingParams(temperature=1.0, top_p=getattr(g,"top_p",1.0), top_k=getattr(g,"top_k",-1) or -1, n=4, max_tokens=1024, seed=3)
    print("SAMPLING:", sp, flush=True)
    outs=llm.generate([{"prompt_token_ids":p} for p in prompts_ids], sp)
    n=0; r_areal=0; r_gold=0; trunc=0; think=0
    for out,raw,gold in zip(outs,raws,golds):
        for o in out.outputs:
            n+=1; t=o.text
            r_areal+=float(gsm8k_reward_fn(None,t,None,None,raw))
            try: r_gold+=bool(verify(parse("$"+gold+"$"),parse(t)))
            except Exception: pass
            trunc+= (o.finish_reason=="length"); think+= ("<think>" in t)
    print(f"RESULT {tag} n={n} areal_reward={r_areal/n:.4f} gold_reward={r_gold/n:.4f} trunc_rate={trunc/n:.3f} think_rate={think/n:.3f}", flush=True)
    print("SAMPLE0:", repr(outs[0].outputs[0].text[:300]), flush=True)
    os._exit(0)
if __name__=="__main__": main()
