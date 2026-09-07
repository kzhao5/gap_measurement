#!/usr/bin/env python
"""GSM8K test-set accuracy eval (greedy, vLLM) for RL checkpoints."""
import re
import sys

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

path = sys.argv[1]
tag = sys.argv[2]
tp = int(sys.argv[3]) if len(sys.argv) > 3 else 1
ds = load_dataset("openai/gsm8k", "main", split="test")
tok = AutoTokenizer.from_pretrained(path)
prompts = [tok.apply_chat_template([{"role": "user", "content": q}],
                                   tokenize=False, add_generation_prompt=True)
           for q in ds["question"]]
llm = LLM(model=path, dtype="bfloat16", gpu_memory_utilization=0.85,
          max_model_len=2048, enforce_eager=True, tensor_parallel_size=tp)
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1024))
correct = 0
for o, ans in zip(outs, ds["answer"]):
    gold = ans.split("####")[-1].strip().replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", o.outputs[0].text.replace(",", ""))
    if nums and gold and nums[-1].rstrip(".") == gold:
        correct += 1
print(f"RESULT {tag} acc={correct/len(ds):.4f} ({correct}/{len(ds)})")
import os, sys; sys.stdout.flush(); os._exit(0)  # vLLM TP workers hang on interpreter shutdown
