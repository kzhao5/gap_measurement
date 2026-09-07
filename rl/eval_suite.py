#!/usr/bin/env python
"""5-benchmark eval suite (greedy, vLLM TP) for RL checkpoints.
Benches: gsm8k(1319), math500(500), svamp(300), minerva(272), olympiad(674).
Unified protocol: chat template + 'put your final answer within \\boxed{}', greedy,
max 1536 new tokens; judged by math-verify; gsm8k also last-number EM (legacy judge).
Usage: eval_suite.py <ckpt_path> <tag> [tp] [--dry]
"""
import os, re, sys
from datasets import load_dataset

path, tag = sys.argv[1], sys.argv[2]
tp = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 4
DRY = "--dry" in sys.argv
INSTR = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

def bench_gsm8k():
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [q + INSTR for q in ds["question"]], [a.split("####")[-1].strip().replace(",", "") for a in ds["answer"]]

def bench_math500():
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [q + INSTR for q in ds["problem"]], list(ds["answer"])

def bench_svamp():
    ds = load_dataset("ChilleD/SVAMP", split="test")
    golds = [str(int(a)) if float(a) == int(a) else str(a) for a in ds["Answer"]]
    return [f"{b.strip()} {q.strip()}" + INSTR for b, q in zip(ds["Body"], ds["Question"])], golds

def bench_minerva():
    ds = load_dataset("math-ai/minervamath", split="test")
    return [q + INSTR for q in ds["question"]], list(ds["answer"])

def bench_olympiad():
    ds = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
    prompts, golds = [], []
    for r in ds:
        ctx = (r.get("context") or "").strip()
        q = (ctx + "\n" + r["question"]) if ctx else r["question"]
        prompts.append(q + INSTR)
        golds.append(r["final_answer"][0] if r["final_answer"] else "")
    return prompts, golds

BENCHES = [("gsm8k", bench_gsm8k), ("math500", bench_math500), ("svamp", bench_svamp),
           ("minerva", bench_minerva), ("olympiad", bench_olympiad)]

data = {}
for name, fn in BENCHES:
    p, g = fn()
    data[name] = (p, g)
    print(f"LOADED {name} n={len(p)} gold0={g[0]!r}", flush=True)
if DRY:
    sys.exit(0)

from math_verify import parse, verify
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def judge_mv(pred_text, gold):
    try:
        gp = parse(gold if gold.lstrip().startswith(("$", "\\")) else f"${gold}$")
        if not gp:
            gp = parse(gold)
        pp = parse(pred_text)
        return bool(gp) and bool(pp) and verify(gp, pp)
    except Exception:
        return False

def judge_em(pred_text, gold):
    nums = re.findall(r"-?\d+\.?\d*", pred_text.replace(",", ""))
    return bool(nums) and bool(gold) and nums[-1].rstrip(".") == gold

def main():
    tok = AutoTokenizer.from_pretrained(path)
    llm = LLM(model=path, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=4096, enforce_eager=True, tensor_parallel_size=tp)
    sp = SamplingParams(temperature=0.0, max_tokens=1536)
    lines = []
    for name, _ in BENCHES:
        prompts, golds = data[name]
        chat = [tok.apply_chat_template([{"role": "user", "content": p}],
                                        tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in prompts]
        outs = llm.generate(chat, sp)
        texts = [o.outputs[0].text for o in outs]
        print(f"SAMPLE {name} len0={len(texts[0])} text0={texts[0][:160]!r}", flush=True)
        acc = sum(judge_mv(t, g) for t, g in zip(texts, golds)) / len(golds)
        lines.append(f"RESULT {tag} {name} acc={acc:.4f} n={len(golds)}")
        if name == "gsm8k":
            em = sum(judge_em(t, g) for t, g in zip(texts, golds)) / len(golds)
            lines.append(f"RESULT {tag} gsm8k_em acc={em:.4f} n={len(golds)}")
        print(lines[-1], flush=True)
    print("\n".join(lines), flush=True)
    with open("/home/kzhao2/gap_measurement/results/eval_suite.tsv", "a") as f:
        for ln in lines:
            _, t, b, a, n = ln.split()
            f.write(f"{t}\t{b}\t{a.split('=')[1]}\t{n.split('=')[1]}\n")
    sys.stdout.flush(); os._exit(0)  # vLLM TP workers hang on interpreter shutdown

if __name__ == "__main__":
    main()
