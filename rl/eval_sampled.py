"""GSM8K sampled-accuracy eval: temperature 1, n=8, mean accuracy (avg@8)."""
import os, sys
path, tag = sys.argv[1], sys.argv[2]
tp = int(sys.argv[3]) if len(sys.argv) > 3 else 4
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="test")
prompts = [q + "\nPlease reason step by step, and put your final answer within \\boxed{}." for q in ds["question"]]
golds = [a.split("####")[-1].strip() for a in ds["answer"]]
from math_verify import parse, verify
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def judge(pred, gold):
    try: return bool(verify(parse("$"+gold+"$"), parse(pred)))
    except Exception: return False

def main():
    tok = AutoTokenizer.from_pretrained(path)
    llm = LLM(model=path, dtype="bfloat16", gpu_memory_utilization=0.85,
              tensor_parallel_size=tp, max_model_len=4096, enforce_eager=True)
    sp = SamplingParams(temperature=1.0, top_p=1.0, n=8, max_tokens=int(os.environ.get("MAXTOK","1536")), seed=1234)
    chat = [tok.apply_chat_template([{"role":"user","content":p}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False) for p in prompts]
    outs = llm.generate(chat, sp)
    tot = sum(judge(o.text, g) for out, g in zip(outs, golds) for o in out.outputs)
    acc = tot / (len(golds)*8)
    mt=os.environ.get("MAXTOK","1536"); line = f"RESULT {tag} gsm8k_avg8_t1_mt{mt} acc={acc:.4f}"
    print(line, flush=True)
    with open(os.path.expanduser("~/gap_measurement/results/eval_suite.tsv"),"a") as f:
        f.write(f"{tag}\tgsm8k_avg8t1_mt{mt}\t{acc:.4f}\t{len(golds)*8}\n")
    os._exit(0)

if __name__ == "__main__":
    main()
