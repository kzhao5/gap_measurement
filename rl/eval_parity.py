"""Engine parity: EXACT AReaL rollout protocol on vLLM (train split, AReaL prompt, avg@4, t=1, max 1024)."""
import os, sys
path, tag = sys.argv[1], sys.argv[2]
tp = int(sys.argv[3]) if len(sys.argv) > 3 else 4
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="train").select(range(2000))
prompts=[q+"\nPlease put your final answer within \\boxed{}." for q in ds["question"]]
golds=[a.split("####")[-1].strip() for a in ds["answer"]]
from math_verify import parse, verify
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def judge(pred,gold):
    try: return bool(verify(parse(str(gold)), parse(str(pred))))
    except Exception: return False

def main():
    tok=AutoTokenizer.from_pretrained(path)
    llm=LLM(model=path,dtype="bfloat16",gpu_memory_utilization=0.85,tensor_parallel_size=tp,max_model_len=4096,enforce_eager=True)
    sp=SamplingParams(temperature=1.0,top_p=1.0,n=4,max_tokens=1024,seed=7)
    chat=[tok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for p in prompts]
    outs=llm.generate(chat,sp)
    tot=sum(judge(o.text,g) for out,g in zip(outs,golds) for o in out.outputs)
    acc=tot/(len(golds)*4)
    print(f"RESULT {tag} parity acc={acc:.4f}",flush=True)
    with open(os.path.expanduser("~/gap_measurement/results/eval_suite.tsv"),"a") as f:
        f.write(f"{tag}\tvllm_parity_train_avg4\t{acc:.4f}\t{len(golds)*4}\n")
    os._exit(0)

if __name__=="__main__":
    main()
