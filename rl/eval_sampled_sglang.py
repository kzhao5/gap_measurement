"""GSM8K avg@8 t=1 via SGLang offline engine (training-identical serving stack)."""
import os, sys
path, tag = sys.argv[1], sys.argv[2]
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="test")
prompts=[q+"\nPlease put your final answer within \\boxed{}." for q in ds["question"]]
golds=[a.split("####")[-1].strip() for a in ds["answer"]]
from math_verify import parse, verify
from transformers import AutoTokenizer
import sglang as sgl

def judge(pred,gold):
    try: return bool(verify(parse("$"+gold+"$"), parse(pred)))
    except Exception: return False

def main():
    tok=AutoTokenizer.from_pretrained(path)
    chat=[tok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for p in prompts]
    eng=sgl.Engine(model_path=path, attention_backend=os.environ.get("ATTN","triton"), mem_fraction_static=0.85, disable_radix_cache=True)
    sp={"temperature":1.0,"top_p":1.0,"max_new_tokens":1024,"n":8}
    outs=eng.generate(chat, sp)
    tot=0; cnt=0
    for o,g in zip(outs,golds):
        gen=o if isinstance(o,list) else [o]
        for oo in gen:
            tot+=judge(oo["text"],g); cnt+=1
    acc=tot/max(cnt,1)
    print(f"RESULT {tag} sglang_avg8_t1 acc={acc:.4f} n={cnt}",flush=True)
    with open(os.path.expanduser("~/gap_measurement/results/eval_suite.tsv"),"a") as f:
        f.write(f"{tag}\tsglang_avg8t1\t{acc:.4f}\t{cnt}\n")
    os._exit(0)

if __name__=="__main__":
    main()
