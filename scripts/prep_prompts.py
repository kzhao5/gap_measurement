"""Dump math RL prompts (DAPO-Math-17k) to a jsonl for the generation stage.

Run on the login node with the AReaL venv (has datasets + internet/cache):
  /home/kzhao2/AReaL/.venv/bin/python scripts/prep_prompts.py
"""

import json
import os
import random

from datasets import load_dataset

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "prompts_math.jsonl")
N_PROMPTS = 2500  # x G=8 x ~1k tok -> comfortably above the 1e7 token target
SEED = 20260809


def main():
    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
    print(f"dataset columns: {ds.column_names}, n={len(ds)}")

    rng = random.Random(SEED)
    idxs = rng.sample(range(len(ds)), min(N_PROMPTS, len(ds)))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for pid, i in enumerate(idxs):
            row = ds[i]
            # DAPO-Math-17k-Processed: 'prompt' is the question text
            text = row["prompt"] if isinstance(row["prompt"], str) else str(row["prompt"])
            f.write(
                json.dumps(
                    {"prompt_id": pid, "src_index": i, "text": text},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {len(idxs)} prompts to {OUT}")


if __name__ == "__main__":
    main()
