"""Merge the PEFT adapter into the bf16 base (PEFT merge_and_unload) for experiment 2.

Needs the `convert` dependency group: uv run --group convert python scripts/merge_adapter.py
With --quantize it also writes a 4bit MLX copy of the merged model.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="Qwen/Qwen3-8B")
    p.add_argument("--adapter", default="machu8/serana-sft")
    p.add_argument("--out", default="models/qwen3-8b-serana-merged-bf16")
    p.add_argument("--quantize", default=None, help="also write a 4bit MLX model to this path")
    args = p.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    out = Path(args.out)
    merged.save_pretrained(out)
    # The adapter repo ships the tokenizer and chat template used in training.
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(out)
    print(f"wrote {out}")

    if args.quantize:
        from mlx_lm import convert

        convert(str(out), args.quantize, quantize=True, q_bits=4, q_group_size=64)
        print(f"wrote {args.quantize}")


if __name__ == "__main__":
    main()
