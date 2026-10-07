"""Convert a PEFT LoRA adapter into an MLX adapter directory.

PEFT:  <prefix>.layers.N.<module>.lora_A.weight  (r, in)    lora_B.weight (out, r)
MLX:   model.layers.N.<module>.lora_a            (in, r)    lora_b        (r, out)
PEFT computes scale * (x A^T) B^T, MLX computes scale * (x a) b, so a = A^T and b = B^T.

Usage: uv run python scripts/convert_adapter.py [--adapter machu8/serana-sft] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import mlx.core as mx

KEY_RE = re.compile(r"^(?:.*\.)?layers\.(\d+)\.((?:self_attn|mlp)\.\w+)\.lora_([AB])\.weight$")


def convert_tensors(peft: dict[str, mx.array]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    for key, w in peft.items():
        m = KEY_RE.match(key)
        if not m:
            raise ValueError(f"unexpected tensor name in PEFT adapter: {key}")
        layer, module, ab = m.groups()
        out[f"model.layers.{layer}.{module}.lora_{ab.lower()}"] = mx.contiguous(w.T)
    return out


def expected_dims(base_config: dict) -> dict[str, tuple[int, int]]:
    """(in, out) of each attention projection, from the base model's config.json."""
    hidden = base_config["hidden_size"]
    head_dim = base_config.get("head_dim", hidden // base_config["num_attention_heads"])
    q_out = base_config["num_attention_heads"] * head_dim
    kv_out = base_config["num_key_value_heads"] * head_dim
    return {
        "q_proj": (hidden, q_out),
        "k_proj": (hidden, kv_out),
        "v_proj": (hidden, kv_out),
        "o_proj": (q_out, hidden),
    }


def validate(
    mlx_tensors: dict[str, mx.array],
    rank: int,
    targets: list[str],
    num_layers: int,
    base_config: dict | None = None,
) -> None:
    expected_count = num_layers * len(targets) * 2
    if len(mlx_tensors) != expected_count:
        raise ValueError(f"expected {expected_count} tensors, got {len(mlx_tensors)}")
    dims = expected_dims(base_config) if base_config else {}
    for layer in range(num_layers):
        for target in targets:
            prefix = f"model.layers.{layer}.self_attn.{target}"
            a, b = mlx_tensors.get(f"{prefix}.lora_a"), mlx_tensors.get(f"{prefix}.lora_b")
            if a is None or b is None:
                raise ValueError(f"missing lora_a/lora_b for {prefix}")
            if a.shape[1] != rank or b.shape[0] != rank:
                raise ValueError(f"{prefix}: rank mismatch {a.shape} {b.shape}, rank {rank}")
            if target in dims and (a.shape[0], b.shape[1]) != dims[target]:
                raise ValueError(f"{prefix}: shape {a.shape}/{b.shape} != dims {dims[target]}")


def mlx_adapter_config(peft_config: dict, num_layers: int) -> dict:
    for flag in ("use_dora", "use_rslora"):
        if peft_config.get(flag):
            raise ValueError(f"{flag} adapters are not supported")
    if peft_config.get("layers_to_transform") or peft_config.get("layers_pattern"):
        # MLX applies LoRA to the last num_layers layers; arbitrary layer subsets don't map.
        raise ValueError("adapters with layers_to_transform/layers_pattern are not supported")
    if peft_config.get("rank_pattern") or peft_config.get("alpha_pattern"):
        raise ValueError("per-layer rank/alpha patterns are not supported")
    rank = peft_config["r"]
    return {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": {
            "rank": rank,
            "scale": peft_config["lora_alpha"] / rank,
            "dropout": 0.0,  # inference only
            "keys": [f"self_attn.{t}" for t in sorted(peft_config["target_modules"])],
        },
    }


def convert(adapter_dir: Path, out_dir: Path, base_config: dict | None = None) -> dict:
    peft_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    peft = mx.load(str(adapter_dir / "adapter_model.safetensors"))
    tensors = convert_tensors(peft)
    num_layers = 1 + max(int(KEY_RE.match(k).group(1)) for k in peft)
    config = mlx_adapter_config(peft_config, num_layers)
    validate(tensors, peft_config["r"], peft_config["target_modules"], num_layers, base_config)
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_dir / "adapters.safetensors"), tensors)
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2))
    template = adapter_dir / "chat_template.jinja"
    if template.exists():
        shutil.copy(template, out_dir / "chat_template.jinja")
    return config


def _snapshot(repo: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo, local_files_only=True, allow_patterns=["*.json", "*.safetensors", "*.jinja"]
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--adapter", default="machu8/serana-sft", help="HF repo (cached) or directory")
    p.add_argument("--base", default="Qwen/Qwen3-8B", help="used only to check shapes")
    p.add_argument("--out", default="models/serana-adapter-mlx")
    args = p.parse_args()

    adapter = Path(args.adapter) if Path(args.adapter).is_dir() else _snapshot(args.adapter)
    base = Path(args.base) if Path(args.base).is_dir() else _snapshot(args.base)
    base_config = json.loads((base / "config.json").read_text())
    config = convert(adapter, Path(args.out), base_config)
    print(f"wrote {args.out}: {json.dumps(config['lora_parameters'])}")


if __name__ == "__main__":
    main()
