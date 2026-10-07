"""Experiment 2: persona replies under different quantization/merge conditions (design section 4).

Conditions generated here (all MLX, greedy):
  mlx4bit_adapter  4bit base + unmerged adapter (the serving setup)
  merged_bf16      bf16 merged model       (scripts/merge_adapter.py)
  merged_4bit      4bit merged model       (scripts/merge_adapter.py --quantize)
The NF4 + adapter condition needs a CUDA GPU; pass its replies with --external nf4_adapter=<jsonl>
(one {"id", "reply"} per line).

Scoring is optional (--score): it imports judge_pcs and paired_bootstrap_diff from the
serana-post-training repo and calls the OpenAI judge, so it needs OPENAI_API_KEY.
Pass criterion 1 (design section 4): the lower CI bound of (mlx4bit_adapter - merged_4bit)
must be above -tolerance, where the tolerance is fixed before the run.

Usage: uv run python experiments/exp2_persona_generate.py [--conditions ...] [--score]
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

POST_TRAINING = Path(
    os.environ.get("SERANA_POST_TRAINING_DIR", "/Users/jongmin/Project/serana-post-training")
)
MODELS_DIR = os.environ.get("SERANA_MODELS_DIR", "models")
DEFAULT_PROMPTS = [
    "data/eval/eval_set_v1/eval_prompts.jsonl",
    "data/eval/eval_set_v1/attack_probes.jsonl",
]
ADAPTER_PATH = f"{MODELS_DIR}/serana-adapter-mlx"
# (base path, adapter path, adapter on)
CONDITIONS = {
    "mlx4bit_adapter": (f"{MODELS_DIR}/qwen3-8b-4bit", ADAPTER_PATH, True),
    "merged_bf16": (f"{MODELS_DIR}/qwen3-8b-serana-merged-bf16", None, False),
    "merged_4bit": (f"{MODELS_DIR}/qwen3-8b-serana-merged-4bit", None, False),
}
FALLBACK_SYSTEM = (
    "You are Serana, a character from The Elder Scrolls V: Skyrim -- Dawnguard.\n"
    "Always reply in natural Korean, in Serana's voice.\n"
    "Do not mention being an AI, a model, or a language system."
)


def load_prompts(paths: list[Path]) -> list[dict[str, str]]:
    out = []
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out.append({"id": row["id"], "prompt": row.get("prompt_ko") or row["prompt"]})
    return out


def read_replies(path: Path) -> dict[str, str]:
    rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {r["id"]: r["reply"] for r in rows}


def system_prompt() -> str:
    """The persona system prompt from serana-post-training PROMPTS.md section 1, if available."""
    try:
        import yaml

        cfg = yaml.safe_load((POST_TRAINING / "config/persona.yaml").read_text())
    except (OSError, ImportError):
        return FALLBACK_SYSTEM
    return (
        f"You are {cfg['persona_name']}, a character from {cfg['source_title']}.\n"
        f"Always reply in natural Korean, in {cfg['persona_name']}'s voice.\n\n"
        f"Personality and voice:\n{cfg['persona_profile'].strip()}\n\n"
        "Rules:\n"
        f"- Stay fully in character. Speak in {cfg['persona_name']}'s voice, register, and "
        "worldview at all times.\n"
        f"- Only use knowledge {cfg['persona_name']} could plausibly have. If asked about "
        "something outside that world or era, react as the character would to an unknown "
        "topic; do not break character to answer.\n"
        "- Do not mention being an AI, a model, or a language system."
    )


def generate(condition: str, prompts: list[dict[str, str]], system: str, max_tokens: int):
    import mlx.core as mx

    from serana_agent.llm.mlx_local import LocalModel

    base, adapter, adapter_on = CONDITIONS[condition]
    lm = LocalModel(base, adapter)
    role = lm.persona(adapter=adapter_on)
    rows = []
    for p in prompts:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": p["prompt"]}]
        reply = role.chat(messages, max_tokens=max_tokens).content
        rows.append({"condition": condition, "id": p["id"], "prompt": p["prompt"], "reply": reply})
    del lm, role
    gc.collect()
    mx.clear_cache()
    return rows


@contextlib.contextmanager
def _in_dir(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def score_rows(rows: list[dict[str, Any]], judge_model: str | None) -> list[dict[str, Any]]:
    # judge_pcs reads config/*.yaml relative to the cwd at import time.
    sys.path.insert(0, str(POST_TRAINING / "src"))
    with _in_dir(POST_TRAINING):
        from eval.judge_pcs import judge_pcs
    scored = []
    for r in rows:
        j = judge_pcs(r["prompt"], r["reply"], judge_model)
        scored.append({**r, "score": j["score"], "violation": j["violation"]})
    return scored


def compare(scored: list[dict[str, Any]], a: str, b: str, tolerance: float) -> dict[str, Any]:
    sys.path.insert(0, str(POST_TRAINING / "src"))
    from eval.metrics import paired_bootstrap_diff

    by = {c: {r["id"]: r["score"] for r in scored if r["condition"] == c} for c in (a, b)}
    ids = sorted(set(by[a]) & set(by[b]))
    diff = paired_bootstrap_diff([by[a][i] for i in ids], [by[b][i] for i in ids])
    return {**diff, "a": a, "b": b, "tolerance": tolerance, "pass": diff["ci_low"] > -tolerance}


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items))


def main() -> None:
    global POST_TRAINING  # the helpers above read it
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--prompts", nargs="*", default=None, help="jsonl files (prompt_ko or prompt)")
    p.add_argument("--external", nargs="*", default=[], metavar="NAME=JSONL")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--score", action="store_true")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--tolerance", type=float, default=0.25, help="score points; fix before running")
    p.add_argument("--out", default="runs/exp2")
    p.add_argument(
        "--post-training-dir", default=str(POST_TRAINING), help="serana-post-training checkout"
    )
    args = p.parse_args()
    POST_TRAINING = Path(args.post_training_dir)

    paths = (
        [Path(x) for x in args.prompts]
        if args.prompts
        else [POST_TRAINING / x for x in DEFAULT_PROMPTS]
    )
    prompts = load_prompts(paths)[: args.limit]
    system = system_prompt()
    out_dir = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    for cond in filter(None, args.conditions.split(",")):
        print(f"generating {cond}")
        rows += generate(cond, prompts, system, args.max_tokens)
    for spec in args.external:
        name, _, path = spec.partition("=")
        replies = read_replies(Path(path))
        rows += [
            {"condition": name, "id": q["id"], "prompt": q["prompt"], "reply": replies[q["id"]]}
            for q in prompts
            if q["id"] in replies
        ]
    write_jsonl(out_dir / "replies.jsonl", rows)

    if args.score:
        scored = score_rows(rows, args.judge_model)
        write_jsonl(out_dir / "scores.jsonl", scored)
        names = {r["condition"] for r in scored}
        if {"mlx4bit_adapter", "merged_4bit"} <= names:
            result = compare(scored, "mlx4bit_adapter", "merged_4bit", args.tolerance)
            (out_dir / "criterion1.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
