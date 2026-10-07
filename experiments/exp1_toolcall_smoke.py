"""Experiment 1: tool-call smoke test, adapter off/on x bf16/4bit (design section 9).

Heavy: loads Qwen3-8B (bf16 ~17GB, 4bit ~6GB) one precision at a time.
Usage: uv run python experiments/exp1_toolcall_smoke.py [--precisions 4bit,bf16] [--limit N]
Writes runs/exp1/<timestamp>/{results.jsonl,summary.json}.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import jsonschema

from serana_agent.llm.base import ChatModel, ToolSpec

DATA = Path(__file__).parent / "data"
SYSTEM = "You are a file assistant. Use the provided tools to do what the user asks."
MODELS_DIR = os.environ.get("SERANA_MODELS_DIR", "models")
BASE_PATHS = {"bf16": "Qwen/Qwen3-8B", "4bit": f"{MODELS_DIR}/qwen3-8b-4bit"}
ADAPTER_PATH = f"{MODELS_DIR}/serana-adapter-mlx"
# Arguments compared exactly (after path normalization); the rest just need to contain the text.
EXACT_KEYS = {"path", "src", "dst", "query", "old", "new", "glob"}


def load_prompts(path: Path = DATA / "toolcall_prompts.jsonl") -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_tools(path: Path = DATA / "tools.json") -> list[ToolSpec]:
    return [ToolSpec(**t) for t in json.loads(path.read_text())]


def _norm(key: str, value: Any) -> str:
    text = str(value).strip().casefold()
    if key in {"path", "src", "dst"}:
        text = text.removeprefix("./").rstrip("/") or "."
    return text


def args_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    for key, want in expected.items():
        if key not in actual:
            return False
        got, want_n = _norm(key, actual[key]), _norm(key, want)
        if (got != want_n) if key in EXACT_KEYS else (want_n not in got):
            return False
    return True


def score(item: dict[str, Any], result: Any, tools: list[ToolSpec]) -> dict[str, Any]:
    """Per-prompt flags. Later flags are False when an earlier step failed."""
    schemas = {t.name: t.input_schema for t in tools}
    call = result.tool_calls[0] if result.tool_calls else None
    format_ok = call is not None and result.parse_error is None
    tool_ok = bool(call) and call.name == item["expected_tool"]
    args_valid = False
    if call and call.name in schemas:
        try:
            jsonschema.validate(call.arguments, schemas[call.name])
            args_valid = True
        except jsonschema.ValidationError:
            pass
    return {
        "format_ok": format_ok,
        "tool_ok": tool_ok,
        "args_valid": args_valid,
        "args_match": tool_ok and args_match(item["expected_args"], call.arguments),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_cond: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    return {
        cond: {
            "n": len(rs),
            **{
                k: sum(r[k] for r in rs) / len(rs)
                for k in ("format_ok", "tool_ok", "args_valid", "args_match")
            },
        }
        for cond, rs in by_cond.items()
    }


def run_condition(
    model: ChatModel,
    condition: str,
    prompts: list[dict],
    tools: list[ToolSpec],
    *,
    think: bool,
    max_tokens: int,
) -> list[dict[str, Any]]:
    rows = []
    for item in prompts:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": item["prompt"]},
        ]
        result = model.chat(messages, tools, think=think, max_tokens=max_tokens)
        rows.append(
            {
                "condition": condition,
                "id": item["id"],
                "content": result.content,
                "tool_calls": [
                    {"name": c.name, "arguments": c.arguments} for c in result.tool_calls
                ],
                "parse_error": result.parse_error,
                "usage": result.usage,
                **score(item, result, tools),
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--precisions", default="4bit,bf16")
    p.add_argument("--adapter", default=ADAPTER_PATH)
    p.add_argument("--base-4bit", default=BASE_PATHS["4bit"])
    p.add_argument("--base-bf16", default=BASE_PATHS["bf16"], help="HF repo or directory")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--think", action="store_true")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--out", default="runs/exp1")
    args = p.parse_args()

    import mlx.core as mx

    from serana_agent.llm.mlx_local import LocalModel

    base_paths = {"4bit": args.base_4bit, "bf16": args.base_bf16}
    prompts = load_prompts()[: args.limit]
    tools = load_tools()
    out_dir = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for precision in args.precisions.split(","):
        lm = LocalModel(base_paths[precision], args.adapter)
        for adapter_on in (False, True):
            cond = f"{precision}/adapter_{'on' if adapter_on else 'off'}"
            print(f"running {cond}")
            role = lm.planner(adapter=adapter_on)
            rows += run_condition(
                role, cond, prompts, tools, think=args.think, max_tokens=args.max_tokens
            )
        del lm
        gc.collect()
        mx.clear_cache()

    (out_dir / "results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    summary = summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    for cond, s in summary.items():
        print(cond, {k: round(v, 3) for k, v in s.items()})
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
