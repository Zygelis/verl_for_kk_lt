from __future__ import annotations

import argparse
import ast
import json
import random
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from vllm import LLM, SamplingParams

from transformers import AutoTokenizer

from kk_lithuanian.kk_lt_reward_function import compute_score_qwen3


DEFAULT_MODEL_PATH = "kk_lithuanian/trained_models/qwen3-1.7B"
DEFAULT_VAL_PATH = "kk_lithuanian/data/qwen3_full/val.parquet"


def _load_structured_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist") and not isinstance(value, str):
        value = value.tolist()
    if isinstance(value, dict):
        return {key: _load_structured_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_load_structured_value(item) for item in value]
    if isinstance(value, tuple):
        return [_load_structured_value(item) for item in value]
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return None

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            continue

    return value


def _read_json_rows(dataset_path: Path) -> list[dict[str, Any]]:
    text = dataset_path.read_text(encoding="utf-8")
    text = text.strip()
    if not text:
        return []

    if text.startswith("["):
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise TypeError("JSON dataset must be a list of objects")
        return rows

    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _load_rows(dataset_path: Path, n: int | None, seed: int) -> list[dict[str, Any]]:
    if dataset_path.suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(dataset_path)
        if frame.empty:
            raise ValueError(f"Dataset is empty: {dataset_path}")
        frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        if n is not None:
            frame = frame.iloc[:n]
        rows_iter: Iterable[dict[str, Any]] = (r.to_dict() for _, r in frame.iterrows())
    elif dataset_path.suffix in {".json", ".jsonl"}:
        rows_iter = _read_json_rows(dataset_path)
        random.Random(seed).shuffle(rows_iter)
        if n is not None:
            rows_iter = rows_iter[:n]
    else:
        raise ValueError("Unsupported dataset format. Use .parquet, .json, or .jsonl")

    rows: list[dict[str, Any]] = []
    for row in rows_iter:
        row = dict(row)
        row["prompt"] = _load_structured_value(row.get("prompt"))
        reward_model = _load_structured_value(row.get("reward_model")) or {}
        extra_info = _load_structured_value(row.get("extra_info")) or {}
        if not isinstance(reward_model, dict):
            raise TypeError(f"reward_model field is not a dictionary: {type(reward_model)!r}")
        if not isinstance(extra_info, dict):
            raise TypeError(f"extra_info field is not a dictionary: {type(extra_info)!r}")
        row["reward_model"] = reward_model
        row["extra_info"] = extra_info
        rows.append(row)

    return rows


def _build_prompt_text(tokenizer: AutoTokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    prompt_lines: list[str] = []
    for message in messages:
        prompt_lines.append(f"{message['role'].upper()}: {message['content']}")
    prompt_lines.append("ASSISTANT:")
    return "\n".join(prompt_lines)


def _build_full_prompt_texts(
    tokenizer: AutoTokenizer,
    rows: list[dict[str, Any]],
) -> tuple[list[str], list[list[dict[str, str]]]]:
    prompt_texts: list[str] = []
    prompt_messages_list: list[list[dict[str, str]]] = []

    for row in rows:
        prompt_messages = row.get("prompt")
        if not isinstance(prompt_messages, list):
            raise TypeError(f"prompt field is not a list of messages: {type(prompt_messages)!r}")

        prompt_text = _build_prompt_text(tokenizer, prompt_messages)
        prompt_texts.append(prompt_text)
        prompt_messages_list.append(prompt_messages)

    return prompt_texts, prompt_messages_list


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--n", type=int, default=100, help="Process only the first N rows")
    parser.add_argument("--seed", type=int, default=111, help="Shuffle seed for dataset sampling")
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--success-threshold", type=float, default=1.0)
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional path to write a JSON report with all prompts, responses, rewards, and summary",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    rows = _load_rows(dataset_path, args.n, args.seed)
    if not rows:
        raise ValueError(f"Dataset is empty: {dataset_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_new_tokens,
        load_format="safetensors",
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    total = 0
    successes = 0
    records: list[dict[str, Any]] = []

    prompt_texts, prompt_messages_list = _build_full_prompt_texts(
        tokenizer=tokenizer,
        rows=rows,
    )

    for batch_start in range(0, len(prompt_texts), args.batch_size):
        batch_texts = prompt_texts[batch_start : batch_start + args.batch_size]
        outputs = llm.generate(batch_texts, sampling_params)

        for offset, output in enumerate(outputs):
            idx = batch_start + offset + 1
            row = rows[batch_start + offset]
            total += 1

            prompt_messages = prompt_messages_list[batch_start + offset]
            prompt_text = batch_texts[offset]
            response_text = output.outputs[0].text.strip()

            ground_truth = row.get("reward_model", {}).get("ground_truth", {})
            extra_info = row.get("extra_info", {})

            reward = compute_score_qwen3(
                data_source=row.get("data_source", ""),
                solution_str=response_text,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(reward, (int, float)) and reward > args.success_threshold:
                successes += 1

            records.append(
                {
                    "row_index": idx,
                    "data_source": row.get("data_source", ""),
                    "prompt": prompt_messages,
                    "prompt_text": prompt_text,
                    "response": response_text,
                    "reward": reward,
                    "ground_truth": ground_truth,
                    "extra_info": extra_info,
                }
            )

            print(f"=== ROW {idx} / {len(rows)} ===")
            print("PROMPT:")
            print(prompt_text)
            print()
            print("RESPONSE:")
            print(response_text)
            print()
            print("REWARD:", reward)
            print("GROUND TRUTH:")
            print(json.dumps(ground_truth, ensure_ascii=False, indent=2))
            print("-" * 60)

    print("=== SUMMARY ===")
    print(f"Processed: {total}")
    print(f"Successes (reward > {args.success_threshold}): {successes}")
    pct = (successes / total * 100) if total else 0.0
    print(f"Success rate: {pct:.1f}%")

    output_path = None
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        model_name = Path(args.model_path).name
        timestamp = int(time.time())
        output_path = Path(f"kk_lithuanian/kk_eval/{model_name}_reward_demo_{timestamp}.json")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "model_path": args.model_path,
            "dataset_path": str(dataset_path),
            "n": args.n,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "success_threshold": args.success_threshold,
            "total": total,
            "successes": successes,
            "success_rate": pct,
            "records": records,
        }
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
