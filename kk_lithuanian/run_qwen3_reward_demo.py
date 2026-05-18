#!/usr/bin/env python3
"""Generate Qwen3 responses on a subset of Lithuanian Knights/Knaves prompts.

This is a small inspection script, not a training entry point. It loads the first
N rows (or last N rows) from the prepared parquet dataset, renders the chat
prompt with the Qwen chat template, generates responses, prints them, and
evaluates them with the same reward function used by the training script.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kk_lithuanian.kk_lt_reward_function import compute_score_qwen3


DEFAULT_MODEL_PATH = "Qwen/Qwen3-1.7B"
DEFAULT_DATASET_PATH = Path("kk_lithuanian/data/qwen3_full/val.parquet")


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


def _load_rows(dataset_path: Path, n: int, from_back: bool = False) -> list[dict[str, Any]]:
    frame = pd.read_parquet(dataset_path)
    if frame.empty:
        raise ValueError(f"Dataset is empty: {dataset_path}")

    if from_back:
        selected = frame.iloc[-n:]
    else:
        selected = frame.iloc[:n]

    rows: list[dict[str, Any]] = []
    for _, r in selected.iterrows():
        row = r.to_dict()
        row["prompt"] = _load_structured_value(row.get("prompt"))
        reward_model = _load_structured_value(row.get("reward_model")) or {}
        extra_info = _load_structured_value(row.get("extra_info")) or {}

        if not isinstance(reward_model, dict):
            raise TypeError(f"reward_model column is not a dictionary: {type(reward_model)!r}")
        if not isinstance(extra_info, dict):
            raise TypeError(f"extra_info column is not a dictionary: {type(extra_info)!r}")

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


def _generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature,
        "top_p": top_p,
        "pad_token_id": tokenizer.eos_token_id,
    }

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    prompt_length = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][prompt_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--last-row", action="store_true", help="Load the last row of the dataset instead of the first")
    parser.add_argument("--n", type=int, default=10, help="Number of rows to process from the dataset")
    parser.add_argument("--from-back", action="store_true", help="Take rows from the end of the dataset (works with --n)")
    parser.add_argument("--custom-system-prompt", action="store_true", help="Use the hardcoded custom system prompt instead of the one from the dataset (for testing/debugging)")
    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)

    # Load requested rows
    rows = _load_rows(dataset_path, args.n, from_back=(args.from_back or args.last_row))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    DEFAULT_SYSTEM_PROMPT_QWEN3_INSTRUCT = (
        "Tu esi pagalbinis asistentas loginėms užduotims spręsti. "
        "Kad išspręstum užduotį, tu pirmiausia atlieki mąstymo procesą ir tada pateiki galutinį atsakymą. "
        "Mąstymo procesas turi būti logiškas ir konkretus (iki 400 žodžių), ir turi prieiti galutinį atsakymą. "
        "Mąstymo procesas ir atsakymas yra pateikti <think> mąstymo procesas čia </think> ir <answer> atsakymas čia </answer> žymėse. "
        "Dabar vartotojas prašo jūsų išspręsti loginę užduotį. "
    )

    total = 0
    matched_eq = 0

    for idx, row in enumerate(rows, start=1):
        total += 1

        prompt_messages = row["prompt"]
        if args.custom_system_prompt:
            prompt_messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT_QWEN3_INSTRUCT},
                *[msg for msg in prompt_messages if msg["role"] != "system"],
            ]
        if not isinstance(prompt_messages, list):
            raise TypeError(f"prompt column is not a list of messages: {type(prompt_messages)!r}")

        prompt_text = _build_prompt_text(tokenizer, prompt_messages)
        response_text = _generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        ground_truth = row["reward_model"].get("ground_truth", {})
        extra_info = row.get("extra_info", {})
        reward = compute_score_qwen3(
            data_source=row.get("data_source", ""),
            solution_str=response_text,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

        # Count matches where reward >= 1
        if isinstance(reward, (int, float)) and reward >= 1:
            matched_eq += 1

        print(f"=== ROW {idx} / {args.n} ===")
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

    # Summary
    print("=== SUMMARY ===")
    print(f"Processed: {total}")
    print(f"Matches (reward >= 1): {matched_eq}")
    pct = (matched_eq / total * 100) if total else 0.0
    print(f"Percentage: {pct:.1f}%")


if __name__ == "__main__":
    main()