"""Preprocess the Lithuanian Knights and Knaves dataset for RLHF training.

The output parquet follows the verl RLHF dataset format:

- ``data_source``: used by the reward function to identify the task
- ``prompt``: a chat message list, not a pre-rendered template string
- ``ability``: task category
- ``reward_model.ground_truth``: normalized target used by the reward function
- ``extra_info``: task metadata that the reward function can optionally use
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from datasets import Dataset

from verl.utils.hdfs_io import copy, makedirs


DEFAULT_SYSTEM_PROMPT_QWEN25 = (
    "Tu esi pagalbinis asistentas. Asistentas pirmiausia galvoja apie samprotavimo procesą ir tada pateikia atsakymą. "
    "Samprotavimo procesas ir atsakymas yra uždaryti <think> </think> ir <answer> </answer> žymėse, atitinkamai, t.y., "
    "<think> samprotavimo procesas čia </think><answer> atsakymas čia </answer>. "
    "Dabar vartotojas prašo jūsų išspręsti loginę samprotavimo problemą. "
)

DEFAULT_SYSTEM_PROMPT_QWEN3 = (
    "Tu esi pagalbinis asistentas. Asistentas pirmiausia galvoja apie samprotavimo procesą ir tada pateikia atsakymą. "
    "Dabar vartotojas prašo jūsų išspręsti loginę samprotavimo problemą. "
)

DEFAULT_SYSTEM_PROMPT_QWEN3_INSTRUCT = (
    "Tu esi pagalbinis asistentas loginėms užduotims spręsti. "
    "Kad išspręstum užduotį, tu pirmiausia atlieki mąstymo procesą ir tada pateiki galutinį atsakymą. "
    "Mąstymo procesas turi būti logiškas ir konkretus (iki 400 žodžių), ir turi prieiti galutinį atsakymą. "
    "Mąstymo procesas ir atsakymas yra pateikti <think> mąstymo procesas čia </think> ir <answer> atsakymas čia </answer> žymėse. "
    "Dabar vartotojas prašo jūsų išspręsti loginę užduotį. "
)
DEFAULT_USER_SUFFIX = (
    "Atlikęs mąstymo procesą padaryk išvadą ir aiškiai nurodyk kiekvieno personažo tapatybę <answer> </answer> žymėse "
    "JSON formato pavidalu, pvz.: <answer>{\"riteriai\": [\"Lukas\"], \"melagiai\": [\"Daiva\", \"Gintare\"]}</answer>."
)

DEFAULT_USER_SUFFIX_QWEN3_INSTRUCT = (
    "Pateik trumpą atsakymą. Jei naudoji samprotavimą, pateik jį <think> </think> žymėse. "
    "Galutinį atsakymą pateik <answer> </answer> žymėse JSON formatu, pvz.: "
    "<answer>{\"riteriai\": [\"Lukas\"], \"melagiai\": [\"Daiva\", \"Gintare\"]}</answer>."
)


def load_jsonl(path: str):
    with open(path, encoding="utf-8") as file_handle:
        for line in file_handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_messages(
    example: dict[str, Any],
    default_system_prompt: str,
    user_suffix: str,
    use_example_system_prompt: bool,
) -> list[dict[str, str]]:
    """Build verl chat messages from the source example."""

    system_prompt = example.get("system_prompt") if use_example_system_prompt else None
    system_prompt = system_prompt or default_system_prompt
    prompt_text = example["prompt"]
    user_prompt = f"{prompt_text}\n\n{user_suffix}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def extract_solution(example: dict[str, Any]) -> dict[str, Any]:
    """Normalize the target answer for reward computation.

    The reward function can compare the generated response against this
    canonical structure. We keep the original labels and also preserve useful
    metadata for later reward shaping or debugging.
    """

    answer = example.get("answer") or {}
    knights = sorted(answer.get("knights", []))
    knaves = sorted(answer.get("knaves", []))

    return {
        "knights": knights,
        "knaves": knaves,
        "answer_text": example.get("answer_text", ""),
        "statements": example.get("statements", []),
        "islanders": example.get("islanders", []),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="kk_lithuanian/data/qwen3_full/hard")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--data_path", default="kk_lithuanian/raw_data/kk_full/kk_lt_train_hard.jsonl")
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--val_size", type=int, default=0)
    parser.add_argument(
        "--model_version",
        default="qwen3_instruct",
        choices=["qwen25", "qwen3", "qwen3_instruct"],
        help="Model family for prompt defaults: qwen25 keeps explicit think+answer guidance; qwen3 avoids explicit think-tag instruction; qwen3_instruct uses a concise instruct prompt.",
    )
    parser.add_argument(
        "--use_example_system_prompt",
        action="store_true",
        help="Use system_prompt from the JSONL examples instead of the default system prompt.",
    )

    args = parser.parse_args()

    data_source = "kk_logic_lithuanian"
    train_size = args.train_size
    val_size = args.val_size
    if args.model_version == "qwen25":
        default_system_prompt = DEFAULT_SYSTEM_PROMPT_QWEN25
        user_suffix = DEFAULT_USER_SUFFIX
    elif args.model_version == "qwen3_instruct":
        default_system_prompt = DEFAULT_SYSTEM_PROMPT_QWEN3_INSTRUCT
        user_suffix = DEFAULT_USER_SUFFIX
    else:
        default_system_prompt = DEFAULT_SYSTEM_PROMPT_QWEN3
        user_suffix = DEFAULT_USER_SUFFIX

    raw_dataset = Dataset.from_generator(load_jsonl, gen_kwargs={"path": args.data_path})
    print(f"Loaded {len(raw_dataset)} examples from {args.data_path}")

    if len(raw_dataset) < train_size + val_size:
        print(f"Warning: dataset has {len(raw_dataset)} examples, but requested {train_size + val_size}")
        train_size = max(0, len(raw_dataset) - val_size)
        print(f"Adjusted to train_size={train_size}, val_size={val_size}")

    train_dataset = raw_dataset.select(range(train_size)) if train_size > 0 else None
    val_dataset = raw_dataset.select(range(train_size, train_size + val_size)) if val_size > 0 else None

    def make_map_fn(split: str):
        def process_fn(example: dict[str, Any], idx: int):
            resolved_system_prompt = (
                example.get("system_prompt") if args.use_example_system_prompt else default_system_prompt
            )
            return {
                "data_source": data_source,
                "prompt": build_messages(
                    example,
                    default_system_prompt=default_system_prompt,
                    user_suffix=user_suffix,
                    use_example_system_prompt=args.use_example_system_prompt,
                ),
                "ability": "logic",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": extract_solution(example),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "id": example.get("id", idx),
                    "difficulty": example.get("difficulty", "unknown"),
                    "islanders": example.get("islanders", []),
                    "statements": example.get("statements", []),
                    "answer_text": example.get("answer_text", ""),
                    "reasoning_text": example.get("reasoning_text", ""),
                    "system_prompt": resolved_system_prompt,
                    "model_version": args.model_version,
                },
            }

        return process_fn


    if train_dataset is not None:
        train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    if val_dataset is not None:
        val_dataset = val_dataset.map(function=make_map_fn("val"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    if train_dataset is not None:
        print(f"\nSaving train dataset to {os.path.join(local_dir, 'train.parquet')}")
        train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    if val_dataset is not None:
        print(f"Saving validation dataset to {os.path.join(local_dir, 'val.parquet')}")
        val_dataset.to_parquet(os.path.join(local_dir, "val.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print(f"Copied to HDFS: {hdfs_dir}")

    print("Data preprocessing complete!")
