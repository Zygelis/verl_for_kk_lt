"""
Run evaluation for multiple models across multiple datasets.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODELS_LIST: list[str] = [
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "kk_lithuanian/trained_models/dapo_v2/qwen3-1.7B-kk-dapo-v2-global_step_240-merged-hf",
    "kk_lithuanian/trained_models/dapo_v2/qwen3-1.7B-kk-dapo-v2-global_step_270-merged-hf",
    "kk_lithuanian/trained_models/dapo_v2/qwen3-1.7B-kk-dapo-v2-global_step_300-merged-hf",
    "kk_lithuanian/trained_models/dapo_v2/qwen3-1.7B-kk-dapo-v2-global_step_312-merged-hf",
    "kk_lithuanian/trained_models/grpo_v2/qwen3-1.7B-kk-dapo-v2-global_step_300-merged-hf",
    "kk_lithuanian/trained_models/grpo_v2/qwen3-1.7B-kk-dapo-v2-global_step_312-merged-hf",
    "kk_lithuanian/trained_models/repp_v2/qwen3-1.7B-kk-dapo-v2-global_step_120-merged-hf",
    "kk_lithuanian/trained_models/repp_v2/qwen3-1.7B-kk-dapo-v2-global_step_150-merged-hf",
    "kk_lithuanian/trained_models/repp_v2/qwen3-1.7B-kk-dapo-v2-global_step_180-merged-hf",
]

DATASETS = [
    ("aime2024_lt", "kk_lithuanian/math_eval/math_datasets/lt_versions/aime2024_lt.jsonl", "aime"),
    ("aime2025_lt", "kk_lithuanian/math_eval/math_datasets/lt_versions/aime2025_lt.jsonl", "aime"),
    ("aime2026_lt", "kk_lithuanian/math_eval/math_datasets/lt_versions/aime2026_lt.jsonl", "aime"),
    ("amc_lt", "kk_lithuanian/math_eval/math_datasets/lt_versions/amc_lt.jsonl", "amc"),
    ("gsm8k_main_lt", "kk_lithuanian/math_eval/math_datasets/lt_versions/gsm8k_main_lt.jsonl", "gsm8k"),
    ("gsm8k_socratic_lt", "kk_lithuanian/math_eval/math_datasets/lt_versions/gsm8k_socratic_lt.jsonl", "gsm8k"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate models on multiple datasets")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=15,
        help="Batch size to pass to test_aime_or_amc.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    return parser.parse_args()


def _extract_step(model_name: str) -> str | None:
    marker = "global_step_"
    if marker not in model_name:
        return None
    tail = model_name.split(marker, 1)[1]
    step = ""
    for ch in tail:
        if ch.isdigit():
            step += ch
        else:
            break
    return step or None


def _build_model_entries(model_paths: list[str]) -> list[tuple[str, str, str | None]]:
    entries: list[tuple[str, str, str | None]] = []
    for model_path in model_paths:
        parts = Path(model_path).parts
        method = None
        if "trained_models" in parts:
            idx = parts.index("trained_models")
            if idx + 1 < len(parts):
                method = parts[idx + 1]

        if method:
            model_name = method
        else:
            base_name = Path(model_path).name
            normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in base_name)
            model_name = f"vanilla_{normalized}"
        step = _extract_step(Path(model_path).name)
        suffix = f"global_step_{step}" if step else None
        entries.append((model_name, model_path, suffix))
    return entries


def _results_filename(model_name: str, dataset_name: str, suffix: str | None) -> str:
    if suffix:
        return f"{model_name}_{dataset_name}_{suffix}.jsonl"
    return f"{model_name}_{dataset_name}.jsonl"


def main() -> None:
    args = parse_args()

    script_path = Path(__file__).resolve().parent / "test_aime_or_amc.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing evaluator script: {script_path}")

    model_entries = _build_model_entries(MODELS_LIST)

    for dataset_name, dataset_path, dataset_type in DATASETS:
        output_dir = f"kk_lithuanian/math_eval/eval_results/lithuanian_full_eval_3072/{dataset_name}"
        for model_name, model_path, suffix in model_entries:
            results_filename = _results_filename(model_name, dataset_name, suffix)
            cmd = [
                sys.executable,
                str(script_path),
                "--model-path",
                model_path,
                "--dataset-path",
                dataset_path,
                "--dataset-type",
                dataset_type,
                "--batch-size",
                str(args.batch_size),
                "--output-dir",
                output_dir,
                "--results-filename",
                results_filename,
            ]
            print("Running:", " ".join(cmd))
            if args.dry_run:
                continue
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
