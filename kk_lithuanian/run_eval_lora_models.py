from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODELS_LIST: list[str] = [
    "Qwen/Qwen3-1.7B",
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


def _build_output_path(output_dir: Path, model_path: str) -> Path:
    parts = Path(model_path).parts
    method = None
    if "trained_models" in parts:
        idx = parts.index("trained_models")
        if idx + 1 < len(parts):
            method = parts[idx + 1]

    model_name = Path(model_path).name
    step = _extract_step(model_name)

    if method:
        if step:
            output_name = f"{method}_global_step_{step}_kk_eval.json"
        else:
            output_name = f"{method}_kk_eval.json"
    else:
        output_name = "vanilla_base_kk_eval.json"

    return output_dir / output_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run eval_lora_model.py for multiple model paths with default args.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="One or more model paths to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default="kk_lithuanian/kk_eval",
        help="Directory for per-model JSON reports.",
    )
    parser.add_argument(
        "--eval-script",
        default="kk_lithuanian/eval_lora_model.py",
        help="Path to the eval script.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_script = Path(args.eval_script)
    if not eval_script.exists():
        raise FileNotFoundError(f"Eval script not found: {eval_script}")

    model_paths = args.models or MODELS_LIST
    if not model_paths:
        raise ValueError("No models specified. Use --models or populate MODELS_LIST.")

    for model_path in model_paths:
        output_path = _build_output_path(output_dir, model_path)
        cmd = [
            sys.executable,
            str(eval_script),
            "--model-path",
            model_path,
            "--output-path",
            str(output_path),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
