import argparse
import json
from contextlib import contextmanager
import subprocess
import sys
from pathlib import Path


DEFAULT_CHECKPOINT_STEPS = [120, 210, 300]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and upload multiple FSDP checkpoints")
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints/verl_grpo_qwen3_1.7B/qwen3_1.7B_grpo_20260514_1158",
        help="Parent directory containing global_step_* checkpoint folders",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=DEFAULT_CHECKPOINT_STEPS,
        help="Checkpoint steps to process",
    )
    parser.add_argument(
        "--base-output-root",
        default="kk_lithuanian/trained_models",
        help="Root directory where merged model folders will be written",
    )
    parser.add_argument(
        "--hf-repo-name",
        default="qwen3-1.7B-lt-dapo-v1",
        help="Hugging Face repository name to upload each merged model to",
    )
    parser.add_argument(
        "--branch-prefix",
        default="step-",
        help="Branch name prefix; the step number is appended to this prefix",
    )
    parser.add_argument(
        "--main-branch",
        default="main",
        help="Branch name to use for the promoted final checkpoint",
    )
    parser.add_argument(
        "--main-step",
        type=int,
        default=None,
        help="Checkpoint step to also upload to the main branch. Defaults to the last step in --steps.",
    )
    parser.add_argument(
        "--base-model-name",
        default="qwen3-1.7B-lt-v1",
        help="Base model directory name under the output root",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype to use when merging LoRA into the base model",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map to use when loading the base model",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
        help="LoRA alpha to write into temporary training metadata before merging",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=64,
        help="LoRA rank to write into temporary training metadata before merging",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Only merge the checkpoints and skip Hugging Face upload",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


@contextmanager
def temporary_lora_train_meta(checkpoint_dir: Path, lora_alpha: int, lora_rank: int | None):
    meta_path = checkpoint_dir / "lora_train_meta.json"
    existed = meta_path.exists()
    original_text = meta_path.read_text(encoding="utf-8") if existed else None
    meta = {"lora_alpha": lora_alpha}
    if lora_rank is not None:
        meta["r"] = lora_rank
    meta_path.write_text(json.dumps(meta, indent=4), encoding="utf-8")
    try:
        yield
    finally:
        if existed and original_text is not None:
            meta_path.write_text(original_text, encoding="utf-8")
        elif meta_path.exists():
            meta_path.unlink()


def main() -> None:
    args = parse_args()
    main_step = args.main_step if args.main_step is not None else args.steps[-1]

    checkpoint_root = Path(args.checkpoint_root)
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"Checkpoint root not found: {checkpoint_root}")

    for step in args.steps:
        checkpoint_dir = checkpoint_root / f"global_step_{step}" / "actor"
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint folder not found: {checkpoint_dir}")

        merged_dir = Path(args.base_output_root) / f"{args.base_model_name}-global_step_{step}-merged"
        merged_hf_dir = merged_dir.with_name(f"{merged_dir.name}-hf")

        print(f"\n=== Processing global_step_{step} ===")
        with temporary_lora_train_meta(checkpoint_dir, args.lora_alpha, args.lora_rank):
            run_command(
                [
                    sys.executable,
                    "-m",
                    "verl.model_merger",
                    "merge",
                    "--backend",
                    "fsdp",
                    "--local_dir",
                    str(checkpoint_dir),
                    "--target_dir",
                    str(merged_dir),
                ]
            )

        run_command(
            [
                sys.executable,
                str(Path("kk_lithuanian") / "merge_lora_adapter_into_model.py"),
                "--base-model-dir",
                str(merged_dir),
                "--output-dir",
                str(merged_hf_dir),
                "--dtype",
                args.dtype,
                "--device-map",
                args.device_map,
            ]
        )

        if not args.skip_upload:
            branch = f"{args.branch_prefix}{step}"
            run_command(
                [
                    sys.executable,
                    str(Path("kk_lithuanian") / "upload_model_to_hf.py"),
                    "--repo-name",
                    args.hf_repo_name,
                    "--local-model-dir",
                    str(merged_hf_dir),
                    "--branch",
                    branch,
                ]
            )

            if step == main_step:
                run_command(
                    [
                        sys.executable,
                        str(Path("kk_lithuanian") / "upload_model_to_hf.py"),
                        "--repo-name",
                        args.hf_repo_name,
                        "--local-model-dir",
                        str(merged_hf_dir),
                        "--branch",
                        args.main_branch,
                    ]
                )

    print("\nAll requested checkpoints processed.")


if __name__ == "__main__":
    main()