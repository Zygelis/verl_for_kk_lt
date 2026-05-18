import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL_DIR = "kk_lithuanian/trained_models/qwen3-1.7B-lt-v1"
DEFAULT_OUTPUT_DIR = "kk_lithuanian/trained_models/qwen3-1.7B-lt-v1-merged"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a saved LoRA adapter into a base model checkpoint")
    parser.add_argument(
        "--base-model-dir",
        default=DEFAULT_BASE_MODEL_DIR,
        help="Directory containing the saved HF base model and lora_adapter/",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the merged model",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype to use when loading the base model",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map to use when loading the base model",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def main() -> None:
    args = parse_args()
    base_model_dir = args.base_model_dir
    lora_adapter_dir = os.path.join(base_model_dir, "lora_adapter")
    output_dir = args.output_dir

    if not os.path.isdir(base_model_dir):
        raise FileNotFoundError(f"Base model folder not found: {base_model_dir}")
    if not os.path.isdir(lora_adapter_dir):
        raise FileNotFoundError(f"LoRA adapter folder not found: {lora_adapter_dir}")

    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_adapter_dir)

    print("Merging LoRA into base weights...")
    model = model.merge_and_unload()
    model.eval()
    model = model.to("cpu")

    print(f"Saving merged model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
