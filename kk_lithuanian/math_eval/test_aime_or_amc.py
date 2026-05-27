"""
AIME evaluation script using vLLM inference.
Evaluates a language model on the AIME dataset with streaming inference.
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams



SYSTEM_PROMPT = (
    "Tu esi pagalbinis asistentas loginėms užduotims spręsti. "
    "Kad išspręstum užduotį, tu pirmiausia atlieki mąstymo procesą ir tada pateiki galutinį atsakymą. "
    "Mąstymo procesas turi būti logiškas ir konkretus (iki 400 žodžių), ir turi prieiti galutinį atsakymą. "
    "Mąstymo procesas ir atsakymas yra pateikti <think> mąstymo procesas čia </think> ir <answer> atsakymas čia </answer> žymėse. "
    "Dabar vartotojas prašo jūsų išspręsti loginę užduotį. "
)


# Note: datasets are normalized to use `problem` and `answer` fields.
# We keep a small fallback to `question` for safety.


def _load_dataset_rows(dataset_path: Path) -> list[dict]:
    if dataset_path.suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(dataset_path)
        if frame.empty:
            raise ValueError(f"Dataset is empty: {dataset_path}")
        return frame.to_dict(orient="records")

    if dataset_path.suffix in {".json", ".jsonl"}:
        with open(dataset_path, encoding="utf-8") as file:
            return [json.loads(line) for line in file.readlines() if line.strip()]

    raise ValueError("Unsupported dataset format. Use .parquet, .pq, .json, or .jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on the AIME/AMC/GSM8K dataset")
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="aime",
        choices=["aime", "amc", "gsm8k"],
        help="Dataset type to evaluate on (default: aime)",
    )
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model (HF repo or local)")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the AIME, AMC, or GSM8K dataset file (.parquet, .pq, .json, or .jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="kk_lithuanian/math_eval/eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--results-filename",
        type=str,
        default=None,
        help="Optional output JSON filename (defaults to {dataset}_step_{step}_results.json)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6 from training)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling (default: -1 to disable)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3072,
        help="Maximum tokens to generate per response",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of prompts to generate per vLLM batch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # System prompt
    system_prompt = SYSTEM_PROMPT

    # Validate inputs
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    # Extract step/identifier from model path for results naming
    step = re.search(r"(\d+)$", args.model_path)
    step_str = step.group(1) if step else "unknown"

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_tokens + 1024,  # allow for prompt + generation
        load_format="safetensors",
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    print(f"Loading dataset from: {dataset_path}")
    data = _load_dataset_rows(dataset_path)
    print(f"Evaluating {len(data)} {args.dataset_type.upper()} problems...")
    
    correct_count = 0
    total_time = 0.0
    results = []
    tokenizer = llm.get_tokenizer()

    texts: list[str] = []
    prompts: list[str] = []
    full_prompts: list[str] = []
    expected_answers = []
    for problem in data:
        prompt = problem.get("problem") or problem.get("question")
        if prompt is None:
            raise TypeError("Dataset row missing 'problem' or 'question' field")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)
        prompts.append(prompt)
        full_prompts.append(text)
        expected_answers.append(problem.get("answer"))

    total_batches = math.ceil(len(texts) / args.batch_size) if texts else 0
    for batch_index in tqdm(range(0, len(texts), args.batch_size), desc="Evaluating", total=total_batches):
        batch_texts = texts[batch_index : batch_index + args.batch_size]
        start_time = time.time()
        outputs = llm.generate(batch_texts, sampling_params)
        elapsed_time = time.time() - start_time
        per_item_time = elapsed_time / max(len(batch_texts), 1)

        for offset, output in enumerate(outputs):
            response = output.outputs[0].text.strip()
            prompt = prompts[batch_index + offset]
            full_prompt = full_prompts[batch_index + offset]
            expected_answer = expected_answers[batch_index + offset]

            # Extract answer from <answer> tags (try closed tag first)
            match = re.search(r"<answer>(.*?)</answer>", response, re.S)
            if match:
                extracted = match.group(1).strip()
            elif "<answer>" in response:
                extracted = response.split("<answer>", 1)[1].strip()
            else:
                # Fallback: take last 50 chars if no tag found
                extracted = response[-50:].strip()

            # Normalize expected answer forms (handle 144.0 vs 144, etc.)
            def _expected_forms(exp):
                if isinstance(exp, (int, float)):
                    if isinstance(exp, float) and exp.is_integer():
                        return [str(int(exp)), str(float(exp))]
                    return [str(exp)]
                # try to parse numeric strings as numbers too
                if isinstance(exp, str):
                    s = exp.strip()
                    try:
                        f = float(s)
                    except Exception:
                        return [s]
                    if f.is_integer():
                        return [str(int(f)), str(f), s]
                    return [s, str(f)]
                return [str(exp)]

            forms = _expected_forms(expected_answer)
            is_correct = False
            for form in forms:
                if re.search(r"\b" + re.escape(form) + r"\b", extracted):
                    is_correct = True
                    break

            result = {
                "question": prompt,
                "full_prompt": full_prompt,
                "expected_answer": expected_answer,
                "generated_output": response,
                "extracted_answer": extracted,
                "correct": is_correct,
                "time_taken_seconds": per_item_time,
            }

            results.append(result)

            if is_correct:
                correct_count += 1

        total_time += elapsed_time

    # Compute accuracy
    accuracy = correct_count / len(data) if data else 0.0
    avg_time = total_time / len(data) if data else 0.0

    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Results ({args.dataset_type.upper()}, Model: step-{step_str})")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.1%} ({correct_count}/{len(data)})")
    print(f"Average time per problem: {avg_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"{'='*60}\n")

    # Save results
    results_filename = (
        args.results_filename
        if args.results_filename
        else f"{args.dataset_type}_step_{step_str}_results.json"
    )
    results_file = output_dir / results_filename
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": args.model_path,
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_problems": len(data),
                "avg_time_per_problem": avg_time,
                "total_time": total_time,
                "sampling_params": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "max_tokens": args.max_tokens,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()