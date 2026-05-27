"""Translate math benchmarks JSONL datasets into Lithuanian.

The script keeps the original `answer` and `id` fields and only translates
the `problem` field. It writes a new JSONL file that mirrors the input schema.

By default this uses the Google Gemini API because it is free to use for small usage.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_INPUT_FILE = Path("kk_lithuanian/math_eval/math_datasets/aime2026.jsonl")
DEFAULT_OUTPUT_FILE = Path("kk_lithuanian/math_eval/math_datasets/lt_versions/aime2026_lt.jsonl")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


SYSTEM_PROMPT = (
    "You translate math competition problems into Lithuanian. "
    "Translate only the natural language. Keep all LaTeX, numbers, symbols, "
    "and variable names exactly as written. Return only the translated problem text."
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file_handle:
        for row in rows:
            file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def translate_with_gemini(problem: str, model: str, api_key: str, base_url: str, max_retries: int = 3, backoff_factor: float = 2.0) -> str:
    """Translate a problem using Gemini API with exponential backoff retry logic.

    Args:
        problem: The problem text to translate
        model: Model name (e.g., "gemini-2.5-flash")
        api_key: Gemini API key
        base_url: Gemini API base URL
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff multiplier (wait time = backoff_factor^attempt)

    Returns:
        Translated problem text
    """
    # Build the Gemini API endpoint URL
    url = f"{base_url}/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": problem}
                ]
            }
        ],
        "systemInstruction": {
            "parts": [
                {"text": SYSTEM_PROMPT}
            ]
        },
        "generationConfig": {
            "temperature": 0.0
        }
    }

    data = json.dumps(payload).encode("utf-8")

    for attempt in range(max_retries + 1):
        try:
            req = request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                },
                method="POST",
            )
        
            with request.urlopen(req, timeout=120) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            
        except (error.HTTPError, TimeoutError) as exc:
            if attempt < max_retries:
                wait_time = backoff_factor ** attempt
                print(f"  Attempt {attempt + 1} failed, retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            else:
                if isinstance(exc, error.HTTPError):
                    details = exc.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"Gemini request failed after {max_retries + 1} attempts: {exc.code} {exc.reason}: {details}") from exc
                else:
                    raise RuntimeError(f"Gemini request timeout after {max_retries + 1} attempts") from exc

        candidates = response_payload.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini response did not contain any candidates: {response_payload}")

        content_obj = candidates[0].get("content") or {}
        parts = content_obj.get("parts") or []
        if not parts:
            raise RuntimeError(f"Gemini response content did not contain any parts: {response_payload}")
        
        content = parts[0].get("text")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"Gemini response did not contain translated text: {response_payload}")

        return content.strip()

    raise RuntimeError("Unreachable code - retry loop exhausted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate EN math problems to LT")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--base-url", type=str, default=GEMINI_BASE_URL)
    parser.add_argument("--api-key", type=str, default=os.environ.get("GEMINI_API_KEY"))
    parser.add_argument("--limit", type=int, default=None, help="Translate only the first N rows")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional delay between requests")
    parser.add_argument("--max-workers", type=int, default=8, help="Number of concurrent translation requests (default: 8)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.api_key:
        raise SystemExit("Missing Gemini API key. Set GEMINI_API_KEY or pass --api-key.")

    if not args.input_file.exists():
        raise SystemExit(f"Input file not found: {args.input_file}")

    rows = load_jsonl(args.input_file)
    if args.limit is not None:
        rows = rows[: args.limit]

    total = len(rows)
    translated_rows: dict[int, dict[str, Any]] = {}
    
    def translate_row(index: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Translate a single row and return (index, translated_row) tuple."""
        problem = row.get("problem")
        if not isinstance(problem, str):
            raise SystemExit(f"Row {index} is missing a string 'problem' field")

        print(f"[{index+1}/{total}] Translating problem id={row.get('id', 'unknown')}...")
        translated_problem = translate_with_gemini(problem, args.model, args.api_key, args.base_url)

        translated_row = dict(row)
        translated_row["problem"] = translated_problem
        
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)
        
        return (index, translated_row)

    # Use ThreadPoolExecutor to send multiple requests concurrently
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(translate_row, i, row): i 
            for i, row in enumerate(rows)
        }
        
        for future in as_completed(futures):
            index, translated_row = future.result()
            translated_rows[index] = translated_row

    # Reconstruct in original order
    ordered_rows = [translated_rows[i] for i in range(total)]

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(args.output_file, ordered_rows)
    print(f"Saved {len(ordered_rows)} rows to {args.output_file}")


if __name__ == "__main__":
    main()