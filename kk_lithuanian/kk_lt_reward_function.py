# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reward function for the Lithuanian Knights and Knaves dataset.

The scorer is intentionally permissive:
- it accepts JSON, Python-literal-style objects, or tagged ``<answer>`` blocks
- it tolerates key aliases like ``riteriai`` / ``melagiai`` / ``knights`` / ``knaves``
- it accepts both list and string values
- it normalizes names before comparing set membership
- it gives a separate format reward and content reward

The final score is a scalar so it stays compatible with DAPO, GRPO-naive,
and other verl reward managers that only expect a numeric reward.
"""

from __future__ import annotations

from ast import literal_eval
import json
import re
import unicodedata
from typing import Any


KNIGHT_KEY_ALIASES = {
    "knight",
    "knights",
    "riteris",
    "riteriai",
    "ryteris",
    "ryteriai",
    "good",
    "truth",
    "truthful",
}

KNAVE_KEY_ALIASES = {
    "knave",
    "knaves",
    "melagis",
    "melagiai",
    "liar",
    "liars",
    "false",
    "lying",
    "bad",
}


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _normalize_text(value: Any) -> str:
    text = _to_text(value).strip()
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = "".join(char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char))
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_candidate_payload(solution_str: str) -> str:
    text = _to_text(solution_str).strip()

    answer_matches = list(re.finditer(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL))
    if answer_matches:
        return answer_matches[-1].group(1).strip()

    brace_start = text.find("{")
    if brace_start != -1:
        balance = 0
        for index in range(brace_start, len(text)):
            if text[index] == "{":
                balance += 1
            elif text[index] == "}":
                balance -= 1
                if balance == 0:
                    return text[brace_start : index + 1]

    return text


def _parse_structured_answer(solution_str: str) -> Any:
    payload = _extract_candidate_payload(solution_str)
    if not payload:
        return None

    for parser in (json.loads, literal_eval):
        try:
            return parser(payload)
        except Exception:
            pass

    return None


def _extract_tag_blocks(text: str, tag: str) -> list[str]:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    return [match.group(1).strip() for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL)]


def _validate_response_structure(solution_str: str) -> dict[str, Any]:
    text = _to_text(solution_str)
    think_blocks = _extract_tag_blocks(text, "think")
    answer_blocks = _extract_tag_blocks(text, "answer")

    think_start_count = text.count("<think>")
    think_end_count = text.count("</think>")
    answer_start_count = text.count("<answer>")
    answer_end_count = text.count("</answer>")

    answer_block_ok = len(answer_blocks) > 0
    think_block_ok = len(think_blocks) > 0

    # Choose the latest parsable answer payload. If the final answer block is
    # invalid, fallback to the latest earlier valid one.
    payload = ""
    parsed_payload = None
    for candidate in reversed(answer_blocks):
        candidate_parsed = _parse_structured_answer(candidate)
        if isinstance(candidate_parsed, (dict, list)):
            payload = candidate
            parsed_payload = candidate_parsed
            break

    think_present = think_block_ok

    return {
        "answer_block_ok": answer_block_ok,
        "think_present": think_present,
        "think_block_ok": think_block_ok,
        "format_ok": answer_block_ok and think_block_ok,
        "parsed_payload": parsed_payload,
        "answer_payload": payload,
        "answer_start_count": answer_start_count,
        "answer_end_count": answer_end_count,
        "think_start_count": think_start_count,
        "think_end_count": think_end_count,
    }


def _split_names(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list | tuple | set):
        items = value
    else:
        text = _to_text(value).replace(" and ", ",")
        text = text.replace(";", ",")
        items = re.split(r"[,/|]", text)

    names: set[str] = set()
    for item in items:
        normalized = _normalize_text(item)
        if normalized:
            names.add(normalized)
    return names


def _empty_prediction() -> dict[str, set[str]]:
    return {"knights": set(), "knaves": set()}


def _alias_to_category(key: Any) -> str | None:
    normalized = _normalize_text(key)
    if normalized in KNIGHT_KEY_ALIASES:
        return "knights"
    if normalized in KNAVE_KEY_ALIASES:
        return "knaves"
    return None


def _extract_prediction_sets(solution_str: str) -> dict[str, set[str]]:
    parsed = _parse_structured_answer(solution_str)
    prediction = _empty_prediction()

    if parsed is None:
        return prediction

    if isinstance(parsed, dict):
        for key, value in parsed.items():
            category = _alias_to_category(key)
            if category is not None:
                prediction[category].update(_split_names(value))
                continue

            inferred_category = None
            if isinstance(value, str):
                inferred_category = _alias_to_category(value)
            elif isinstance(value, list | tuple | set):
                inferred_category = None

            if inferred_category is not None:
                prediction[inferred_category].add(_normalize_text(key))

        return prediction

    if isinstance(parsed, list):
        # Allow a sequence of objects or a sequence of names assigned to one class.
        for item in parsed:
            if isinstance(item, dict):
                nested = _extract_prediction_sets(json.dumps(item, ensure_ascii=False))
                prediction["knights"].update(nested["knights"])
                prediction["knaves"].update(nested["knaves"])
            else:
                prediction["knights"].add(_normalize_text(item))

    return prediction


def _extract_ground_truth_sets(ground_truth: Any, extra_info: dict[str, Any]) -> dict[str, set[str]]:
    if isinstance(ground_truth, dict):
        knights = ground_truth.get("knights", [])
        knaves = ground_truth.get("knaves", [])
        roster = ground_truth.get("islanders", [])
    else:
        knights = []
        knaves = []
        roster = []

    if not roster:
        roster = extra_info.get("islanders", [])

    return {
        "knights": {_normalize_text(name) for name in knights if _normalize_text(name)},
        "knaves": {_normalize_text(name) for name in knaves if _normalize_text(name)},
        "roster": {_normalize_text(name) for name in roster if _normalize_text(name)},
    }


def _canonicalize_prediction(prediction: dict[str, set[str]], roster: set[str]) -> dict[str, set[str]]:
    canonical = _empty_prediction()
    for category in ("knights", "knaves"):
        for name in prediction[category]:
            if roster and name not in roster:
                continue
            canonical[category].add(name)
    return canonical


def _score_prediction(prediction: dict[str, set[str]], ground_truth: dict[str, set[str]]) -> tuple[float, dict[str, Any]]:
    roster = ground_truth["roster"]
    prediction = _canonicalize_prediction(prediction, roster)

    gt_knights = ground_truth["knights"]
    gt_knaves = ground_truth["knaves"]
    predicted_knights = prediction["knights"]
    predicted_knaves = prediction["knaves"]

    all_names = roster or (gt_knights | gt_knaves | predicted_knights | predicted_knaves)
    if not all_names:
        return 0.0, {
            "exact_match": False,
            "parsed": False,
            "predicted_knights": [],
            "predicted_knaves": [],
            "ground_truth_knights": [],
            "ground_truth_knaves": [],
            "accuracy": 0.0,
        }

    correct = 0
    for name in all_names:
        gt_is_knight = name in gt_knights
        pred_is_knight = name in predicted_knights
        gt_is_knave = name in gt_knaves
        pred_is_knave = name in predicted_knaves

        if gt_is_knight and pred_is_knight:
            correct += 1
        elif gt_is_knave and pred_is_knave:
            correct += 1

    accuracy = correct / len(all_names)
    exact_match = predicted_knights == gt_knights and predicted_knaves == gt_knaves
    if exact_match:
        accuracy = 1.0

    return accuracy, {
        "exact_match": exact_match,
        "parsed": True,
        "predicted_knights": sorted(predicted_knights),
        "predicted_knaves": sorted(predicted_knaves),
        "ground_truth_knights": sorted(gt_knights),
        "ground_truth_knaves": sorted(gt_knaves),
        "num_names": len(all_names),
        "num_correct": correct,
        "accuracy": accuracy,
    }


def compute_score_qwen25(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    format_reward: float = 1.0,
    answer_reward: float = 2.0,
    **kwargs,
):
    """Compute reward for Qwen2.5 (requires explicit think tags).
    
    Uses tolerant multi-block validation:
    - Require at least one properly closed <think> and one <answer> block
    - Do not penalize extra think/answer blocks
    - Score using the latest valid parsable <answer> block
    - Answer: +2 exact match, -1.5 mismatch, -2 parse failure
    
    Since Qwen2.5 has no native thinking, rewarding think tags encourages reasoning.
    """
    extra_info = extra_info or {}
    structure_info = _validate_response_structure(solution_str)
    ground_truth_sets = _extract_ground_truth_sets(ground_truth, extra_info)

    # Require at least one properly closed think and answer block.
    format_ok = structure_info["answer_block_ok"] and structure_info["think_block_ok"]
    format_score = format_reward if format_ok else -abs(format_reward)

    # Answer validation: content scoring
    answer_score = -abs(answer_reward)
    if format_ok:
        answer_payload = structure_info["answer_payload"]
        predicted_sets = _extract_prediction_sets(answer_payload)
        parsed_payload = structure_info["parsed_payload"]
        
        if parsed_payload is not None:
            accuracy, parsed_info = _score_prediction(predicted_sets, ground_truth_sets)
            if parsed_info["exact_match"]:
                answer_score = answer_reward  # +2
            else:
                answer_score = -1.5  # Mismatch penalty (LogicRL style)
        else:
            answer_score = -2.0  # Parse failure penalty (LogicRL style)
            parsed_info = {
                "parsed": False,
                "exact_match": False,
                "predicted_knights": [],
                "predicted_knaves": [],
                "ground_truth_knights": sorted(ground_truth_sets["knights"]),
                "ground_truth_knaves": sorted(ground_truth_sets["knaves"]),
                "num_names": len(ground_truth_sets["roster"]),
                "num_correct": 0,
                "accuracy": 0.0,
            }
    else:
        parsed_info = {
            "parsed": False,
            "exact_match": False,
            "predicted_knights": [],
            "predicted_knaves": [],
            "ground_truth_knights": sorted(ground_truth_sets["knights"]),
            "ground_truth_knaves": sorted(ground_truth_sets["knaves"]),
            "num_names": len(ground_truth_sets["roster"]),
            "num_correct": 0,
            "accuracy": 0.0,
        }

    total_score = format_score + answer_score
    return float(total_score)


def compute_score_qwen3_instruct(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    format_reward: float = 0.4,
    answer_reward: float = 1.0,
    think_min_chars: int = 512,
    think_max_chars: int = 8192,
    think_length_reward: float = 0.5,
    **kwargs,
):
    """Compute reward for Qwen3 Instruct (think tags optional).

    Format reward is split into two independent checks:
    - +0.4 if at least one properly closed <think> block is present
    - +0.4 if at least one properly closed <answer> block is present

    Answer reward mirrors Qwen2.5 scoring:
    - +1 exact match, -0.5 mismatch, -1 parse failure

    Optional think-length shaping:
    - If a <think> block is present, reward/penalize by its length in chars.
    - +think_length_reward if within [think_min_chars, think_max_chars]
    - -think_length_reward otherwise
    """
    extra_info = extra_info or {}
    structure_info = _validate_response_structure(solution_str)
    ground_truth_sets = _extract_ground_truth_sets(ground_truth, extra_info)

    think_block_ok = structure_info["think_block_ok"]
    answer_block_ok = structure_info["answer_block_ok"]

    format_score = 0.0
    if think_block_ok:
        format_score += 0
    else:
        format_score -= format_reward
    if answer_block_ok:
        format_score += format_reward
    else:
        format_score -= (format_reward*2)  # times two penalty for missing answer block since it's more critical

    answer_score = -abs(answer_reward)
    if answer_block_ok:
        answer_payload = structure_info["answer_payload"]
        predicted_sets = _extract_prediction_sets(answer_payload)
        parsed_payload = structure_info["parsed_payload"]

        if parsed_payload is not None:
            accuracy, parsed_info = _score_prediction(predicted_sets, ground_truth_sets)
            if parsed_info["exact_match"]:
                answer_score = answer_reward
            else:
                answer_score = -0.5 * abs(answer_reward)
        else:
            answer_score = -abs(answer_reward)
            parsed_info = {
                "parsed": False,
                "exact_match": False,
                "predicted_knights": [],
                "predicted_knaves": [],
                "ground_truth_knights": sorted(ground_truth_sets["knights"]),
                "ground_truth_knaves": sorted(ground_truth_sets["knaves"]),
                "num_names": len(ground_truth_sets["roster"]),
                "num_correct": 0,
                "accuracy": 0.0,
            }
    else:
        parsed_info = {
            "parsed": False,
            "exact_match": False,
            "predicted_knights": [],
            "predicted_knaves": [],
            "ground_truth_knights": sorted(ground_truth_sets["knights"]),
            "ground_truth_knaves": sorted(ground_truth_sets["knaves"]),
            "num_names": len(ground_truth_sets["roster"]),
            "num_correct": 0,
            "accuracy": 0.0,
        }

    think_length_score = 0.0
    if think_block_ok:
        think_blocks = _extract_tag_blocks(_to_text(solution_str), "think")
        if think_blocks:
            think_text = think_blocks[-1]
            think_len = len(_to_text(think_text))
            if think_min_chars <= think_len <= think_max_chars:
                think_length_score = 0.0
            else:
                think_length_score = -abs(think_length_reward)



    total_score = format_score + answer_score + think_length_score


    # # ---- DEBUG BLOCK ----
    # # We only print occasionally so we don't flood the terminal
    # import random
    # if random.random() < 0.1: # Print 10% of the time
    #     #print("\n" + "="*50)
    #     #print(f"--- MODEL OUTPUT PREVIEW ---")
    #     #print(f"\n{solution_str[-1000:]}\n") # Print the last 300 chars of the solution for debugging
    #     #print(f"--- SCORE BREAKDOWN ---")
    #     print(f"Format Score: {format_score}")
    #     print(f"Answer Score: {answer_score}")
    #     print(f"Think Length Score: {think_length_score}")
    #     print(f"Total: {total_score}")
    #     #print("="*50 + "\n")
    # # ---------------------


    return float(total_score)


def compute_score_qwen3(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    format_reward: float = 0.3,
    answer_reward: float = 1.0,
    think_min_chars: int = 1024,
    think_max_chars: int = 8192,
    think_length_reward: float = 0.5,
    **kwargs,
):
    """Compute reward for Qwen3 with stricter answer scoring.

        - Format: reward/penalty for <answer> tags
        - Think length: reward/penalty for being within [think_min_chars, think_max_chars]
        - Answer: +answer_reward for exact match, -0.5 * answer_reward for
            accuracy >= 0.5, and -answer_reward for accuracy < 0.5
    - Parse failure: -answer_reward
    """
    extra_info = extra_info or {}
    structure_info = _validate_response_structure(solution_str)
    ground_truth_sets = _extract_ground_truth_sets(ground_truth, extra_info)

    answer_block_ok = structure_info["answer_block_ok"]

    format_score = format_reward if answer_block_ok else -abs(format_reward)

    answer_score = -abs(answer_reward)
    if answer_block_ok:
        answer_payload = structure_info["answer_payload"]
        predicted_sets = _extract_prediction_sets(answer_payload)
        parsed_payload = structure_info["parsed_payload"]

        if parsed_payload is not None:
            accuracy, parsed_info = _score_prediction(predicted_sets, ground_truth_sets)
            if parsed_info["exact_match"]:
                answer_score = answer_reward
            elif accuracy >= 0.5:
                answer_score = -0.5 * abs(answer_reward)
            else:
                answer_score = -abs(answer_reward)
        else:
            answer_score = -abs(answer_reward)
            parsed_info = {
                "parsed": False,
                "exact_match": False,
                "predicted_knights": [],
                "predicted_knaves": [],
                "ground_truth_knights": sorted(ground_truth_sets["knights"]),
                "ground_truth_knaves": sorted(ground_truth_sets["knaves"]),
                "num_names": len(ground_truth_sets["roster"]),
                "num_correct": 0,
                "accuracy": 0.0,
            }
    else:
        parsed_info = {
            "parsed": False,
            "exact_match": False,
            "predicted_knights": [],
            "predicted_knaves": [],
            "ground_truth_knights": sorted(ground_truth_sets["knights"]),
            "ground_truth_knaves": sorted(ground_truth_sets["knaves"]),
            "num_names": len(ground_truth_sets["roster"]),
            "num_correct": 0,
            "accuracy": 0.0,
        }

    think_length_score = -abs(think_length_reward)
    if structure_info["think_block_ok"]:
        think_blocks = _extract_tag_blocks(_to_text(solution_str), "think")
        if think_blocks:
            think_text = think_blocks[-1]
            think_len = len(_to_text(think_text))
            if think_min_chars <= think_len <= think_max_chars:
                think_length_score = abs(think_length_reward)

    total_score = format_score + think_length_score + answer_score
    return float(total_score)


# Default entry point (for backward compatibility)
compute_score = compute_score_qwen3