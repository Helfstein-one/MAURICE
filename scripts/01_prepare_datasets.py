#!/usr/bin/env python3
"""
MAURICE Dataset Preparation Pipeline (scripts/01_prepare_datasets.py)

Filters and formats datasets into ChatML JSONL format with calibrated <think> tags.
- Variant 'c' (Code & Refactor): Validates syntax (Python AST, C/Rust/Go patterns) and structures unified diff patches.
- Variant 'r' (Pure Reasoning): Preserves multi-step logic with explicit <think>...</think> blocks.
- Variant 'g' (General Purpose): Injects minimal <think>\n</think> tags for simple instructions to suppress overthinking.
"""

import argparse
import ast
import json
import os
from typing import Any

SYSTEM_PROMPTS = {
    "c": (
        "You are mau-llm-1.0-c, an expert code and refactoring engine. "
        "Provide clean, syntactically verified code, unified diff patches, "
        "and structural refactoring instructions."
    ),
    "r": (
        "You are mau-llm-1.0-r, a pure reasoning engine. "
        "Think carefully before answering by placing your step-by-step "
        "reasoning process inside <think>...</think> tags."
    ),
    "g": (
        "You are mau-llm-1.0-g, a versatile general-purpose assistant. "
        "Provide concise and accurate responses, using minimal thinking when appropriate."
    ),
}

RAW_FILE_PATHS = {
    "c": "data/raw/python_code_instructions.jsonl",
    "r": "data/raw/gsm8k_train.jsonl",
    "g": "data/raw/ultrachat_sample.jsonl",
}


def load_hf_dataset(name: str, split: str = "train") -> Any:
    """Loads a HuggingFace dataset split."""
    from datasets import load_dataset

    if name == "openai/gsm8k":
        return load_dataset(name, "main", split=split)
    return load_dataset(name, split=split)


def validate_code_syntax(code: str, language: str = "python") -> bool:
    """Validates code syntax using ast for Python and basic pattern checks for C/Rust/Go."""
    if not code or not isinstance(code, str):
        return False

    language = language.lower()
    if language in ["python", "py"]:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    elif language in ["c", "cpp", "c++", "rust", "go"]:
        brace_count = 0
        paren_count = 0
        for char in code:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
            elif char == "(":
                paren_count += 1
            elif char == ")":
                paren_count -= 1
            if brace_count < 0 or paren_count < 0:
                return False
        return brace_count == 0 and paren_count == 0
    return True


def validate_think_tags(response: str) -> bool:
    """Validates explicit presence and structural ordering of <think>...</think> tags in response."""
    if not response or not isinstance(response, str):
        return False
    think_start = response.find("<think>")
    think_end = response.find("</think>")
    if think_start == -1 or think_end == -1:
        return False
    return think_start < think_end


def format_chatml_example(
    system_prompt: str, user_prompt: str, assistant_response: str
) -> dict[str, Any]:
    """Formats prompt components into standard ChatML schema."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.strip()},
            {"role": "assistant", "content": assistant_response.strip()},
        ]
    }


def generate_synthetic_samples(variant: str, count: int = 10) -> list[dict[str, Any]]:
    """Generates valid synthetic samples for testing and dry runs."""
    samples = []
    sys_prompt = SYSTEM_PROMPTS[variant]

    if variant == "c":
        for i in range(count):
            user_msg = (
                f"Refactor Python function #{i + 1} to calculate factorial efficiently."
            )
            code_diff = (
                "```diff\n"
                "--- a/math_utils.py\n"
                "+++ b/math_utils.py\n"
                "@@ -1,4 +1,4 @@\n"
                "-def factorial(n):\n"
                "-    return 1 if n <= 1 else n * factorial(n - 1)\n"
                "+import math\n"
                "+def factorial(n):\n"
                "+    return math.factorial(n)\n"
                "```"
            )
            resp = (
                "<think>\nAnalyzing factorial recursive call vs math.factorial binding in C-extension.\n"
                "Replacing recursion with standard math library for efficiency.\n</think>\n"
                f"Here is the refactored unified diff:\n\n{code_diff}"
            )
            samples.append(format_chatml_example(sys_prompt, user_msg, resp))

    elif variant == "r":
        for i in range(count):
            a, b = (i + 1) * 7, (i + 1) * 13
            user_msg = f"Solve for x: {a} + x = {b} * 2."
            rhs = b * 2
            ans = rhs - a
            resp = (
                f"<think>\n"
                f"Step 1: Calculate RHS: {b} * 2 = {rhs}.\n"
                f"Step 2: Set up equation: {a} + x = {rhs}.\n"
                f"Step 3: Subtract {a} from both sides: x = {rhs} - {a} = {ans}.\n"
                f"</think>\n"
                f"The value of x is {ans}."
            )
            samples.append(format_chatml_example(sys_prompt, user_msg, resp))

    elif variant == "g":
        for i in range(count):
            user_msg = (
                f"What is the capital of country #{i + 1}?"
                if i > 0
                else "What is the capital of France?"
            )
            city = "Paris" if i == 0 else f"CapitalCity_{i + 1}"
            resp = f"<think>\n</think>\nThe capital is {city}."
            samples.append(format_chatml_example(sys_prompt, user_msg, resp))

    return samples


def process_variant(
    variant: str, output_path: str, sample_size: int = 1000, synthetic: bool = False
) -> int:
    """Processes dataset for a specific variant and writes to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    raw_path = RAW_FILE_PATHS.get(variant)
    if raw_path:
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    records = []
    raw_records = []

    if synthetic:
        print(f"Generating synthetic records for variant '{variant}'...")
        records = generate_synthetic_samples(variant, count=sample_size)
    else:
        try:
            print(f"Attempting to load HF dataset for variant '{variant}'...")
            fetch_limit = int(sample_size * 1.5) if sample_size > 0 else 5000
            if variant == "c":
                ds = load_hf_dataset(
                    "iamtarun/python_code_instructions_18k_alpaca",
                    split=f"train[:{fetch_limit}]",
                )
                sys_prompt = SYSTEM_PROMPTS["c"]
                for row in ds:
                    if len(records) >= sample_size:
                        break
                    instruction = row.get("instruction", "")
                    inp = row.get("input", "")
                    output = row.get("output", "")

                    if validate_code_syntax(output, "python"):
                        raw_records.append(
                            {
                                "instruction": instruction,
                                "input": inp,
                                "output": output,
                            }
                        )
                        user_msg = (
                            f"{instruction}\n\nInput: {inp}".strip()
                            if inp
                            else instruction.strip()
                        )
                        assistant_resp = (
                            f"<think>\nValidating Python AST syntax and logic...\n</think>\n{output}".strip()
                        )
                        records.append(
                            format_chatml_example(sys_prompt, user_msg, assistant_resp)
                        )

            elif variant == "r":
                ds = load_hf_dataset(
                    "openai/gsm8k",
                    split=f"train[:{fetch_limit}]",
                )
                sys_prompt = SYSTEM_PROMPTS["r"]
                for row in ds:
                    if len(records) >= sample_size:
                        break
                    question = row.get("question", "")
                    answer = row.get("answer", "")

                    parts = answer.split("####")
                    reasoning_part = parts[0].strip()
                    final_ans = parts[1].strip() if len(parts) > 1 else ""
                    steps = [
                        line.strip()
                        for line in reasoning_part.split("\n")
                        if line.strip()
                    ]

                    # Filter: multi-step reasoning (len(steps) >= 2)
                    if len(steps) >= 2:
                        raw_records.append({"question": question, "answer": answer})
                        assistant_resp = (
                            "<think>\n"
                            + "\n".join(steps)
                            + f"\n</think>\nThe final answer is {final_ans}."
                        )
                        records.append(
                            format_chatml_example(sys_prompt, question, assistant_resp)
                        )

            elif variant == "g":
                ds = load_hf_dataset(
                    "HuggingFaceH4/ultrachat_200k",
                    split=f"train_sft[:{fetch_limit}]",
                )
                sys_prompt = SYSTEM_PROMPTS["g"]
                for row in ds:
                    if len(records) >= sample_size:
                        break
                    messages = row.get("messages", [])
                    if len(messages) >= 2:
                        user_msg = messages[0].get("content", "")
                        resp_msg = messages[1].get("content", "")

                        # Filter: exclude responses shorter than 50 chars
                        if len(resp_msg) >= 50:
                            raw_records.append(
                                {"prompt": user_msg, "response": resp_msg}
                            )
                            if not resp_msg.startswith("<think>"):
                                assistant_resp = f"<think>\n</think>\n{resp_msg}"
                            else:
                                assistant_resp = resp_msg
                            records.append(
                                format_chatml_example(
                                    sys_prompt, user_msg, assistant_resp
                                )
                            )

        except Exception as e:  # noqa: BLE001
            print(
                f"Warning: Failed to load HF dataset ({e}). Falling back to synthetic sample generation."
            )
            records = generate_synthetic_samples(variant, count=sample_size)

    # Save raw records if cached from HF dataset
    if raw_records and raw_path:
        with open(raw_path, "w", encoding="utf-8") as f:
            f.writelines(
                json.dumps(r, ensure_ascii=False) + "\n" for r in raw_records
            )
        print(f"Successfully wrote {len(raw_records)} raw records to {raw_path}")

    valid_records = []
    for record in records:
        msgs = record.get("messages", [])
        if (
            len(msgs) == 3
            and msgs[0]["role"] == "system"
            and msgs[1]["role"] == "user"
            and msgs[2]["role"] == "assistant"
        ):
            assistant_content = msgs[2].get("content", "")
            if variant == "r" and not validate_think_tags(assistant_content):
                continue
            valid_records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(
            json.dumps(rec, ensure_ascii=False) + "\n" for rec in valid_records
        )

    print(f"Successfully wrote {len(valid_records)} records to {output_path}")
    return len(valid_records)


def main():
    parser = argparse.ArgumentParser(
        description="MAURICE Dataset Preparation Pipeline"
    )
    parser.add_argument(
        "--variant",
        choices=["c", "r", "g", "all"],
        default="all",
        help="Target model variant",
    )
    parser.add_argument(
        "--sample-size", type=int, default=1000, help="Number of samples to process"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=False,
        help="Use synthetic sample generator",
    )
    parser.add_argument(
        "--real-hf",
        action="store_false",
        dest="synthetic",
        help="Use real HuggingFace datasets",
    )

    args = parser.parse_args()
    variants = ["c", "r", "g"] if args.variant == "all" else [args.variant]

    total_prepared = 0
    for v in variants:
        out_file = f"data/processed/train_{v}.jsonl"
        count = process_variant(
            v, out_file, sample_size=args.sample_size, synthetic=args.synthetic
        )
        total_prepared += count

    print(
        f"Pipeline complete. Total records prepared across variants: {total_prepared}"
    )


if __name__ == "__main__":
    main()
