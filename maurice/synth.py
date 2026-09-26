"""
MAURICE RLAIF Preference Synthesizer (maurice/synth.py)

Generates candidate responses using LLM inference (vLLM/HF/Mock),
evaluates candidates with LLM-as-a-Judge quality scoring,
and constructs preference datasets (chosen vs. rejected) saved to data/processed/prefs_{variant}.jsonl.
"""

import argparse
import json
import logging
import os
import re
from typing import Any

from maurice.prepare import (
    generate_synthetic_samples,
    validate_code_in_assistant_content,
    validate_think_tags,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = "You are an impartial and expert LLM judge evaluating response quality."

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator assessing the quality of an AI assistant's response.

[User Prompt]
{prompt}

[Response to Evaluate]
{response}

Evaluate the response on a scale of 1 to 10 based on correctness, clarity, adherence to instructions, and technical quality.
First, provide a brief reasoning analysis.
At the end of your response, output the final numeric score on a new line in the exact format: "Score: X" (where X is a number from 1 to 10, e.g. "Score: 8.5").
"""


def parse_score_from_judge_response(judge_text: str, default_score: float = 5.0) -> float:
    """Parses numeric score from judge response text matching 'Score: X' pattern."""
    if not judge_text or not isinstance(judge_text, str):
        return default_score

    match = re.search(r"Score:\s*([0-9]+(?:\.[0-9]+)?)", judge_text, re.IGNORECASE)
    if match:
        try:
            score = float(match.group(1))
            return max(1.0, min(10.0, score))
        except ValueError:
            pass
    return default_score


def score_response_heuristically(prompt: str, response: str, variant: str) -> float:
    """Calculates heuristic quality score (1.0 - 10.0) based on response structural validity."""
    if not response or not response.strip():
        return 1.0

    score = 6.0

    if variant == "r":
        if validate_think_tags(response):
            score += 3.0
            think_start = response.find("<think>")
            think_end = response.find("</think>")
            think_len = think_end - think_start
            if think_len > 30:
                score += 1.0
        else:
            score -= 4.0
    elif variant == "c":
        if "```" in response:
            score += 1.5
        if validate_code_in_assistant_content(response):
            score += 2.0
        else:
            score -= 3.0
        if "diff" in response or "def " in response or "class " in response:
            score += 0.5
    elif variant == "g":
        if "<think>" in response and "</think>" in response:
            score += 1.0
        if len(response.strip()) > 20:
            score += 2.0
        else:
            score -= 2.0

    return max(1.0, min(10.0, score))


def generate_candidate_responses_mock(prompt: str, variant: str, count: int = 2) -> list[str]:
    """Generates mock candidate responses with varying quality for synthetic/mock execution."""
    responses = []

    if variant == "c":
        resp_good = (
            "<think>\nAnalyzing code structure and applying pythonic optimization.\n"
            "Validating syntax using AST parser.\n</think>\n"
            "```python\ndef solution(x):\n    return x * 2\n```"
        )
        resp_flawed = "Here is code without block formatting:\ndef solution(x) return x*2"
        responses = [resp_good, resp_flawed]
    elif variant == "r":
        resp_good = (
            "<think>\nStep 1: Identify given conditions.\nStep 2: Calculate target value step-by-step.\n"
            "Step 3: Verify math accuracy.\n</think>\nThe solution is 42."
        )
        resp_flawed = "The answer is probably 42 without step-by-step reasoning."
        responses = [resp_good, resp_flawed]
    else:  # variant == "g"
        resp_good = "<think>\nConcise response required.\n</think>\nHere is the detailed, clear, and helpful answer."
        resp_flawed = "Short incomplete answer."
        responses = [resp_good, resp_flawed]

    while len(responses) < count:
        responses.append(f"Candidate response #{len(responses) + 1} for prompt: {prompt[:30]}...")

    return responses[:count]


def generate_responses(
    prompt: str,
    variant: str = "c",
    num_responses: int = 2,
    engine_type: str = "mock",
    model_obj: Any = None,
) -> list[str]:
    """Generates candidate responses using specified engine or mock fallback."""
    if engine_type == "mock" or model_obj is None:
        return generate_candidate_responses_mock(prompt, variant, count=num_responses)

    if engine_type == "vllm":
        try:
            from vllm.sampling_params import SamplingParams

            sampling_params = SamplingParams(
                n=num_responses,
                temperature=0.8,
                top_p=0.95,
                max_tokens=512,
            )
            outputs = model_obj.generate([prompt], sampling_params)
            if outputs and outputs[0].outputs:
                return [out.text for out in outputs[0].outputs]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"vLLM response generation failed ({e}). Falling back to mock generation.")
            return generate_candidate_responses_mock(prompt, variant, count=num_responses)

    elif engine_type == "hf":
        try:
            tokenizer, model = model_obj
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                num_return_sequences=num_responses,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
            )
            responses = []
            for out in outputs:
                resp_text = tokenizer.decode(out[inputs.input_ids.shape[-1] :], skip_special_tokens=True)
                responses.append(resp_text)
            return responses
        except Exception as e:  # noqa: BLE001
            logger.warning(f"HF response generation failed ({e}). Falling back to mock generation.")
            return generate_candidate_responses_mock(prompt, variant, count=num_responses)

    return generate_candidate_responses_mock(prompt, variant, count=num_responses)


def judge_response(
    prompt: str,
    response: str,
    variant: str = "c",
    engine_type: str = "mock",
    judge_model_obj: Any = None,
) -> float:
    """Uses LLM-as-a-Judge or heuristic fallback to evaluate response quality score."""
    if engine_type == "mock" or judge_model_obj is None:
        return score_response_heuristically(prompt, response, variant)

    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(prompt=prompt, response=response)

    if engine_type == "vllm":
        try:
            from vllm.sampling_params import SamplingParams

            params = SamplingParams(temperature=0.0, max_tokens=256)
            outputs = judge_model_obj.generate([judge_prompt], params)
            if outputs and outputs[0].outputs:
                return parse_score_from_judge_response(outputs[0].outputs[0].text)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"vLLM judge evaluation failed ({e}). Falling back to heuristic scoring.")

    elif engine_type == "hf":
        try:
            tokenizer, model = judge_model_obj
            inputs = tokenizer(judge_prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=False)
            judge_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True)
            return parse_score_from_judge_response(judge_text)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"HF judge evaluation failed ({e}). Falling back to heuristic scoring.")

    return score_response_heuristically(prompt, response, variant)


def load_sft_prompts(input_file: str, variant: str = "c") -> list[str]:
    """Loads SFT user prompts from JSONL file or generates synthetic prompts if missing."""
    prompts = []

    if os.path.exists(input_file):
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if "messages" in item:
                        user_msg = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
                        if user_msg:
                            prompts.append(user_msg)
                    elif "prompt" in item:
                        prompts.append(item["prompt"])
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to read SFT dataset {input_file} ({e}). Generating synthetic prompts.")

    if not prompts:
        logger.info(f"Generating synthetic prompts for variant '{variant}'...")
        synth_samples = generate_synthetic_samples(variant, count=10)
        for sample in synth_samples:
            user_msg = next((m["content"] for m in sample["messages"] if m["role"] == "user"), "")
            if user_msg:
                prompts.append(user_msg)

    return prompts


def synth_prefs_variant(
    variant: str = "c",
    input_file: str | None = None,
    output_file: str | None = None,
    num_responses: int = 2,
    dry_run: bool = False,
    engine_type: str = "mock",
) -> int:
    """Synthesizes preference dataset (prompt, chosen, rejected) for a model variant."""
    if input_file is None:
        input_file = f"data/processed/train_{variant}.jsonl"
    if output_file is None:
        output_file = f"data/processed/prefs_{variant}.jsonl"

    logger.info(f"Synthesizing preferences for variant '{variant}' (Engine: {engine_type}, Dry-run: {dry_run})...")
    prompts = load_sft_prompts(input_file, variant=variant)

    if dry_run:
        prompts = prompts[:5]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pref_records = []

    model_obj = None
    if not dry_run and engine_type != "mock":
        model_path = f"maurice-final-{variant}"
        if engine_type == "vllm":
            try:
                from vllm import LLM

                model_obj = LLM(model=model_path, trust_remote_code=True)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not load vLLM engine ({e}). Proceeding in mock mode.")
                engine_type = "mock"
        elif engine_type == "hf":
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                tok = AutoTokenizer.from_pretrained(model_path)
                mod = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                model_obj = (tok, mod)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not load HF model ({e}). Proceeding in mock mode.")
                engine_type = "mock"

    for prompt in prompts:
        candidates = generate_responses(
            prompt,
            variant=variant,
            num_responses=num_responses,
            engine_type=engine_type,
            model_obj=model_obj,
        )

        scored_candidates = []
        for cand in candidates:
            score = judge_response(
                prompt,
                cand,
                variant=variant,
                engine_type=engine_type,
                judge_model_obj=model_obj,
            )
            scored_candidates.append({"text": cand, "score": score})

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        chosen = scored_candidates[0]["text"]
        rejected = scored_candidates[-1]["text"]

        pref_records.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in pref_records)

    logger.info(f"Successfully generated {len(pref_records)} preference records in {output_file}")
    return len(pref_records)


def main(args_list: list[str] | None = None):
    parser = argparse.ArgumentParser(description="MAURICE RLAIF Preference Synthesizer")
    parser.add_argument(
        "--variant",
        choices=["c", "r", "g", "all"],
        default="c",
        help="Target model variant",
    )
    parser.add_argument("--input-file", type=str, default=None, help="Input SFT dataset path")
    parser.add_argument("--output-file", type=str, default=None, help="Output preference dataset path")
    parser.add_argument("--num-responses", type=int, default=2, help="Number of responses per prompt")
    parser.add_argument(
        "--engine",
        choices=["hf", "vllm", "mock"],
        default="mock",
        help="Inference engine for response generation & judge",
    )
    parser.add_argument("--dry-run", action="store_true", help="Perform quick dry run")

    args = parser.parse_args(args_list)

    variants = ["c", "r", "g"] if args.variant == "all" else [args.variant]
    total_generated = 0

    for v in variants:
        in_f = args.input_file if args.variant != "all" else None
        out_f = args.output_file if args.variant != "all" else None
        count = synth_prefs_variant(
            variant=v,
            input_file=in_f,
            output_file=out_f,
            num_responses=args.num_responses,
            dry_run=args.dry_run,
            engine_type=args.engine,
        )
        total_generated += count

    print(f"RLAIF synthesis complete. Total preference records generated: {total_generated}")


if __name__ == "__main__":
    main()
