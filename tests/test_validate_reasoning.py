import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

# Dynamically import scripts/validate_reasoning.py
script_path = Path(__file__).parent.parent / "scripts" / "validate_reasoning.py"
spec = importlib.util.spec_from_file_location("validate_reasoning", script_path)
validate_reasoning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate_reasoning)


class TestValidateReasoning(unittest.TestCase):
    def test_parse_and_validate_think_tags_valid(self):
        sample_response = (
            "<think>\n"
            "Distance = Speed * Time. Here speed = 60 mph, time = 2.5 hours. "
            "60 * 2.5 = 150 miles.\n"
            "</think>\n\n"
            "The total distance traveled by the train is 150 miles."
        )
        res = validate_reasoning.parse_and_validate_think_tags(sample_response, expected_keywords=["150"])
        self.assertTrue(res["has_think_tags"])
        self.assertTrue(res["has_reasoning_steps"])
        self.assertTrue(res["has_expected_keywords"])
        self.assertTrue(res["passed"])
        self.assertIn("150 miles", res["think_content"])

    def test_parse_and_validate_think_tags_missing_tags(self):
        sample_response = "The distance is 150 miles."
        res = validate_reasoning.parse_and_validate_think_tags(sample_response, expected_keywords=["150"])
        self.assertFalse(res["has_think_tags"])
        self.assertFalse(res["passed"])

    def test_parse_and_validate_think_tags_empty_think(self):
        sample_response = "<think>\n</think>\nThe distance is 150 miles."
        res = validate_reasoning.parse_and_validate_think_tags(sample_response, expected_keywords=["150"])
        self.assertTrue(res["has_think_tags"])
        self.assertFalse(res["has_reasoning_steps"])
        self.assertFalse(res["passed"])

    def test_load_modelfile_system_prompt(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write('SYSTEM """Custom System Prompt for mau-llm-1.0-r"""\n')
            temp_path = f.name

        try:
            prompt = validate_reasoning.load_modelfile_system_prompt(temp_path)
            self.assertEqual(prompt, "Custom System Prompt for mau-llm-1.0-r")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_run_reasoning_validation_dry_run_success(self):
        success, report = validate_reasoning.run_reasoning_validation(
            dry_run=True,
            simulate_failure=False,
            output_json="build/test_reasoning_results.json",
        )
        self.assertTrue(success)
        self.assertEqual(report["passed_prompts"], report["total_prompts"])
        self.assertEqual(report["pass_rate"], 1.0)
        self.assertTrue(os.path.exists("build/test_reasoning_results.json"))

    def test_run_reasoning_validation_dry_run_failure(self):
        success, report = validate_reasoning.run_reasoning_validation(
            dry_run=True,
            simulate_failure=True,
            output_json="build/test_reasoning_results.json",
        )
        self.assertFalse(success)
        self.assertEqual(report["passed_prompts"], 0)

    def test_cli_execution_dry_run(self):
        res = subprocess.run(
            ["python3", "scripts/validate_reasoning.py", "--dry-run"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, f"CLI dry-run failed: {res.stderr}")
        self.assertIn("Validation Summary", res.stdout)


if __name__ == "__main__":
    unittest.main()
