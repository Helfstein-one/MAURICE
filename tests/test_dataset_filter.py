import importlib.util
import os
import sys
import unittest

# Import functions from scripts/01_prepare_datasets.py dynamically
spec = importlib.util.spec_from_file_location(
    "prepare_datasets",
    os.path.join(os.path.dirname(__file__), "../scripts/01_prepare_datasets.py"),
)
prepare_datasets = importlib.util.module_from_spec(spec)
sys.modules["prepare_datasets"] = prepare_datasets
spec.loader.exec_module(prepare_datasets)

from unittest.mock import patch

validate_code_syntax = prepare_datasets.validate_code_syntax
_brace_balance_check = prepare_datasets._brace_balance_check
validate_js_ts_syntax = prepare_datasets.validate_js_ts_syntax
validate_think_tags = prepare_datasets.validate_think_tags
format_chatml_example = prepare_datasets.format_chatml_example
generate_synthetic_samples = prepare_datasets.generate_synthetic_samples
SYSTEM_PROMPTS = prepare_datasets.SYSTEM_PROMPTS


class TestDatasetFilter(unittest.TestCase):
    def test_validate_js_valid_arrow_function(self):
        code = "const add = (a, b) => a + b;"
        self.assertTrue(validate_code_syntax(code, "javascript"))
        self.assertTrue(validate_code_syntax(code, "js"))

    def test_validate_js_unbalanced_braces(self):
        code = "function test() { console.log('hello');"
        self.assertFalse(validate_code_syntax(code, "javascript"))

    def test_validate_ts_interface(self):
        code = "interface User {\n  name: string;\n  age: number;\n}"
        self.assertTrue(validate_code_syntax(code, "typescript"))
        self.assertTrue(validate_code_syntax(code, "ts"))

    def test_brace_balance_check_with_brackets(self):
        valid_code = "const arr = [1, 2, (3 + 4)];"
        invalid_code = "const arr = [1, 2, (3 + 4);"
        self.assertTrue(_brace_balance_check(valid_code))
        self.assertFalse(_brace_balance_check(invalid_code))

    def test_validate_js_node_fallback(self):
        code = "const x = [1, 2, 3];"
        with patch("subprocess.run", side_effect=FileNotFoundError("node not found")):
            self.assertTrue(validate_code_syntax(code, "javascript"))

    def test_ast_validation_valid(self):
        valid_code = "def add(a: int, b: int) -> int:\n    return a + b"
        self.assertTrue(validate_code_syntax(valid_code, "python"))

    def test_ast_validation_syntax_error(self):
        invalid_code = "def broken_func(\n    return 42"
        self.assertFalse(validate_code_syntax(invalid_code, "python"))

    def test_c_syntax_validation(self):
        valid_c = 'int main() { printf("Hello"); return 0; }'
        invalid_c = 'int main() { printf("Hello");'
        self.assertTrue(validate_code_syntax(valid_c, "c"))
        self.assertFalse(validate_code_syntax(invalid_c, "c"))

    def test_think_tag_validation(self):
        valid_think = "<think>\nValid reasoning chain.\n</think>\nActual answer."
        missing_end = "<think>\nIncomplete reasoning chain."
        missing_start = "No think tag here.\n</think>"
        unordered = "</think>\n<think>Reversed tags"

        self.assertTrue(validate_think_tags(valid_think))
        self.assertFalse(validate_think_tags(missing_end))
        self.assertFalse(validate_think_tags(missing_start))
        self.assertFalse(validate_think_tags(unordered))

    def test_general_variant_minimal_think_tags(self):
        samples = generate_synthetic_samples("g", count=3)
        for sample in samples:
            assistant_content = sample["messages"][2]["content"]
            self.assertTrue(assistant_content.startswith("<think>\n</think>"))

    def test_code_variant_diff_formatting(self):
        samples = generate_synthetic_samples("c", count=3)
        for sample in samples:
            assistant_content = sample["messages"][2]["content"]
            self.assertIn("```diff", assistant_content)
            self.assertTrue(validate_think_tags(assistant_content))

    def test_chatml_schema(self):
        chatml = format_chatml_example(SYSTEM_PROMPTS["c"], "User query", "Assistant reply")
        self.assertIn("messages", chatml)
        self.assertEqual(len(chatml["messages"]), 3)
        self.assertEqual(chatml["messages"][0]["role"], "system")
        self.assertEqual(chatml["messages"][1]["role"], "user")
        self.assertEqual(chatml["messages"][2]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
