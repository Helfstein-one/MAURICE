import ast
import unittest

def validate_python_code(code_str: str) -> bool:
    try:
        ast.parse(code_str)
        return True
    except SyntaxError:
        return False

class TestDatasetFilter(unittest.TestCase):
    def test_ast_validation_valid(self):
        valid_code = "def add(a: int, b: int) -> int:\n    return a + b"
        self.assertTrue(validate_python_code(valid_code))

    def test_ast_validation_syntax_error(self):
        invalid_code = "def broken_func(\n    return 42"
        self.assertFalse(validate_python_code(invalid_code))

    def test_think_tag_closure(self):
        sample_response = "<think>\nValid reasoning chain.\n</think>\nActual answer."
        self.assertTrue("<think>" in sample_response and "</think>" in sample_response)

if __name__ == "__main__":
    unittest.main()
