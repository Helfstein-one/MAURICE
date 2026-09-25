#!/usr/bin/env python3
"""
Backwards-compatible wrapper delegating to maurice.prepare.
"""

import sys

from maurice.prepare import (
    SYSTEM_PROMPTS,
    _brace_balance_check,
    format_chatml_example,
    generate_synthetic_samples,
    main,
    process_hf_dataset,
    process_variant,
    validate_code_in_assistant_content,
    validate_code_syntax,
    validate_js_ts_syntax,
    validate_think_tags,
)

__all__ = [
    "SYSTEM_PROMPTS",
    "_brace_balance_check",
    "format_chatml_example",
    "generate_synthetic_samples",
    "main",
    "process_hf_dataset",
    "process_variant",
    "validate_code_in_assistant_content",
    "validate_code_syntax",
    "validate_js_ts_syntax",
    "validate_think_tags",
]

if __name__ == "__main__":
    sys.exit(main())
