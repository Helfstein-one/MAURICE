#!/usr/bin/env python3
"""
Backwards-compatible wrapper delegating to maurice.train.
"""

import sys

from maurice.train import load_variant_config, main, run_training

__all__ = ["load_variant_config", "main", "run_training"]

if __name__ == "__main__":
    sys.exit(main())
