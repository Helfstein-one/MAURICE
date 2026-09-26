#!/usr/bin/env python3
"""
Backwards-compatible wrapper delegating to maurice.merge.
"""

import sys

from maurice.merge import main, merge_weights

__all__ = ["main", "merge_weights"]

if __name__ == "__main__":
    sys.exit(main())
