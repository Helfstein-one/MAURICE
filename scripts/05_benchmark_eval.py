#!/usr/bin/env python3
"""
Backwards-compatible wrapper delegating to maurice.eval.
"""

import sys

from maurice.eval import analyze_benchmark_results, detect_hardware_accel, get_peak_rss_mb, main, run_benchmark_variant

__all__ = [
    "analyze_benchmark_results",
    "detect_hardware_accel",
    "get_peak_rss_mb",
    "main",
    "run_benchmark_variant",
]

if __name__ == "__main__":
    sys.exit(main())
