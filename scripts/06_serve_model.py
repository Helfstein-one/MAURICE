#!/usr/bin/env python3
"""
Backwards-compatible wrapper delegating to maurice.serve.
"""

import sys

from maurice.serve import ChatCompletionRequest, ChatMessage, app, generate_stream, load_model, main

__all__ = ["ChatCompletionRequest", "ChatMessage", "app", "generate_stream", "load_model", "main"]

if __name__ == "__main__":
    sys.exit(main())
