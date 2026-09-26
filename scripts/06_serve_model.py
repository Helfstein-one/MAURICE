#!/usr/bin/env python3
"""
Backwards-compatible wrapper delegating to maurice.serve.
"""

import sys

from maurice.serve import (
    ChatCompletionRequest,
    ChatMessage,
    ServerState,
    app,
    format_chatml_prompt,
    generate_stream,
    generate_vllm_stream,
    load_model,
    main,
    parse_args,
    server_state,
)

__all__ = [
    "ChatCompletionRequest",
    "ChatMessage",
    "ServerState",
    "app",
    "format_chatml_prompt",
    "generate_stream",
    "generate_vllm_stream",
    "load_model",
    "main",
    "parse_args",
    "server_state",
]

if __name__ == "__main__":
    sys.exit(main())
