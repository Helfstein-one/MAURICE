"""
scripts/06_serve_model.py
Implements an OpenAI-compatible /v1/chat/completions endpoint using FastAPI with SSE streaming.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPTS = {
    "c": (
        "You are mau-llm-1.0-c, an expert code and refactoring engine. "
        "Provide clean, syntactically verified code, unified diff patches, "
        "and structural refactoring instructions."
    ),
    "r": (
        "You are mau-llm-1.0-r, a pure reasoning engine. "
        "Think carefully before answering by placing your step-by-step "
        "reasoning process inside <think>...</think> tags."
    ),
    "g": (
        "You are mau-llm-1.0-g, a versatile general-purpose assistant. "
        "Provide concise and accurate responses, using minimal thinking when appropriate."
    ),
}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "maurice"
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=512, gt=0)
    stream: bool | None = False


class ServerState:
    def __init__(self) -> None:
        self.host: str = os.getenv("MAURICE_HOST", "0.0.0.0")
        self.port: int = int(os.getenv("MAURICE_PORT", "8000"))
        self.variant: str = os.getenv("MAURICE_VARIANT", "c")
        self.model_path: str | None = os.getenv("MAURICE_MODEL_PATH")
        self.backend: str = os.getenv("MAURICE_BACKEND", "hf")
        self.model: Any = None
        self.tokenizer: Any = None
        self.device: str = "cuda" if (torch and torch.cuda.is_available()) else "cpu"


server_state = ServerState()


def get_default_model_path(variant: str) -> str:
    return f"checkpoints/merged_{variant}"


def format_chatml_prompt(messages: list[ChatMessage], variant: str) -> str:
    """Formats chat messages into ChatML prompt standard (<|im_start|>...<|im_end|>)."""
    prompt_parts: list[str] = []
    has_system = any(msg.role == "system" for msg in messages)

    if not has_system:
        sys_prompt = SYSTEM_PROMPTS.get(variant, SYSTEM_PROMPTS["c"])
        prompt_parts.append(f"<|im_start|>system\n{sys_prompt}<|im_end|>\n")

    for msg in messages:
        prompt_parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n")

    prompt_parts.append("<|im_start|>assistant\n")
    return "".join(prompt_parts)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Modern FastAPI asynccontextmanager lifespan protocol for model/tokenizer initialization."""
    model_path = server_state.model_path or get_default_model_path(server_state.variant)

    if server_state.backend == "hf":
        try:
            if torch is None:
                raise ImportError("torch is not installed in the current environment.")

            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading tokenizer and model from {model_path} (variant: {server_state.variant})...")
            server_state.tokenizer = AutoTokenizer.from_pretrained(model_path)
            server_state.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            if not torch.cuda.is_available():
                server_state.model.to(server_state.device)
            logger.info(f"Model and tokenizer loaded successfully on {server_state.device}.")
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Failed to load HF backend model/tokenizer from '{model_path}': {e}. Falling back to 'mock' backend."
            )
            server_state.backend = "mock"
            server_state.model = None
            server_state.tokenizer = None
    else:
        logger.info(f"Initialized server in 'mock' mode for variant '{server_state.variant}'.")

    yield

    logger.info("Shutting down MAURICE Inference Server.")


app = FastAPI(title="MAURICE Inference Server", lifespan=lifespan)


def generate_mock_response(prompt: str, variant: str) -> str:
    """Generates deterministic mock response formatted for variant reasoning boundaries."""
    if variant == "r":
        return (
            "<think>\n"
            "Step 1: Analyze user input and reasoning constraints.\n"
            "Step 2: Derive optimal step-by-step logic.\n"
            "</think>\n"
            "This is a mock reasoning response derived for mau-llm-1.0-r."
        )
    elif variant == "c":
        return (
            "<think>\n"
            "Analyzing code syntax and structural refactoring requirements.\n"
            "</think>\n"
            "```python\n"
            "# Refactored mock code\n"
            "def solution():\n"
            "    return True\n"
            "```"
        )
    else:
        return "<think>\n</think>\nThis is a mock general response derived for mau-llm-1.0-g."


def _generate_hf_response(prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
    inputs = server_state.tokenizer(prompt, return_tensors="pt").to(server_state.device)
    with torch.no_grad():
        outputs = server_state.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            top_p=top_p,
            pad_token_id=server_state.tokenizer.eos_token_id,
        )
    response: str = server_state.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True)
    return response


async def generate_stream_hf(request: ChatCompletionRequest, prompt: str) -> AsyncGenerator[str, None]:
    req_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    try:
        from threading import Thread

        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(server_state.tokenizer, skip_prompt=True, skip_special_tokens=True)
        inputs = server_state.tokenizer(prompt, return_tensors="pt").to(server_state.device)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=request.max_tokens or 512,
            temperature=request.temperature if (request.temperature and request.temperature > 0) else 1.0,
            do_sample=(request.temperature is not None and request.temperature > 0),
            top_p=request.top_p or 1.0,
            pad_token_id=server_state.tokenizer.eos_token_id,
        )
        thread = Thread(target=server_state.model.generate, kwargs=generation_kwargs)
        thread.start()

        while True:
            try:
                token = await asyncio.to_thread(next, streamer)
            except StopIteration:
                break

            chunk_obj = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk_obj)}\n\n"

    except Exception as e:  # noqa: BLE001
        logger.error(f"Streaming exception: {e}")

    final_obj = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_obj)}\n\n"
    yield "data: [DONE]\n\n"


async def generate_stream_mock(request: ChatCompletionRequest, prompt: str) -> AsyncGenerator[str, None]:
    req_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    full_text = generate_mock_response(prompt, server_state.variant)
    words = full_text.split(" ")

    for idx, word in enumerate(words):
        token_chunk = word if idx == len(words) - 1 else word + " "
        chunk_obj = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {"content": token_chunk}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk_obj)}\n\n"
        await asyncio.sleep(0.005)

    final_obj = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_obj)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "variant": server_state.variant,
        "backend": server_state.backend,
    }


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    model_id = f"mau-llm-1.0-{server_state.variant}"
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 1700000000,
                "owned_by": "maurice",
            }
        ],
    }


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
) -> JSONResponse | StreamingResponse:
    if not request.messages:
        raise HTTPException(status_code=422, detail="Field 'messages' must contain at least one message.")

    prompt = format_chatml_prompt(request.messages, server_state.variant)

    if request.stream:
        stream_gen = (
            generate_stream_hf(request, prompt)
            if server_state.backend == "hf" and server_state.model is not None
            else generate_stream_mock(request, prompt)
        )
        return StreamingResponse(stream_gen, media_type="text/event-stream")

    if server_state.backend == "hf" and server_state.model is not None:
        response_text = _generate_hf_response(
            prompt,
            request.max_tokens or 512,
            request.temperature or 0.7,
            request.top_p or 1.0,
        )
    else:
        response_text = generate_mock_response(prompt, server_state.variant)

    return JSONResponse(
        content={
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split()),
            },
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAURICE OpenAI-compatible FastAPI Inference Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument(
        "--variant",
        choices=["c", "r", "g"],
        default="c",
        help="Model variant (c: code, r: reasoning, g: general)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Custom checkpoint directory or GGUF path. Defaults to checkpoints/merged_{variant}",
    )
    parser.add_argument(
        "--backend",
        choices=["hf", "mock"],
        default="hf",
        help="Inference backend (hf: HuggingFace transformers, mock: simulated generation)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server_state.host = args.host
    server_state.port = args.port
    server_state.variant = args.variant
    server_state.model_path = args.model_path
    server_state.backend = args.backend

    uvicorn.run(app, host=server_state.host, port=server_state.port)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        parse_args()
    main()
