"""
scripts/06_serve_model.py
Implements an OpenAI-compatible /v1/chat/completions endpoint using FastAPI.
"""

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MAURICE Inference Server")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "maurice"
    messages: list[ChatMessage]
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    max_tokens: int | None = 512
    stream: bool | None = False


# Global state for model and tokenizer
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"


@app.on_event("startup")
def load_model() -> None:
    global model, tokenizer
    model_name_or_path = "maurice-final"  # Default model path
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        ).to(device)
        logger.info(f"Model {model_name_or_path} loaded successfully.")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not load model {model_name_or_path}: {e}")
        logger.warning("Starting without a loaded model (for testing purposes).")


def _generate_response(
    prompt: str, max_tokens: int, temperature: float, top_p: float
) -> str:
    if model is None or tokenizer is None:
        return "This is a mock response because the model is not loaded."

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True
    )
    return response


async def generate_stream(
    request: ChatCompletionRequest, prompt: str
) -> AsyncGenerator[str, None]:
    req_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    full_response = _generate_response(
        prompt,
        request.max_tokens or 512,
        request.temperature or 0.7,
        request.top_p or 1.0,
    )

    chunk_size = 10
    for i in range(0, len(full_response), chunk_size):
        chunk = full_response[i : i + chunk_size]

        response_obj = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [
                {"index": 0, "delta": {"content": chunk}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(response_obj)}\n\n"

    final_obj = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_obj)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
) -> JSONResponse | StreamingResponse:
    prompt = ""
    for msg in request.messages:
        prompt += f"{msg.role}: {msg.content}\n"
    prompt += "assistant: "

    if request.stream:
        return StreamingResponse(
            generate_stream(request, prompt), media_type="text/event-stream"
        )

    response_text = _generate_response(
        prompt,
        request.max_tokens or 512,
        request.temperature or 0.7,
        request.top_p or 1.0,
    )

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
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
