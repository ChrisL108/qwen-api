import time
import json
import re
import traceback
import asyncio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from app.schemas import ChatCompletionRequest
from app.config import DEFAULT_PROMPT

router = APIRouter()

def extract_image_and_prompt(messages) -> (str, str):
    prompt = DEFAULT_PROMPT
    image_url = None
    for message in reversed(messages):
        if message.role == "user":
            if isinstance(message.content, str):
                prompt = message.content
            else:
                for item in message.content:
                    if item.type == "text" and item.text:
                        prompt = item.text
                    elif item.type == "image_url" and item.image_url:
                        image_url = item.image_url.url
            break
    if not image_url:
        raise HTTPException(status_code=400, detail="No image found in the request")
    return image_url, prompt

async def stream_openai_response(result: dict) -> str:
    response_id = f"chatcmpl-{int(time.time())}"
    timestamp = int(time.time())
    age = result.get("age", 0)
    choice = {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
    response_data = {"id": response_id, "object": "chat.completion.chunk", "created": timestamp, "model": "age-estimation", "choices": [choice]}
    yield f"data: {json.dumps(response_data)}\n\n"
    choice = {"index": 0, "delta": {"content": str(age)}, "finish_reason": None}
    response_data = {"id": response_id, "object": "chat.completion.chunk", "created": timestamp, "model": "age-estimation", "choices": [choice]}
    yield f"data: {json.dumps(response_data)}\n\n"
    choice = {"index": 0, "delta": {}, "finish_reason": "stop"}
    response_data = {"id": response_id, "object": "chat.completion.chunk", "created": timestamp, "model": "age-estimation", "choices": [choice]}
    yield f"data: {json.dumps(response_data)}\n\n"
    yield "data: [DONE]\n\n"

@router.post("/v1/chat/completions")
async def create_chat_completion(request: Request, chat_request: ChatCompletionRequest):
    model = request.app.state.model
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        image_url, prompt = extract_image_and_prompt(chat_request.messages)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract image from request: {str(e)}")
    start_time = time.time()
    result = await model.estimate_age(image_url, prompt)
    processing_time = time.time() - start_time
    result["processing_time_seconds"] = processing_time
    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    if chat_request.stream:
        return StreamingResponse(stream_openai_response(result), media_type="text/event-stream")
    else:
        response_id = f"chatcmpl-{int(time.time())}"
        timestamp = int(time.time())
        age = result.get("age", 0)
        response = {
            "id": response_id,
            "object": "chat.completion",
            "created": timestamp,
            "model": chat_request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": str(age)},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": len(str(age)),
                "total_tokens": len(prompt) + len(str(age))
            }
        }
        return response

@router.get("/v1/models")
async def list_models_openai():
    return {
        "data": [
            {
                "id": "age-estimation",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "organization-owner"
            }
        ],
        "object": "list"
    }
