import time
import json
import re
import traceback
import asyncio
from fastapi import APIRouter, HTTPException, Request, Depends, FastAPI
from fastapi.responses import StreamingResponse
from app.schemas import ChatCompletionRequest
from app.endpoints.auth import get_api_key
from app.utils import format_message_for_qwen

router = APIRouter()

def extract_images_and_prompt(messages):
    prompt = None
    image_urls = []
    
    # Process the most recent user message
    for message in reversed(messages):
        if message.role == "user":
            if isinstance(message.content, str):
                prompt = message.content
            else:
                # Extract all images and the text prompt
                for item in message.content:
                    if item.type == "text" and item.text:
                        prompt = item.text
                    elif item.type == "image_url" and item.image_url:
                        image_urls.append(item.image_url.url)
            break  # Only process the most recent user message
    
    if not image_urls:
        raise HTTPException(status_code=400, detail="No images found in the request")
    
    # Return both the formatted message and the original prompt text
    return format_message_for_qwen(image_urls, prompt), prompt


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: Request, 
    chat_request: ChatCompletionRequest,
    api_key: str = Depends(get_api_key)
):
    model = request.app.state.model

    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        formatted_messages, prompt_text = extract_images_and_prompt(chat_request.messages)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract image from request: {str(e)}")
    
    start_time = time.time()
    result = await model.generate_response(formatted_messages)
    processing_time = time.time() - start_time
    result["processing_time_seconds"] = processing_time
    
    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

    else:
        response_id = f"chatcmpl-{int(time.time())}"
        raw_response = result.get("response", "")
        timestamp = int(time.time())
        response = {
            "id": response_id,
            "object": "chat.completion",
            "created": timestamp,
            "model": chat_request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": raw_response},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_text),
                "completion_tokens": len(raw_response),
                "total_tokens": len(prompt_text) + len(raw_response)
            }
        }
        return response

@router.get("/v1/models")
async def list_models_openai(api_key: str = Depends(get_api_key)):
    return {
        "data": [
            {
                "id": "Qwen/Qwen2.5-VL-72B-Instruct",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "organization-owner"
            }
        ],
        "object": "list"
    }