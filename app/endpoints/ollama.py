import time
import json
import traceback
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from app.endpoints.auth import get_api_key
from app.utils import format_message_for_qwen, get_system_info

router = APIRouter()

async def stream_ollama_response(result: dict) -> str:
    raw_response = result.get("response", "")
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    response_data = {
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "created_at": timestamp,
        "response": raw_response,
        "done": False
    }
    yield json.dumps(response_data) + "\n"
    response_data["done"] = True
    response_data["total_duration"] = int(result.get("processing_time_seconds", 0) * 1e9)
    yield json.dumps(response_data) + "\n"


@router.post("/api/generate")
async def generate_ollama(request: Request, api_key: str = Depends(get_api_key)):
    model = request.app.state.model
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_data = await request.json()
    model_name = request_data.get("model", "Qwen/Qwen2.5-VL-3B-Instruct")
    prompt = request_data.get("prompt", "")
    images = request_data.get("images", [])
    stream = request_data.get("stream", False)
    
    if not images:
        raise HTTPException(status_code=400, detail="Images are required for visual analysis")

    formatted_messages = format_message_for_qwen(images, prompt)
    
    start_time = time.time()
    raw_response = await model.generate_response(formatted_messages)
    processing_time = time.time() - start_time
    raw_response["processing_time_seconds"] = processing_time
    
    if not raw_response.get("success", False):
        raise HTTPException(status_code=500, detail=raw_response.get("error", "Unknown error"))
    if stream:
        return StreamingResponse(stream_ollama_response(raw_response), media_type="application/json")
    else:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        response = raw_response.get("response", "")
        response = {
            "model": model_name,
            "created_at": timestamp,
            "response": response,
            "done": True,
            "total_duration": int(processing_time * 1e9)
        }
        return response

@router.get("/api/health")
async def health_check_ollama(request: Request):
    model = request.app.state.model
    return {
        "status": "ok" if model else "loading",
        "info": get_system_info(),
    }

@router.get("/api/tags")
async def list_models_ollama(api_key: str = Depends(get_api_key)):
    return {
        "models": [
            {
                "name": "Qwen/Qwen2.5-VL-3B-Instruct",
                "modified_at": int(time.time()),
                "size": 0,
                "digest": "Qwen/Qwen2.5-VL-3B-Instruct",
                "details": {
                    "parent_model": "Qwen/Qwen2.5-VL-3B-Instruct",
                    "format": "vision",
                    "family": "qwen",
                    "parameter_size": "3B",
                    "quantization_level": "none"
                }
            }
        ]
    }