import time
import json
import traceback
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from app.utils import get_system_info
from app.config import DEFAULT_PROMPT

router = APIRouter()

async def stream_ollama_response(result: dict) -> str:
    age = result.get("age", 0)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    response_data = {
        "model": "age-estimation",
        "created_at": timestamp,
        "response": f"{age}",
        "done": False
    }
    yield json.dumps(response_data) + "\n"
    response_data["done"] = True
    response_data["total_duration"] = int(result.get("processing_time_seconds", 0) * 1e9)
    yield json.dumps(response_data) + "\n"

@router.post("/api/generate")
async def generate_ollama(request: Request):
    model = request.app.state.model
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_data = await request.json()
    model_name = request_data.get("model", "age-estimation")
    prompt = request_data.get("prompt", DEFAULT_PROMPT)
    images = request_data.get("images", [])
    stream = request_data.get("stream", False)
    
    if not images:
        raise HTTPException(status_code=400, detail="Images are required for age estimation")

    start_time = time.time()
    result = await model.estimate_age(images[0], prompt)
    processing_time = time.time() - start_time
    result["processing_time_seconds"] = processing_time
    
    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    if stream:
        return StreamingResponse(stream_ollama_response(result), media_type="application/json")
    else:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        age = result.get("age", 0)
        response = {
            "model": model_name,
            "created_at": timestamp,
            "response": str(age),
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
async def list_models_ollama():
    return {
        "models": [
            {
                "name": "age-estimation",
                "modified_at": int(time.time()),
                "size": 0,
                "digest": "age-estimation",
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
