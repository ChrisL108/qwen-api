#!/usr/bin/env python
"""
Age Estimation API - Single File - Compatible with both OpenAI and Ollama APIs

https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

Requirements:
    pip install git+https://github.com/huggingface/transformers accelerate
    pip install qwen-vl-utils[decord]==0.0.8
    pip install aiohttp
    (M1 Mac) had to run: `pip install qwen-vl-utils`

Test on an M1 Mac using CPU. 
TODO: For GPU deployment, update `device_map` and `torch_dtype` accordingly.
"""

import os
import time
import asyncio
import aiohttp
import base64
import re
import logging
import json
import traceback
from io import BytesIO
from typing import Dict, Any, List, Optional, Union, Literal, Tuple

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# Import model classes and utilities per the Qwen2.5-VL docs
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AgeEstimationModel:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct"):
        """
        Initialize the model using the Qwen2.5-VL class and AutoProcessor.
        This follows the examples from the Qwen2.5-VL-3B-Instruct docs.
        """
        # For testing on M1 Mac; for GPU deployment, update device_map and torch_dtype accordingly.
        self.device = "cpu"
        logger.info(f"Using device: {self.device}")

        # Load the processor with optional image resolution parameters 
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        logger.info(f"Processor loaded from {model_name}")

        # Load the model using the dedicated class, per the docs
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # CPU testing; for GPU consider "auto" or torch.bfloat16 with flash_attention_2.
            device_map="cpu",
        )
        logger.info("Model loaded successfully")

    def image_to_base64(self, image_data: bytes) -> str:
        """
        Convert raw image bytes to a base64-encoded string suitable for Qwen2.5-VL.
        This method mirrors the input format options shown in the docs.
        """
        try:
            img = Image.open(BytesIO(image_data))
            if img.mode != "RGB":
                img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            logger.error("Error converting image to base64: " + str(e))
            raise

    async def estimate_age(self, image_data: bytes, prompt: str = "What is the age of this person? Please reply with just a number.") -> Dict[str, Any]:
        """
        Estimate age by constructing a chat-style input following the Qwen2.5-VL guidelines.
        Uses a fixed prompt asking for the age of the person.
        """
        try:
            # Convert the image data to a base64-encoded string
            img_b64 = self.image_to_base64(image_data)
            
            # Build a messages structure as expected by the Qwen2.5-VL model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_b64},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Prepare the text prompt using the processor's chat template 
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process vision inputs using qwen_vl_utils (this supports various input types as per the docs)
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Prepare the inputs for the model
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # Generate response using max_new_tokens
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=30)
            
            output_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            logger.info(f"Response: {output_text}")
            
            # Extract number from generated response
            numbers = re.findall(r'\d+', output_text)
            if numbers:
                age = int(numbers[0])
                if 0 < age < 120:
                    return {"age": age, "success": True, "raw_response": output_text}
            
            # fallback failure
            return {"age": 0, "success": False, "raw_response": output_text, "error": "Could not extract age"}
        
        except Exception as e:
            logger.error("Error during age estimation: " + str(e))
            logger.error(traceback.format_exc())
            return {"age": None, "error": str(e), "traceback": traceback.format_exc(), "success": False}


# ----------------- PYDANTIC MODELS -----------------

# OpenAI API Models
class ImageURL(BaseModel):
    url: str

class ContentItem(BaseModel):
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[ContentItem]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 300
    temperature: Optional[float] = 1.0
    stream: Optional[bool] = False

class Delta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class Choice(BaseModel):
    index: int
    message: Optional[Message] = None
    delta: Optional[Delta] = None
    finish_reason: Optional[str] = None

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None

# Ollama API Models
class GenerateRequest(BaseModel):
    model: str = Field("age-estimation", description="Model name to use")
    prompt: str = Field(..., description="Prompt to generate a response for")
    images: Optional[List[str]] = Field(None, description="Base64 encoded images or URLs")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    options: Optional[Dict[str, Any]] = Field(None, description="Additional model parameters")


# ----------------- FASTAPI APPLICATION -----------------

app = FastAPI(
    title="Age Estimation API (OpenAI and Ollama Compatible)",
    description="API for estimating age using Qwen2.5-VL-3B-Instruct with both OpenAI and Ollama API compatibility.",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model on startup in the background to avoid blocking
model = None
model_loading = False

def load_model_task():
    global model, model_loading
    try:
        model_loading = True
        logger.info("Starting model loading...")
        start_time = time.time()
        _model = AgeEstimationModel()
        elapsed = time.time() - start_time
        logger.info(f"Model loaded successfully in {elapsed:.2f} seconds")
        model = _model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
    finally:
        model_loading = False

@app.on_event("startup")
async def startup_event():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(load_model_task)
    asyncio.create_task(asyncio.to_thread(load_model_task))

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "ok" if model else "loading" if model_loading else "unavailable",
        "info": get_system_info(),
    }

# ----------------- UTILITY FUNCTIONS -----------------

async def download_image_from_url(url: str) -> bytes:
    """Download image from URL and return as bytes"""
    try:
        logger.info(f"Downloading image from URL: {url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    raise HTTPException(status_code=400, detail=f"Failed to download image: HTTP {response.status}")
                image_data = await response.read()
                logger.info(f"Downloaded {len(image_data)} bytes from URL")
                return image_data
    except aiohttp.ClientError as e:
        logger.error(f"Error downloading image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")

def _extract_image_data_from_base64(image_data: str) -> bytes:
    """Extract binary image data from base64 encoded string"""
    # Handle various base64 formats
    if "base64," in image_data:
        image_data = image_data.split("base64,")[1]
    
    try:
        return base64.b64decode(image_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

def _extract_image_data_from_openai_messages(messages: List[Message]) -> Tuple[Optional[bytes], Optional[str], str]:
    """
    Extract information from OpenAI messages format
    Returns: (image_data, image_url, prompt)
    """
    # Start with default prompt
    prompt = "What is the age of this person? Please reply with just a number."
    image_data = None
    image_url = None
    
    # Process the last user message
    for message in reversed(messages):
        if message.role == "user":
            if isinstance(message.content, str):
                prompt = message.content
                continue
                
            for item in message.content:
                if item.type == "text" and item.text:
                    prompt = item.text
                elif item.type == "image_url" and item.image_url:
                    url = item.image_url.url
                    # Handle base64 encoded images
                    if url.startswith("data:"):
                        image_data = _extract_image_data_from_base64(url)
                    # For external URLs, store for later download
                    else:
                        image_url = url
            break
    
    if not image_data and not image_url:
        raise HTTPException(status_code=400, detail="No image found in the request")
        
    return image_data, image_url, prompt

async def _stream_openai_response(result: Dict[str, Any]) -> str:
    """Stream response in the OpenAI format"""
    response_id = f"chatcmpl-{int(time.time())}"
    timestamp = int(time.time())
    age = result.get("age", 0)
    
    # First chunk with role
    choice = {
        "index": 0,
        "delta": {"role": "assistant"},
        "finish_reason": None
    }
    response_data = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": timestamp,
        "model": "age-estimation",
        "choices": [choice]
    }
    yield f"data: {json.dumps(response_data)}\n\n"
    
    # Content chunk
    choice = {
        "index": 0,
        "delta": {"content": str(age)},
        "finish_reason": None
    }
    response_data = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": timestamp,
        "model": "age-estimation",
        "choices": [choice]
    }
    yield f"data: {json.dumps(response_data)}\n\n"
    
    # Final chunk
    choice = {
        "index": 0,
        "delta": {},
        "finish_reason": "stop"
    }
    response_data = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": timestamp,
        "model": "age-estimation",
        "choices": [choice]
    }
    yield f"data: {json.dumps(response_data)}\n\n"
    yield "data: [DONE]\n\n"

async def _stream_ollama_response(result: Dict[str, Any]) -> str:
    """Stream response in the Ollama format"""
    age = result.get("age", 0)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    
    response_data = {
        "model": "age-estimation",
        "created_at": timestamp,
        "response": f"{age}",
        "done": False
    }
    yield json.dumps(response_data) + "\n"
    
    # Final response
    response_data["done"] = True
    response_data["total_duration"] = int(result.get("processing_time_seconds", 0) * 1e9)  # nanoseconds
    yield json.dumps(response_data) + "\n"

# ----------------- OPENAI COMPATIBLE ENDPOINTS -----------------

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Generate a chat completion (OpenAI compatible)"""
    if not model:
        if model_loading:
            raise HTTPException(status_code=503, detail="Model is still loading, please try again later")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Extract image data and prompt from the request
    try:
        image_data, image_url, prompt = _extract_image_data_from_openai_messages(request.messages)
        
        # If we have a URL but no image data, download the image
        if not image_data and image_url:
            logger.info(f"Downloading image from URL in OpenAI format: {image_url}")
            image_data = await download_image_from_url(image_url)
            
        if not image_data:
            raise HTTPException(status_code=400, detail="Failed to process image from request")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Failed to extract image from request: {str(e)}")
    
    # Process image with the model
    start_time = time.time()
    result = await model.estimate_age(image_data, prompt)
    processing_time = time.time() - start_time
    result["processing_time_seconds"] = processing_time
    
    if not result.get("success", False):
        error_detail = result.get("error", "Unknown error")
        if "traceback" in result:
            logger.error(f"Error details: {result['traceback']}")
        raise HTTPException(status_code=500, detail=error_detail)
    
    # Format response according to OpenAI API
    response_id = f"chatcmpl-{int(time.time())}"
    timestamp = int(time.time())
    age = result.get("age", 0)
    
    if request.stream:
        return StreamingResponse(
            _stream_openai_response(result),
            media_type="text/event-stream"
        )
    else:
        response = {
            "id": response_id,
            "object": "chat.completion",
            "created": timestamp,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": str(age)
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt), # Approximate
                "completion_tokens": len(str(age)),
                "total_tokens": len(prompt) + len(str(age))
            }
        }
        return response

@app.get("/v1/models")
async def list_models_openai() -> Dict[str, Any]:
    """List available models (OpenAI compatible)"""
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

# ----------------- OLLAMA COMPATIBLE ENDPOINTS -----------------

@app.post("/api/generate")
async def generate_ollama(request: Request):
    """Generate a completion (Ollama compatible)"""
    if not model:
        if model_loading:
            raise HTTPException(status_code=503, detail="Model is still loading, please try again later")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    request_data = await request.json()
    
    # Extract data from Ollama format
    model_name = request_data.get("model", "age-estimation")
    prompt = request_data.get("prompt", "What is the age of this person?")
    images = request_data.get("images", [])
    stream = request_data.get("stream", False)
    
    if not images:
        raise HTTPException(status_code=400, detail="Images are required for age estimation")
    
    # Extract image data
    try:
        # Check if the image is a URL or base64
        image_source = images[0]
        
        # If it's a base64 encoded string
        if image_source.startswith("data:"):
            image_data = _extract_image_data_from_base64(image_source)
        # If it's a URL
        else:
            logger.info(f"Downloading image from URL in Ollama format: {image_source}")
            image_data = await download_image_from_url(image_source)
            
    except Exception as e:
        logger.error(f"Error processing Ollama image: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")
    
    # Process image with the model
    start_time = time.time()
    result = await model.estimate_age(image_data, prompt)
    processing_time = time.time() - start_time
    result["processing_time_seconds"] = processing_time
    
    if not result.get("success", False):
        error_detail = result.get("error", "Unknown error")
        if "traceback" in result:
            logger.error(f"Error details: {result['traceback']}")
        raise HTTPException(status_code=500, detail=error_detail)
    
    # Format response according to Ollama API
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    age = result.get("age", 0)
    
    if stream:
        return StreamingResponse(
            _stream_ollama_response(result),
            media_type="application/json"
        )
    else:
        response = {
            "model": model_name,
            "created_at": timestamp,
            "response": str(age),
            "done": True,
            "total_duration": int(processing_time * 1e9),  # nanoseconds
        }
        return response

@app.get("/api/health")
async def health_check_ollama() -> Dict[str, Any]:
    """Health check endpoint (Ollama compatible)"""
    return {
        "status": "ok" if model else "loading" if model_loading else "unavailable",
        "info": get_system_info(),
    }

@app.get("/api/tags")
async def list_models_ollama() -> Dict[str, Any]:
    """List available models (Ollama compatible)"""
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

def get_api_version() -> str:
    """Return the API version"""
    return "0.1.0"

def get_system_info() -> Dict[str, Any]:
    """Return system information"""
    return {
        "api_version": get_api_version(),
        "timestamp": int(time.time()),
        "platform": os.uname().sysname,
        "machine": os.uname().machine
    }

# Run the application if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)