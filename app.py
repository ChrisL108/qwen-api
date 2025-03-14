#!/usr/bin/env python
"""
Age Estimation API - Single File

https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

Requirements:
    pip install git+https://github.com/huggingface/transformers accelerate
    pip install qwen-vl-utils[decord]==0.0.8
    (M1 Mac) had to run: `pip install qwen-vl-utils`

Test on an M1 Mac using CPU. 
TODO: For GPU deployment, update `device_map` and `torch_dtype` accordingly.
"""

import os
import time
import asyncio
import base64
import re
import logging
import traceback
from io import BytesIO
from typing import Dict, Any

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

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

    async def estimate_age(self, image_data: bytes) -> Dict[str, Any]:
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
                        {"type": "text", "text": "What is the age of this person? Please reply with just a number."}
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

# ----------------- FASTAPI APPLICATION -----------------

app = FastAPI(
    title="Age Estimation API",
    description="API for estimating age using Qwen2.5-VL-3B-Instruct. Refer to https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct for docs and code examples.",
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
        "ok": True,
        "info": get_system_info(),
        "model_status": "loaded" if model else "loading" if model_loading else "not_loaded"
    }

@app.post("/estimate-age")
async def estimate_age(image: UploadFile = File(...)) -> Dict[str, Any]:
    """Estimate age from an uploaded image"""
    if not model:
        if model_loading:
            raise HTTPException(status_code=503, detail="Model is still loading, please try again later")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file extension
    valid_extensions = [".jpg", ".jpeg", ".png"]
    file_ext = os.path.splitext(image.filename)[1].lower()
    if file_ext not in valid_extensions:
        raise HTTPException(status_code=400, detail=f"Invalid file format. Supported formats: {', '.join(valid_extensions)}")
    
    # Read image data
    image_data = await image.read()
    if not image_data:
        raise HTTPException(status_code=400, detail="Empty file")
    
    # Process image with the model
    start_time = time.time()
    result = await model.estimate_age(image_data)
    processing_time = time.time() - start_time
    
    if not result.get("success", False):
        error_detail = result.get("error", "Unknown error")
        if "traceback" in result:
            logger.error(f"Error details: {result['traceback']}")
        raise HTTPException(status_code=500, detail=error_detail)
    
    return {
        "filename": image.filename,
        "age": result["age"],
        "processing_time_seconds": processing_time,
        "raw_response": result.get("raw_response", ""),
        "info": get_system_info()
    }

# ----------------- UTILITY FUNCTIONS -----------------

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
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
