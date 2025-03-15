import base64
import re
import logging
import time
import traceback
from io import BytesIO
from typing import Dict, Any

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from app.config import MODEL_NAME, MIN_PIXELS, MAX_PIXELS, USE_FAST, TORCH_DTYPE, DEVICE_MAP, DEFAULT_PROMPT

logger = logging.getLogger(__name__)

class AgeEstimationModel:
    def __init__(self, model_name=MODEL_NAME):
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
            use_fast=USE_FAST
        )
        logger.info(f"Processor loaded from {model_name}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=TORCH_DTYPE,
            device_map=DEVICE_MAP,
        )

        logger.info(f"Model loaded successfully, using device: {self.model.device}")

    async def estimate_age(self, image_data: str, prompt: str = DEFAULT_PROMPT) -> Dict[str, Any]:
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_data},
                    {"type": "text", "text": prompt}
                ]
            }]
            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False, 
                add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=30)

            output_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Extract age from response
            numbers = re.findall(r'\d+', output_text)
            if numbers:
                age = int(numbers[0])
                if 0 < age < 120:
                    return {"age": age, "success": True, "raw_response": output_text}
            return {"age": 0, "success": False, "raw_response": output_text, "error": "Could not extract age"}
        except Exception as e:
            logger.error("Error during age estimation: " + str(e))
            logger.error(traceback.format_exc())
            return {"age": None, "error": str(e), "traceback": traceback.format_exc(), "success": False}

