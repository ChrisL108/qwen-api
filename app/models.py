import base64
import re
import logging
import time
import traceback
from io import BytesIO
from typing import Dict, Any, List

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from app.config import MODEL_NAME, USE_FAST, TORCH_DTYPE, DEVICE_MAP, CACHE_DIR, USE_FLASH_ATTN
from app.utils import extract_assistant_response
logger = logging.getLogger(__name__)

class AgeEstimationModel:
    def __init__(self, model_name=MODEL_NAME):
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            use_fast=USE_FAST
        )
        logger.info(f"Processor loaded from {model_name}")
        
        model_params = {
            "torch_dtype": TORCH_DTYPE,
            "device_map": DEVICE_MAP,
            "cache_dir": CACHE_DIR,
            "trust_remote_code": True,
        }
        
        # Add flash attention if enabled
        if USE_FLASH_ATTN:
            model_params["attn_implementation"] = "flash_attention_2"

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            **model_params,
        )

        logger.info(f"Model loaded successfully, using device: {self.model.device}")

    async def generate_response(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
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
                generated_ids = self.model.generate(**inputs, max_new_tokens=512)

            output_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            logger.info(f"Raw output text: {output_text}")
            
            # Extract just the assistant's response
            assistant_response = extract_assistant_response(output_text)
            logger.info(f"Extracted assistant response: {assistant_response}")

            return {"success": True, "response": assistant_response}
        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}")
            return {"success": False, "error": str(e)}
