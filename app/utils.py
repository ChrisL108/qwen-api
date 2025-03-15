import base64
import json
import time
import os
import logging
from PIL import Image
from io import BytesIO
from fastapi import HTTPException
from typing import Dict, Any

logger = logging.getLogger(__name__)

async def download_image_from_url(url: str) -> bytes:
    import aiohttp
    logger.info(f"Downloading image from URL: {url}")
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=10) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download image: HTTP {response.status}")
            image_data = await response.read()
            logger.info(f"Downloaded {len(image_data)} bytes from URL")
            return image_data

def image_to_base64(self, image_data: bytes) -> str:
    try:
        img = Image.open(BytesIO(image_data))
        logger.info(f"Converting image to base64, image mode: {img.mode}")
        if img.mode != "RGB":
            img = img.convert("RGB")
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        logger.error("Error converting image to base64: " + str(e))
        raise

def extract_image_data_from_base64(image_data: str) -> bytes:
    if "base64," in image_data:
        image_data = image_data.split("base64,")[1]
    try:
        return base64.b64decode(image_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

def get_api_version() -> str:
    return "0.1.0"

def get_system_info() -> Dict[str, Any]:
    return {
        "api_version": get_api_version(),
        "timestamp": int(time.time()),
        "platform": os.uname().sysname,
        "machine": os.uname().machine
    }
