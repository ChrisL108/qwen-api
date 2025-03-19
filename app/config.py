import os
import torch
from dotenv import load_dotenv

load_dotenv()

# Single hardcoded token, no token system
API_TOKEN = os.getenv("API_TOKEN")

MODEL_NAME = os.getenv("MODEL_NAME")

USE_FAST = os.getenv("USE_FAST", "True").lower() == "true"

USE_FLASH_ATTN = os.getenv("USE_FLASH_ATTN", "True").lower() == "true"

dtype_str = os.getenv("TORCH_DTYPE", "")
if dtype_str == "float16":
    TORCH_DTYPE = torch.float16
elif dtype_str == "bfloat16":
    TORCH_DTYPE = torch.bfloat16
else:
    TORCH_DTYPE = "auto"

DEVICE_MAP = os.getenv("DEVICE_MAP", "auto")

CACHE_DIR = os.getenv("CACHE_DIR", None)