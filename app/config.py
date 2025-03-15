import torch

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"

DEFAULT_PROMPT = "What is the estimated age of this person? Please reply with an exact number of your best guess."

# lower resolution
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

# higher resolution
# MIN_PIXELS = 224 * 224
# MAX_PIXELS = 2048 * 2048

USE_FAST = True

# torch.bfloat16 is preferred if running on TPUs or specific NVIDIA A100/H100 GPUs
# TORCH_DTYPE = torch.float16
TORCH_DTYPE = torch.float32

DEVICE_MAP = "cpu"
# DEVICE_MAP = "cuda"
# DEVICE_MAP = "mps" # Metal Performance Shaders - Apple Silicon (has issues with qwen2.5-vl afaict)