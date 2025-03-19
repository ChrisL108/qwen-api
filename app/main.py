import time
import asyncio
import logging
import traceback
from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.security.api_key import APIKeyHeader
from app.endpoints import openai, ollama
from app.utils import get_system_info
from app.models import AgeEstimationModel
from app.config import API_TOKEN

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(
    title="Qwen-2.5-VL Model Gateway (OpenAI and Ollama Compatible)",
    description="Gateway API for Qwen-2.5-VL model",
    version="0.1.0"
)

# Enable CORS as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(openai.router)
app.include_router(ollama.router)

# Load the model on startup
@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Initializing model...")
        app.state.model = await asyncio.to_thread(AgeEstimationModel)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())

@app.get("/health")
async def health_check():
    model = app.state.model if hasattr(app.state, "model") else None
    return {
        "status": "ok" if model else "loading",
        "info": get_system_info(),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=True)
