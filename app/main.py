import os
import time
import asyncio
import logging
import traceback
from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
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

@app.get("/", response_class=HTMLResponse)
async def read_readme():
    """
    Display README.md as HTML on the homepage
    """
    import markdown
    
    try:
        # Simple relative path - one directory up from the app directory
        app_dir = os.path.dirname(__file__)
        readme_path = os.path.join(os.path.dirname(app_dir), "README.md")
        
        # Check if README exists
        if not os.path.exists(readme_path):
            return HTMLResponse(content="<html><body><h1>README.md not found</h1></body></html>")
        
        # Read the README file
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown.markdown(
            readme_content,
            extensions=['fenced_code', 'tables', 'nl2br', 'codehilite']
        )
        
        # Wrap in HTML with some basic styling
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Qwen-2.5-VL Model Gateway</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 900px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                pre {{
                    background-color: #f6f8fa;
                    border-radius: 3px;
                    padding: 16px;
                    overflow: auto;
                }}
                code {{
                    background-color: #f6f8fa;
                    border-radius: 3px;
                    padding: 0.2em 0.4em;
                    font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                }}
                th {{
                    background-color: #f6f8fa;
                }}
                img {{
                    max-width: 100%;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        return HTMLResponse(content=full_html)
    except Exception as e:
        logger.error(f"Error rendering README: {str(e)}")
        return HTMLResponse(content=f"<html><body><h1>Error rendering README</h1><p>{str(e)}</p></body></html>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=True)