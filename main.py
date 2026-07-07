import os
import json
import logging
from typing import AsyncGenerator
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#MODEL_NAME = "openai/gpt-oss-120b"
MODEL_NAME = "llama-3.1-70b-versatile"


if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in environment variables.")

# Initialize FastAPI
app = FastAPI(title="Groq Mirror Professional")

# Setup Static & Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize Groq Client (OpenAI Compatible)
client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main UI."""
    return templates.TemplateResponse("mirror.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """
    Handle streaming chat requests.
    Expects JSON: { "messages": [...], "model": "..." }
    """
    try:
        data = await request.json()
        messages = data.get("messages", [])
        
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        async def stream_generator() -> AsyncGenerator[str, None]:
            try:
                stream = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    stream=True,
                    temperature=0.7,
                    max_tokens=8192
                )
                
                async for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        # Send as Server-Sent Events (SSE)
                        yield f"data: {json.dumps({'content': content})}\n\n"
                
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Encoding": "none",
            }
        )

    except Exception as e:
        logger.error(f"Endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
