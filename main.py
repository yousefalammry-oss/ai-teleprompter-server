import os
import json
import logging
import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI
from dotenv import load_dotenv
import uvicorn

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-8b-instant" 

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set.")

app = FastAPI(title="Groq Mirror Professional")

broadcast_queue = asyncio.Queue(maxsize=10)

SYSTEM_CONFIG = {
    "base_prompt": "You are a Mermaid generator. Rules: Start with ```mermaid, then a new line, then graph TD/LR. Never combine mermaid and graph."
}

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# تم تصحيح الرابط (إزالة الأقواس الزائدة)
client = AsyncOpenAI(base_url="[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)", api_key=GROQ_API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("mirror.html", {"request": request})

@app.post("/api/update-config")
async def update_config(config: dict):
    # لا حاجة لـ request.json هنا، الـ FastAPI سيقوم بتحويل الـ JSON إلى dict تلقائياً
    SYSTEM_CONFIG.update(config)
    logger.info(f"Configuration updated: {SYSTEM_CONFIG}")
    return {"status": "success", "new_config": SYSTEM_CONFIG}

@app.get("/api/stream-mirror")
async def stream_mirror():
    async def generator():
        while True:
            content = await broadcast_queue.get()
            yield f"data: {json.dumps({'content': content})}\n\n"
    return StreamingResponse(generator(), media_type="text/event-stream")

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        messages = data.get("messages", [])[-3:]
        
        async def stream_generator() -> AsyncGenerator[str, None]:
            try:
                stream = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "system", "content": SYSTEM_CONFIG['base_prompt']}] + messages,
                    stream=True,
                    max_tokens=1000
                )
                async for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        corrected = content.replace("mermaidgraph", "mermaid\ngraph")
                        if not broadcast_queue.full():
                            await broadcast_queue.put(corrected)
                        yield f"data: {json.dumps({'content': corrected})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # استخدام المنفذ من المتغيرات البيئية أو 8000 افتراضياً
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
