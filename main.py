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

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-8b-instant" 

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set.")

app = FastAPI(title="Groq Mirror Professional")

# الوسيط لنقل البيانات بين الشات والمرآة
broadcast_queue = asyncio.Queue(maxsize=10)

SYSTEM_CONFIG = {
    "base_prompt": "You are a Mermaid generator. Rules: Start with ```mermaid, then a new line, then graph TD/LR. Never combine mermaid and graph."
}

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

client = AsyncOpenAI(base_url="[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)", api_key=GROQ_API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("mirror.html", {"request": request})

# المسار المفقود الذي كان يسبب 404
@app.get("/api/stream-mirror")
async def stream_mirror():
    async def generator():
        while True:
            # انتظار بيانات جديدة من الـ Queue
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
                        # إرسال البيانات للـ Queue ليتم بثها للمرآة
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
