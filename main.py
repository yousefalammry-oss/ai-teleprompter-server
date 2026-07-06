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

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set.")

app = FastAPI(title="Groq Mirror Professional")

# إعدادات النظام
SYSTEM_CONFIG = {
    "base_prompt": "أنت خبير في إنشاء الرسوم البيانية. استخدم كود Mermaid حصراً داخل ```mermaid [الكود] ```. ممنوع استخدام رسومات ASCII نهائياً."
}

# عداد التوكنز الكلي
TOTAL_TOKENS_USED = 0

# استخدام Queue مع حجم محدود لمنع تراكم البيانات
broadcast_queue = asyncio.Queue(maxsize=100)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("mirror.html", {"request": request})

@app.post("/api/update-config")
async def update_config(config: dict):
    SYSTEM_CONFIG.update(config)
    return {"status": "success", "config": SYSTEM_CONFIG}

@app.get("/api/stream-mirror")
async def stream_mirror():
    async def mirror_generator():
        try:
            while True:
                content = await broadcast_queue.get()
                yield f"data: {json.dumps({'content': content})}\n\n"
        except asyncio.CancelledError:
            pass
    return StreamingResponse(mirror_generator(), media_type="text/event-stream")

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        messages = data.get("messages", [])
        
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        # دمج البرومت مع آخر 6 رسائل فقط لتوفير التوكنز
        enhanced_messages = [{"role": "system", "content": SYSTEM_CONFIG['base_prompt']}] + messages[-6:]

        async def stream_generator() -> AsyncGenerator[str, None]:
            try:
                stream = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=enhanced_messages,
                    stream=True,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                full_response = ""
                async for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        if not broadcast_queue.full():
                            await broadcast_queue.put(content)
                        yield f"data: {json.dumps({'content': content})}\n\n"
                
                # حساب التوكنز: (عدد الكلمات * 1.3) + توكنز البرومت
                tokens_response = int(len(full_response.split()) * 1.3)
                tokens_prompt = int(len(json.dumps(enhanced_messages).split()) * 1.3)
                total_req = tokens_response + tokens_prompt
                
                global TOTAL_TOKENS_USED
                TOTAL_TOKENS_USED += total_req
                
                logger.info(f"Tokens consumed: {total_req} | Total Daily: {TOTAL_TOKENS_USED}")
                
                # إضافة عداد التوكنز لنهاية الرد في المرآة
                token_msg = f"\n\n---\n*استهلاك التوكن لهذا الطلب: {total_req}*"
                await broadcast_queue.put(token_msg)
                
                await broadcast_queue.put("[DONE]")
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    except Exception as e:
        logger.error(f"Endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
