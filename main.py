import os
import json
import logging
import asyncio
from typing import AsyncGenerator, Dict, Any, List
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI, APIError
from dotenv import load_dotenv
import uvicorn

# -----------------------------------------------------------------------------
# LOGGING CONFIGURATION
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("openai_json_mirror")

load_dotenv()

# اختيار الموديل والمفتاح بناءً على الإعدادات المتوفرة
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
BASE_URL = "https://api.groq.com/openai/v1" if os.getenv("GROQ_API_KEY") else "https://api.openai.com/v1"
MODEL_NAME = "llama-3.3-8b-instant" if os.getenv("GROQ_API_KEY") else "gpt-4o-mini"

if not API_KEY:
    raise ValueError("لا يوجد API KEY معرّف في إعدادات البيئة!")

# -----------------------------------------------------------------------------
# INITIALIZATION
# -----------------------------------------------------------------------------
app = FastAPI(title="Pro JSON Mirror")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
SYSTEM_CONFIG = {"base_prompt": "You are a strict Mermaid diagram generator. Respond only in raw JSON."}

# -----------------------------------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------------------------------

@app.post("/api/update-config", status_code=status.HTTP_200_OK)
async def update_system_configuration(config: Dict[str, Any]):
    new_prompt = config.get("base_prompt")
    if not new_prompt:
        raise HTTPException(status_code=400, detail="Missing base_prompt")
    SYSTEM_CONFIG["base_prompt"] = str(new_prompt)
    return {"status": "success", "new_config": SYSTEM_CONFIG}

@app.post("/api/chat")
async def process_chat_stream(request: Request) -> StreamingResponse:
    body = await request.json()
    raw_messages = body.get("messages", [])
    
    strict_json_prompt = (
        f"{SYSTEM_CONFIG['base_prompt']}\n"
        "Respond strictly with JSON schema: {\"mermaid_code\": \"...\", \"explanation\": \"...\"}"
    )

    compiled_messages = [{"role": "system", "content": strict_json_prompt}] + raw_messages[-3:]

    async def chat_sse_stream_generator():
        try:
            # ملاحظة: Groq لا يدعم response_format={"type": "json_object"} في بعض الموديلات
            # لذا سنعتمد على البرومبت الصارم
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=compiled_messages,
                temperature=0.1
            )
            
            raw_content = response.choices[0].message.content
            # تنظيف الرد من علامات الكود إذا وجدت
            clean_json = raw_content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_json)
            
            yield f"data: {json.dumps({'content': parsed.get('mermaid_code', '')})}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(chat_sse_stream_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
