import os
import json
import logging
import asyncio
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from dotenv import load_dotenv
import uvicorn

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("groq_mirror")
load_dotenv()

# إعدادات Groq
API_KEY = os.getenv("GROQ_API_KEY")
BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "llama-3.3-8b-instant"

if not API_KEY:
    raise ValueError("GROQ_API_KEY غير موجود في إعدادات البيئة!")

# -----------------------------------------------------------------------------
# INITIALIZATION
# -----------------------------------------------------------------------------
app = FastAPI(title="Groq Mirror API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
SYSTEM_CONFIG = {"base_prompt": "You are a strict Mermaid diagram generator. Respond only in raw JSON."}

# -----------------------------------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------------------------------

@app.post("/api/update-config")
async def update_system_configuration(config: Dict[str, Any]):
    new_prompt = config.get("base_prompt")
    if not new_prompt:
        raise HTTPException(status_code=400, detail="Missing base_prompt")
    SYSTEM_CONFIG["base_prompt"] = str(new_prompt)
    return {"status": "success"}

@app.post("/api/chat")
async def process_chat_stream(request: Request) -> StreamingResponse:
    body = await request.json()
    raw_messages = body.get("messages", [])
    
    # تحضير الرسائل
    compiled_messages = [{"role": "system", "content": f"{SYSTEM_CONFIG['base_prompt']} Return JSON only."}] + raw_messages[-3:]

    async def chat_sse_stream_generator():
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=compiled_messages,
                temperature=0.1
            )
            
            raw_content = response.choices[0].message.content
            # تنظيف الـ JSON
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
