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
from openai import AsyncOpenAI, APIError  # استخدام مكتبة OpenAI الرسمية بشكل مباشر
from dotenv import load_dotenv
import uvicorn
import re

# -----------------------------------------------------------------------------
# LOGGING CONFIGURATION
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("openai_mirror")

# -----------------------------------------------------------------------------
# ENVIRONMENT & CONFIGURATION
# -----------------------------------------------------------------------------
load_dotenv()

# جلب مفتاح الـ API الخاص بـ OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.critical("Initialization failed: OPENAI_API_KEY environment variable is not set.")
    raise ValueError("OPENAI_API_KEY is not set.")

# تعيين الموديل الافتراضي المستقر جداً في هيكلة النصوص وكتابة كود ميرميد نظيف
MODEL_NAME = "gpt-4o-mini" 

# الـ System Prompt المحسن والموجه بدقة لإنتاج أسطر برمجية منفصلة
SYSTEM_CONFIG: Dict[str, str] = {
    "base_prompt": (
        "You are a strict Mermaid diagram generator. Rules:\n"
        "1. Output ONLY valid, compilable mermaid code blocks.\n"
        "2. Start exactly with ```mermaid followed by a newline.\n"
        "3. Write 'graph TD' or 'graph LR' on the next line.\n"
        "4. Every single node connection (e.g., A --> B) MUST be written on a completely separate new line.\n"
        "5. Do not include any conversational text or summary before or after the code block."
    )
}

# -----------------------------------------------------------------------------
# INITIALIZATION
# -----------------------------------------------------------------------------
app = FastAPI(
    title="OpenAI Mirror Professional",
    version="1.0.0",
    description="Production-ready FastAPI middleware linking architectures to OpenAI API via SSE."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

broadcast_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=10)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# تهيئة عميل OpenAI غير المتزامن (Async Client) رسمي ومباشر
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS / HELPER LOGIC
# -----------------------------------------------------------------------------
def sanitize_mermaid_syntax(text: str) -> str:
    """
    دالة تنظيف ومعالجة كود Mermaid الملتصق وفصله بأسطر جديدة تلقائياً.
    """
    if not text:
        return text
    
    text = re.sub(r'mermaidgraph\s*(TB|TD|LR|BT)', r'mermaid\ngraph \1', text)
    text = text.replace("mermaidgraph", "mermaid\ngraph TD")
    text = text.replace("```mermaidgraph", "```mermaid\ngraph TD")
    
    # تصحيح العلاقات الملتصقة
    text = re.sub(r'(\])\s*([A-Za-z0-9_]+)(\[|\()', r'\1\n\2\3', text)
    text = re.sub(r'(\S+-->\S+)\s+([A-Za-z0-9_]+-->)', r'\1\n\2', text)
    text = re.sub(r'(\S+-\s*>\s*>\S+)\s+([A-Za-z0-9_]+-\s*>\s*>)', r'\1\n\2', text)
    
    return text

async def safely_enqueue_broadcast(content: str) -> None:
    if broadcast_queue.full():
        try:
            broadcast_queue.get_nowait()
            broadcast_queue.task_done()
        except asyncio.QueueEmpty:
            pass
    await broadcast_queue.put(content)

# -----------------------------------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def render_index_page(request: Request):
    try:
        return templates.TemplateResponse("mirror.html", {"request": request})
    except Exception as e:
        logger.error(f"Failed to render template: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Frontend template missing or improperly configured."
        )

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}

@app.post("/api/update-config", status_code=status.HTTP_200_OK)
async def update_system_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    new_prompt = config.get("base_prompt")
    if new_prompt is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing payload attribute: 'base_prompt' field is required."
        )
    
    SYSTEM_CONFIG["base_prompt"] = str(new_prompt)
    logger.info(f"Configuration updated. New base prompt: {SYSTEM_CONFIG['base_prompt']}")
    return {"status": "success", "new_config": SYSTEM_CONFIG}

@app.get("/api/stream-mirror")
async def stream_mirror_events() -> StreamingResponse:
    async def mirror_event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                content = await broadcast_queue.get()
                payload = json.dumps({"content": content})
                yield f"data: {payload}\n\n"
                broadcast_queue.task_done()
        except asyncio.CancelledError:
            logger.info("Auxiliary consumer client disconnected from stream.")
        except Exception as e:
            logger.error(f"Error inside mirror queue stream generator: {str(e)}")
            yield f"data: {json.dumps({'error': 'Internal server broadcasting failure'})}\n\n"

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
        "X-Accel-Buffering": "no"
    }
    return StreamingResponse(mirror_event_generator(), media_type="text/event-stream", headers=headers)

@app.post("/api/chat")
async def process_chat_stream(request: Request) -> StreamingResponse:
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Malformed JSON request payload structure.")
    
    raw_messages: List[Dict[str, Any]] = body.get("messages", [])
    if not raw_messages or not isinstance(raw_messages, list):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Parameter field 'messages' must be a populated array.")

    # تحسين الـ Context عبر إبقاء آخر 3 رسائل فقط
    optimized_history = raw_messages[-3:]
    
    logger.info("=" * 60)
    logger.info(f"Pipeline Engine: OpenAI -> {MODEL_NAME}")
    logger.info("=" * 60)

    compiled_messages = [{"role": "system", "content": SYSTEM_CONFIG["base_prompt"]}]
    for msg in optimized_history:
        compiled_messages.append({
            "role": msg.get("role", "user"),
            "content": str(msg.get("content", ""))
        })

    async def chat_sse_stream_generator() -> AsyncGenerator[str, None]:
        try:
            # استدعاء الـ API من OpenAI مع تشغيل ميزة البث (stream=True)
            response_stream = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=compiled_messages,
                stream=True,
                temperature=0.1,  # درجة حرارة منخفضة لضمان التزام تام بالهيكلية البرمجية
                max_tokens=1500
            )
            
            async for chunk in response_stream:
                if not chunk.choices:
                    continue
                
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    # معالجة النصوص التالفة برمجياً لضمان سلامتها قبل بثها
                    processed_token = sanitize_mermaid_syntax(delta_content)
                    
                    await safely_enqueue_broadcast(processed_token)
                    yield f"data: {json.dumps({'content': processed_token})}\n\n"
            
            # إرسال إشارة التوقف المعتمدة بالفرونت إند لإنهاء المعالجة ورسم المخطط
            yield "data: [DONE]\n\n"
            logger.info("POST /api/chat - OpenAI Stream transmission completed successfully.")
            
        except APIError as api_err:
            logger.error(f"OpenAI API Connection Interface Failure: {str(api_err)}")
            yield f"data: {json.dumps({'error': f'OpenAI service error: {api_err.message}'})}\n\n"
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'error': 'Upstream request sequence processing timed out.'})}\n\n"
        except asyncio.CancelledError:
            logger.warning("Client severed the processing response pipeline.")
        except Exception as general_err:
            logger.error(f"Operational structural crash: {str(general_err)}")
            yield f"data: {json.dumps({'error': 'Internal operational failure occurred.'})}\n\n"

    custom_headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
        "X-Accel-Buffering": "no"
    }
    return StreamingResponse(chat_sse_stream_generator(), media_type="text/event-stream", headers=custom_headers)

# -----------------------------------------------------------------------------
# APPLICATION ENTRYPOINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    target_port = int(os.getenv("PORT", 8000))
    logger.info(f"Spinning up production ASGI web server on 0.0.0.0:{target_port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=target_port,
        workers=1,
        log_level="info",
        reload=False
    )
