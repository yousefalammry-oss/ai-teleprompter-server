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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("openai_json_mirror")

# -----------------------------------------------------------------------------
# ENVIRONMENT & CONFIGURATION
# -----------------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.critical("Initialization failed: OPENAI_API_KEY environment variable is not set.")
    raise ValueError("OPENAI_API_KEY is not set.")

MODEL_NAME = "gpt-4o-mini"

# الـ System Prompt المحسن لإنتاج رسومات بيانية صحيحة
SYSTEM_CONFIG: Dict[str, str] = {
    "base_prompt": (
        "You are an expert systems architect and strict Mermaid diagram generator.\n"
        "Your sole task is to generate valid, compilable Mermaid syntax based on the user request.\n"
        "Ensure every node connection and statement is separated on a proper new line."
    )
}

# -----------------------------------------------------------------------------
# INITIALIZATION
# -----------------------------------------------------------------------------
app = FastAPI(
    title="OpenAI JSON Mirror Pro",
    version="2.1.0",
    description="FastAPI middleware serving production-ready strict JSON responses via OpenAI API."
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

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

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

    optimized_history = raw_messages[-3:]
    
    logger.info("=" * 60)
    logger.info(f"Pipeline Engine: OpenAI (Strict JSON Mode) -> {MODEL_NAME}")
    logger.info("=" * 60)

    # إجبار الموديل على الالتزام ببنية الـ JSON مع تحديد الحقول المطلوبة
    strict_json_prompt = (
        f"{SYSTEM_CONFIG['base_prompt']}\n\n"
        "CRITICAL: You MUST respond with a raw JSON object matching this schema:\n"
        "{\n"
        "  \"mermaid_code\": \"The complete, valid and beautifully indented Mermaid diagram code starting with graph TD or LR.\",\n"
        "  \"explanation\": \"Brief textual overview or description of the diagram architecture.\"\n"
        "}"
    )

    compiled_messages = [{"role": "system", "content": strict_json_prompt}]
    for msg in optimized_history:
        compiled_messages.append({
            "role": msg.get("role", "user"),
            "content": str(msg.get("content", ""))
        })

    async def chat_sse_stream_generator() -> AsyncGenerator[str, None]:
        try:
            # استخدام الطريقة القياسية المستقرة والمتوافقة مع كافة إصدارات openai
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=compiled_messages,
                response_format={"type": "json_object"},  # تفعيل نمط الـ JSON الإجباري
                temperature=0.1
            )
            
            raw_content = response.choices[0].message.content
            if raw_content:
                # معالجة وتفكيك الـ JSON المستلم لاستخراج الكود النظيف والمفصول أسطرياً
                parsed_json = json.loads(raw_content)
                final_mermaid = parsed_json.get("mermaid_code", "")
                
                if final_mermaid:
                    # بث الكود النظيف والمستقر بالكامل للـ listener الفرعي دفعة واحدة
                    await safely_enqueue_broadcast(final_mermaid)
                    yield f"data: {json.dumps({'content': final_mermaid})}\n\n"
            
            # بث إشارة الانتهاء لإبلاغ الفرونت إند ببدء الرسم
            yield "data: [DONE]\n\n"
            logger.info("POST /api/chat - JSON Structured response completed successfully.")
            
        except APIError as api_err:
            logger.error(f"OpenAI API Interface Failure: {str(api_err)}")
            yield f"data: {json.dumps({'error': f'OpenAI service error: {api_err.message}'})}\n\n"
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON schema from model response.")
            yield f"data: {json.dumps({'error': 'Model failed to output a strict valid JSON structure.'})}\n\n"
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
