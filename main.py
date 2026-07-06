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
logger = logging.getLogger("groq_mirror")

# -----------------------------------------------------------------------------
# ENVIRONMENT & CONFIGURATION
# -----------------------------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.critical("Initialization failed: GROQ_API_KEY environment variable is not set.")
    raise ValueError("GROQ_API_KEY is not set.")

MODEL_NAME = "llama-3.3-8b-instant"
BASE_URL = "https://api.groq.com/openai/v1"

# In-memory shared configuration
SYSTEM_CONFIG: Dict[str, str] = {
    "base_prompt": (
        "You are a Mermaid generator. Rules:\n"
        "1. Start exactly with ```mermaid followed by a newline.\n"
        "2. Write 'graph TD' or 'graph LR' on the next line.\n"
        "3. Never combine 'mermaid' and 'graph' into a single word like 'mermaidgraph'.\n"
        "4. Do not include introductory or concluding text outside the block."
    )
}

# -----------------------------------------------------------------------------
# INITIALIZATION
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Groq Mirror Professional",
    version="1.0.0",
    description="Production-ready FastAPI middleware linking C# desktop applications, browsers, and Groq API via SSE."
)

# Enable CORS for cross-origin local desktop application architectures
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up global asynchronous broadcasting queue with a maximum size limit
broadcast_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=10)

# Mount statics and templates safely
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize asynchronous OpenAI client for Groq
client = AsyncOpenAI(base_url=BASE_URL, api_key=GROQ_API_KEY)

# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS / HELPER LOGIC
# -----------------------------------------------------------------------------
def sanitize_mermaid_syntax(text: str) -> str:
    """
    Corrects common LLM formatting abnormalities related to Mermaid blocks.
    Fixes inline 'mermaidgraph' and code block fence attachments.
    """
    if not text:
        return text
    text = text.replace("mermaidgraph", "mermaid\ngraph")
    text = text.replace("```mermaidgraph", "```mermaid\ngraph")
    return text

async def safely_enqueue_broadcast(content: str) -> None:
    """
    Attempts to place content into the global broadcast queue.
    If the queue is full, clears the oldest item to prevent memory leaks and blocking.
    """
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
    """
    Renders and serves the web frontend UI view.
    """
    logger.info("GET / - Serving application mirror frontend interface.")
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
    """
    Simple application liveness and readiness probe endpoint.
    """
    return {"status": "healthy"}

@app.post("/api/update-config", status_code=status.HTTP_200_OK)
async def update_system_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dynamically modifies the system context prompt configuration instructions runtime.
    """
    new_prompt = config.get("base_prompt")
    if new_prompt is None:
        logger.warning("POST /api/update-config - Missing required field 'base_prompt'")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing payload attribute: 'base_prompt' field is required."
        )
    
    SYSTEM_CONFIG["base_prompt"] = str(new_prompt)
    logger.info(f"System system configuration modified. New base prompt: {SYSTEM_CONFIG['base_prompt']}")
    return {"status": "success", "new_config": SYSTEM_CONFIG}

@app.get("/api/stream-mirror")
async def stream_mirror_events() -> StreamingResponse:
    """
    Server-Sent Events endpoint broadcasting real-time stream state out to secondary consumers.
    """
    logger.info("GET /api/stream-mirror - Connection established for auxiliary listener.")
    
    async def mirror_event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                content = await broadcast_queue.get()
                payload = json.dumps({"content": content})
                yield f"data: {payload}\n\n"
                broadcast_queue.task_done()
        except asyncio.CancelledError:
            logger.info("GET /api/stream-mirror - Auxiliary consumer client disconnected from stream.")
        except Exception as e:
            logger.error(f"Unexpected error inside mirror queue stream generator: {str(e)}")
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
    """
    Processes chat prompts, truncates history contexts to save processing cost overhead tokens,
    communicates downstream with the Groq API infrastructure, and returns an optimized SSE chunk stream.
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        logger.error("POST /api/chat - Invalid JSON document structure submitted.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Malformed JSON request payload structure.")
    
    raw_messages: List[Dict[str, Any]] = body.get("messages", [])
    if not raw_messages or not isinstance(raw_messages, list):
        logger.warning("POST /api/chat - Missing validation rule elements inside input structure.")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Parameter field 'messages' must be a populated array.")

    # Only keep the last 3 messages for history token optimization
    optimized_history = raw_messages[-3:]
    
    logger.info("=" * 60)
    logger.info(f"Execution Target Model Pipeline: {MODEL_NAME}")
    logger.info(f"Incoming Frame Count: {len(raw_messages)} | Sliced Context Count: {len(optimized_history)}")
    for idx, msg in enumerate(optimized_history):
        role = msg.get("role", "unknown")
        content_len = len(str(msg.get("content", "")))
        logger.info(f" [{idx + 1}] Role: '{role}' -> Magnitude: {content_len} characters.")
    logger.info("=" * 60)

    # Prepend system prompt
    compiled_messages = [{"role": "system", "content": SYSTEM_CONFIG["base_prompt"]}]
    for msg in optimized_history:
        compiled_messages.append({
            "role": msg.get("role", "user"),
            "content": str(msg.get("content", ""))
        })

    async def chat_sse_stream_generator() -> AsyncGenerator[str, None]:
        try:
            response_stream = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=compiled_messages,
                stream=True,
                temperature=0.2,
                top_p=0.9,
                max_tokens=1024
            )
            
            async for chunk in response_stream:
                if not chunk.choices:
                    continue
                
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    # Automatically fix mermaid syntax
                    processed_token = sanitize_mermaid_syntax(delta_content)
                    
                    await safely_enqueue_broadcast(processed_token)
                    yield f"data: {json.dumps({'content': processed_token})}\n\n"
            
            yield "data: [DONE]\n\n"
            logger.info("POST /api/chat - Downstream streaming pipeline transmission completed successfully.")
            
        except APIError as api_err:
            logger.error(f"Groq Cloud API Connection Interface Failure: {str(api_err)}")
            yield f"data: {json.dumps({'error': f'Groq service interface connection error: {api_err.message}'})}\n\n"
        except asyncio.TimeoutError:
            logger.error("Timeout threshold reached while awaiting processing nodes responses.")
            yield f"data: {json.dumps({'error': 'Upstream request sequence processing timed out.'})}\n\n"
        except asyncio.CancelledError:
            logger.warning("Upstream client severed the processing response pipeline before execution concluded.")
        except Exception as general_err:
            logger.error(f"Unmanaged operational structural crash during runtime chunk sequences: {str(general_err)}")
            yield f"data: {json.dumps({'error': 'Internal operational execution matrix failure occurred.'})}\n\n"

    custom_headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
        "X-Accel-Buffering": "no"
    }
    return StreamingResponse(chat_sse_stream_generator(), media_type="text/event-stream", headers=custom_headers)

# -----------------------------------------------------------------------------
# APPLICATION ENTRYPOINT EXECUTION ARCHITECTURE
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    target_port = int(os.getenv("PORT", 8000))
    logger.info(f"Spinning up production ASGI web server pipeline on interface binding 0.0.0.0:{target_port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=target_port,
        workers=1,
        log_level="info",
        reload=False
    )
