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

# --------------------------------------------------
# Load Environment
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-8b-instant"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set.")

# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info(f"Using Model: {MODEL_NAME}")
logger.info("=" * 60)

# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(title="Groq Mirror Professional")

broadcast_queue = asyncio.Queue(maxsize=10)

SYSTEM_CONFIG = {
    "base_prompt":
    """You are a professional AI assistant.

If you generate Mermaid diagrams:

- Always start with:

```mermaid
graph TD
