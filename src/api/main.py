"""FastAPI entrypoint for LégiRoute."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes import chat, health
from src.version import __version__

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = FastAPI(
    title="LégiRoute API",
    description="HTTP API for the LégiRoute RAG assistant (French Highway Code).",
    version=__version__,
)

default_origins = "http://localhost:5173,http://127.0.0.1:5173"
cors_allow_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", default_origins).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["meta"])
app.include_router(chat.router, prefix="/api", tags=["chat"])

frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend_dist.exists():
    # Mount frontend after API routes so /api/* keeps priority.
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
