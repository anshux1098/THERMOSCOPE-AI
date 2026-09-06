"""
main.py — THERMOSCOPE-AI FastAPI entry point (Production Intelligence Engine).

Exposes the existing hotspot-analysis service as a thin HTTP layer for the
frontend dashboard. CORS origins come from app.core.config (pydantic-settings,
.env aware). No classification logic lives here.

Run:
    uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.api.hotspots import router as hotspots_router

settings = get_settings()

app = FastAPI(
    title="THERMOSCOPE-AI — Production Intelligence Engine",
    description=(
        "Thermal-hotspot classification backend: NASA FIRMS ingestion, OSM spatial "
        "context, 14 explainable labeling functions, weak supervision, XGBoost and "
        "hybrid Rules x ML decision engine, human-review logic."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(hotspots_router, prefix="/api/v1")


@app.get("/health", summary="Liveness probe")
def health() -> dict:
    return {"status": "ok", "engine": "thermoscope-ai"}