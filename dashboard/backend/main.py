"""
dashboard/backend/main.py
==========================
FastAPI application entry point for the Satellite Orbit Lab.

Usage
-----
From the project root:
    uvicorn dashboard.backend.main:app --reload --port 8000

Or from dashboard/backend/:
    uvicorn main:app --reload --port 8000

CORS is configured for local development (localhost:5173 — Vite default).
For production, update the origins list.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Resolve project root so src/ is importable regardless of CWD
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent
_DASHBOARD_DIR = _BACKEND_DIR.parent
_PROJECT_ROOT = _DASHBOARD_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("satellite_orbit_lab")

# ---------------------------------------------------------------------------
# Service singletons (lazy initialised at first request)
# ---------------------------------------------------------------------------
_registry = None
_sgp4_service = None
_pistn_service = None
_eval_service = None


def get_registry():
    global _registry
    if _registry is None:
        from dashboard.backend.satellite_registry import SatelliteRegistry
        _registry = SatelliteRegistry(_PROJECT_ROOT)
        log.info("SatelliteRegistry initialised | root=%s", _PROJECT_ROOT)
    return _registry


def get_sgp4_service():
    global _sgp4_service
    if _sgp4_service is None:
        from dashboard.backend.services.sgp4_service import SGP4Service
        _sgp4_service = SGP4Service(_PROJECT_ROOT)
        log.info("SGP4Service initialised")
    return _sgp4_service


def get_pistn_service():
    global _pistn_service
    if _pistn_service is None:
        from dashboard.backend.services.pistn_service import PISTNRolloutService
        _pistn_service = PISTNRolloutService(_PROJECT_ROOT)
        log.info("PISTNRolloutService initialised")
    return _pistn_service


def get_eval_service():
    global _eval_service
    if _eval_service is None:
        from dashboard.backend.services.evaluation_service import EvaluationService
        _eval_service = EvaluationService(_PROJECT_ROOT)
        log.info("EvaluationService initialised")
    return _eval_service


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Satellite Orbit Lab API",
    description=(
        "Backend API for the Satellite Orbit Lab dashboard. "
        "Wraps the SGP4 propagation pipeline, PI-STN neural model inference, "
        "and evaluation metrics from the Satellite PINN State Estimation research project."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Vite dev server and production frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://localhost:4173",  # vite preview
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------
from dashboard.backend.api.satellites import router as satellites_router
from dashboard.backend.api.simulation import router as simulation_router
from dashboard.backend.api.prediction import router as prediction_router
from dashboard.backend.api.evaluation import router as evaluation_router

app.include_router(satellites_router)
app.include_router(simulation_router)
app.include_router(prediction_router)
app.include_router(evaluation_router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/api/health", tags=["health"])
def health():
    """Health check — returns project root and registered satellite IDs."""
    registry = get_registry()
    return {
        "status": "ok",
        "project_root": str(_PROJECT_ROOT),
        "satellites": registry.ids(),
        "version": "1.0.0",
    }

# ---------------------------------------------------------------------------
# Mount frontend static files if built
# ---------------------------------------------------------------------------
_FRONTEND_DIST = _DASHBOARD_DIR / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
    log.info("Frontend dist mounted at /")

