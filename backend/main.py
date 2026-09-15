from __future__ import annotations
import sys
from pathlib import Path

# ---------------------------------------------------------
# Backend root
# ---------------------------------------------------------

BACKEND_ROOT = Path(__file__).resolve().parent

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


# ---------------------------------------------------------
# FastAPI imports
# ---------------------------------------------------------

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

from config import (
    API_TITLE,
    API_VERSION,
    CORS_ORIGINS,
    LOCAL_DATA_ROOT,
    ensure_artifact_dirs,
    spacing_dict,
)

from routers import celldino, data_prep, spatiotemporal, gat, tracking, journey, upload


# ---------------------------------------------------------
# Initialize directories
# ---------------------------------------------------------

ensure_artifact_dirs()


# ---------------------------------------------------------
# Create application
# ---------------------------------------------------------

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Routers
# ---------------------------------------------------------

app.include_router(data_prep.router)
app.include_router(celldino.router)
app.include_router(spatiotemporal.router)
app.include_router(gat.router)
app.include_router(tracking.router)
app.include_router(journey.router)
app.include_router(upload.router)


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "CellTrace",
        "version": API_VERSION,
        "review": "review2",
        "modules": {
            "module0": "complete",
            "module1": "complete-prototype",
            "module2": "complete",
            "module3": "complete-contrastive",
            "module4": "half-matching-confidence",
        },
        "spacing_um": spacing_dict(),
        "local_data": str(LOCAL_DATA_ROOT),
    }


# ---------------------------------------------------------
# Root
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "CellTrace API",
        "docs": "/docs",
        "health": "/api/health",
    }


# ---------------------------------------------------------
# Run directly
# ---------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )