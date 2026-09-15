"""Module 2 FastAPI routes — complete spatiotemporal feature fusion."""
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from modules import module2_spatiotemporal as m2

router = APIRouter(prefix="/api/module2", tags=["module2"])


class MovieRequest(BaseModel):
    movie_id: str = Field(..., min_length=1)


@router.post("/run")
def run(body: MovieRequest):
    out = m2.safe_call(m2.run_full, body.movie_id)

    if not out["ok"]:
        raise HTTPException(status_code=400, detail=out["error"])

    return out

@router.get("/graph-preview")
def graph_preview(
    movie_id: str,
    frame: Optional[int] = None,
    focus_node_id: Optional[int] = None,
):
    out = m2.safe_call(
        m2.graph_preview,
        movie_id,
        frame,
        focus_node_id,
    )

    if not out["ok"]:
        raise HTTPException(status_code=400, detail=out["error"])

    return out

@router.get("/status")
def status():
    return {"ok": True, "result": m2.module_status()}