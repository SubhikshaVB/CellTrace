from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from modules import module3_gat as m3

router = APIRouter(prefix="/api/module3", tags=["module3"])


class GATRequest(BaseModel):
    movie_id: str = Field(..., min_length=1)
    epochs: int = Field(default=50, ge=1, le=500)


@router.post("/run")
def run(body: GATRequest):
    out = m3.safe_call(m3.run_gat, body.movie_id, body.epochs)

    if not out["ok"]:
        raise HTTPException(status_code=400, detail=out["error"])

    return out


@router.get("/status")
def status():
    return {"ok": True, "result": m3.module_status()}