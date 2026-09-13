from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from modules import module4_theatre as theatre
from modules import module4_tracking as m4

router = APIRouter(prefix="/api/module4", tags=["module4"])


class TrackingRequest(BaseModel):
    movie_id: str = Field(..., min_length=1)
    group_weights: Optional[Dict[str, float]] = Field(default=None)
    tag: str = Field(default="")


class FitWeightsRequest(BaseModel):
    train_movie_ids: List[str] = Field(..., min_length=1)


@router.post("/run")
def run(body: TrackingRequest):
    tag = body.tag if body.tag in ("", "custom") else ""
    out = m4.safe_call(m4.ENGINE.run, body.movie_id, body.group_weights, tag)

    if not out["ok"]:
        raise HTTPException(status_code=400, detail=out["error"])

    return out


@router.post("/fit-weights")
def fit_weights(body: FitWeightsRequest):
    out = m4.safe_call(m4.fit_weights, body.train_movie_ids)

    if not out["ok"]:
        raise HTTPException(status_code=400, detail=out["error"])

    return out


@router.get("/theatre")
def theatre_bundle(movie_id: str, source: str = "fitted"):
    if source not in ("fitted", "custom"):
        raise HTTPException(status_code=400, detail="source must be fitted|custom")

    try:
        return {"ok": True, "result": theatre.build_theatre_bundle(movie_id, source)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/status")
def status():
    return {"ok": True, "result": m4.module_status()}
