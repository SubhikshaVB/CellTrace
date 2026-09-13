"""Module 0 FastAPI routes — dataset load, validate, patches, tensors, slices."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from modules import module0_data_prep as m0

router = APIRouter(prefix="/api/module0", tags=["module0"])


class MovieRequest(BaseModel):
    movie_id: str = Field(..., min_length=1)
    max_cells: Optional[int] = Field(default=256, ge=1, le=100000)


@router.get("/datasets")
def list_datasets():
    return m0.safe_call(m0.list_datasets)


@router.post("/load")
def load_dataset(body: MovieRequest):
    out = m0.safe_call(m0.load_movie, body.movie_id)
    if not out["ok"]:
        raise HTTPException(status_code=400, detail=out["error"])
    return out


@router.post("/validate")
def validate(body: MovieRequest):
    out = m0.safe_call(m0.validate_movie, body.movie_id)
    if not out["ok"]:
        raise HTTPException(status_code=400, detail=out["error"])
    return out


@router.post("/extract")
def extract(body: MovieRequest):
    out = m0.safe_call(m0.extract_cell_patches, body.movie_id, body.max_cells)
    if not out["ok"]:
        raise HTTPException(status_code=400, detail=out["error"])
    return out


@router.post("/standardize")
def standardize(
    movie_id: str = Query(...),
    t: int = Query(0, ge=0),
    z: int = Query(0, ge=0),
):
    out = m0.safe_call(m0.standardize_slice, movie_id, t, z)
    if not out["ok"]:
        raise HTTPException(status_code=400, detail=out["error"])
    return out


@router.post("/tensorize")
def tensorize(body: MovieRequest):
    out = m0.safe_call(m0.tensorize_movie, body.movie_id, body.max_cells)
    if not out["ok"]:
        raise HTTPException(status_code=400, detail=out["error"])
    return out


@router.post("/run")
def run_all(body: MovieRequest):
    out = m0.safe_call(m0.run_module0, body.movie_id, body.max_cells)
    if not out["ok"]:
        raise HTTPException(status_code=400, detail=out["error"])
    return out


@router.get("/slice")
def slice_view(
    movie_id: str = Query(...),
    t: int = Query(0, ge=0),
    z: int = Query(0, ge=0),
):
    out = m0.safe_call(m0.get_slice, movie_id, t, z)
    if not out["ok"]:
        raise HTTPException(status_code=400, detail=out["error"])
    return out

@router.get("/inspect")
def inspect_dataset(
    movie_id: str = Query(...),
    t: int = Query(0, ge=0),
    z: int = Query(0, ge=0),
):
    out = m0.safe_call(m0.inspect_dataset, movie_id, t, z)
    if not out["ok"]:
        raise HTTPException(status_code=400, detail=out["error"])
    return out


@router.get("/geff-overlay")
def geff_overlay(
    movie_id: str = Query(...),
    t: int = Query(0, ge=0),
    z: int = Query(0, ge=0),
):
    out = m0.safe_call(m0.get_geff_overlay, movie_id, t, z)
    if not out["ok"]:
        raise HTTPException(status_code=400, detail=out["error"])
    return out

@router.get("/status")
def status():
    return m0.module_status()
