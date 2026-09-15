"""Folder-upload routes — user-supplied .zarr / .geff movies (Stage 00)."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from modules import module_upload as up

router = APIRouter(prefix="/api/upload", tags=["upload"])


@router.post("/zarr")
async def upload_zarr(
    movie_name: str = Form(...),
    files: List[UploadFile] = File(...),
    paths: List[str] = Form(...),
):
    try:
        result = await up.commit_folder("zarr", movie_name, files, paths)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "result": result}


@router.post("/geff")
async def upload_geff(
    movie_name: str = Form(...),
    files: List[UploadFile] = File(...),
    paths: List[str] = Form(...),
):
    try:
        result = await up.commit_folder("geff", movie_name, files, paths)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "result": result}


@router.get("/registry")
def registry():
    return {"ok": True, "result": up.load_registry()}


@router.delete("/movie")
def delete_movie(movie_id: str = Query(...)):
    try:
        result = up.delete_movie(movie_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "result": result}


@router.get("/train-contents")
def train_contents():
    return {"ok": True, "result": up.train_contents()}
