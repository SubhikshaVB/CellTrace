# """Module 1 FastAPI routes — CellDINO appearance embeddings."""
# from __future__ import annotations

# from typing import Optional

# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel, Field

# from modules import module1_celldino as m1

# router = APIRouter(prefix="/api/module1", tags=["module1"])


# class MovieRequest(BaseModel):
#     movie_id: str = Field(..., min_length=1)
#     max_cells: Optional[int] = Field(default=None, ge=1, le=100000)


# @router.post("/run")
# def run(body: MovieRequest):
#     out = m1.safe_call(m1.run_celldino, body.movie_id, body.max_cells)
#     if not out["ok"]:
#         raise HTTPException(status_code=400, detail=out["error"])
#     return out


# @router.get("/embeddings/{movie_id}")
# def embeddings(movie_id: str):
#     out = m1.safe_call(m1.get_embedding_meta, movie_id)
#     if not out["ok"]:
#         raise HTTPException(status_code=404, detail=out["error"])
#     return out


# @router.get("/similarity/{movie_id}")
# def similarity(movie_id: str):
#     out = m1.safe_call(m1.similarity_preview, movie_id)
#     if not out["ok"]:
#         raise HTTPException(status_code=404, detail=out["error"])
#     return out


# @router.get("/status")
# def status():
#     return m1.module_status()

"""Module 1 FastAPI routes — CellDINO appearance embeddings and training."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from modules import module1_celldino as m1
from modules import module1_training as training

router = APIRouter(prefix="/api/module1", tags=["module1"])


class MovieRequest(BaseModel):
    movie_id: str = Field(..., min_length=1)
    max_cells: Optional[int] = Field(default=None, ge=1, le=100000)


class TrainingRequest(BaseModel):
    epochs: int = Field(default=4, ge=1, le=100)
    batch_size: int = Field(default=4, ge=1, le=32)
    learning_rate: float = Field(default=1e-4, gt=0, le=1e-2)
    source_cells_per_movie: int = Field(default=64, ge=8, le=512)
    samples_per_movie: int = Field(default=16, ge=4, le=128)
    seed: int = Field(default=42, ge=0)


@router.post("/run")
def run(body: MovieRequest):
    out = m1.safe_call(m1.run_celldino, body.movie_id, body.max_cells)
    if not out["ok"]:
        raise HTTPException(status_code=400, detail=out["error"])
    return out


@router.post("/train")
def train(body: TrainingRequest):
    try:
        result = training.train_self_supervised(
            epochs=body.epochs,
            batch_size=body.batch_size,
            learning_rate=body.learning_rate,
            source_cells_per_movie=body.source_cells_per_movie,
            samples_per_movie=body.samples_per_movie,
            seed=body.seed,
        )
        return {"ok": True, "result": result}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/training-status")
def training_status():
    return {"ok": True, "result": training.training_status()}


@router.get("/embeddings/{movie_id}")
def embeddings(movie_id: str):
    out = m1.safe_call(m1.get_embedding_meta, movie_id)
    if not out["ok"]:
        raise HTTPException(status_code=404, detail=out["error"])
    return out


@router.get("/similarity/{movie_id}")
def similarity(movie_id: str):
    out = m1.safe_call(m1.similarity_preview, movie_id)
    if not out["ok"]:
        raise HTTPException(status_code=404, detail=out["error"])
    return out


@router.get("/status")
def status():
    return m1.module_status()


@router.get("/vector/{movie_id}")
def vector(movie_id: str, node_id: int):
    out = m1.safe_call(m1.get_node_vector, movie_id, node_id)
    if not out["ok"]:
        raise HTTPException(status_code=404, detail=out["error"])
    return out


@router.get("/neighbors/{movie_id}")
def neighbors(movie_id: str, node_id: int, k: int = 5):
    out = m1.safe_call(m1.node_neighbors, movie_id, node_id, k)
    if not out["ok"]:
        raise HTTPException(status_code=404, detail=out["error"])
    return out