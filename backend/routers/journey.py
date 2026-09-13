"""Guided Journey routes — geff cloud, artifact summaries, GAT attention."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from modules import module_journey as J

router = APIRouter(prefix="/api/journey", tags=["journey"])


@router.get("/geff-cloud")
def geff_cloud(movie_id: str = Query(...), max_points: int = Query(3000, ge=200, le=20000)):
    try:
        return {"ok": True, "result": J.geff_cloud(movie_id, max_points)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/artifact")
def artifact(name: str = Query(...), movie_id: str = Query("")):
    try:
        return {"ok": True, "result": J.read_summary(name, movie_id)}
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/attention")
def attention(movie_id: str = Query(...), node_id: int = Query(...),
              k: int = Query(8, ge=1, le=32)):
    try:
        return {"ok": True, "result": J.focus_attention(movie_id, node_id, k)}
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
