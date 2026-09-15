"""User folder upload for Stage 00 (Specimen).

Accepts whole ``.zarr`` / ``.geff`` *folders* (directory upload from the
browser), stages them under ``backend/uploads/_staging/``, validates the real
store contents, then moves them into ``backend/local_data/train/`` so the
entire pipeline (discovery, load, validate, extract, …) works unchanged.

Safety rules:
- every relative path is jailed inside the staging dir (no ``..`` escapes);
- folder kind is enforced by suffix AND by real store validation;
- failed uploads never touch the train directory (staging is deleted);
- only registry-tracked uploads can be deleted (vault data is protected).
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import aiofiles
from fastapi import UploadFile

from config import BACKEND_ROOT, LOCAL_DATA_ROOT
from utils.io_geff import load_geff
from utils.io_zarr import inspect_zarr

LOGGER = logging.getLogger("celltrace.upload")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

UPLOAD_ROOT: Path = BACKEND_ROOT / "uploads"
STAGING_ROOT: Path = UPLOAD_ROOT / "_staging"
REGISTRY_PATH: Path = UPLOAD_ROOT / "registry.json"

MAX_UPLOAD_BYTES: int = 50 * 1024**3  # 50 GiB per folder; documented, generous
CHUNK_BYTES: int = 1024 * 1024  # 1 MiB streaming writes
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}")


def _ensure_dirs() -> None:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    LOCAL_DATA_ROOT.mkdir(parents=True, exist_ok=True)


def load_registry() -> Dict[str, Any]:
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_registry(registry: Dict[str, Any]) -> None:
    _ensure_dirs()
    tmp = REGISTRY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    tmp.replace(REGISTRY_PATH)


def sanitize_name(name: str) -> str:
    name = (name or "").strip()
    if not NAME_RE.fullmatch(name):
        raise ValueError(
            f"Invalid movie name {name!r}: use 1-128 chars of A–Z a–z 0–9 _ . -"
        )
    return name


def _safe_join(root: Path, rel: str) -> Path:
    """Join an uploader-supplied relative path, jailing it under root."""
    rel_path = Path(rel.replace("\\", "/"))
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise ValueError(f"Rejected unsafe path: {rel!r}")
    target = (root / rel_path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Rejected escaping path: {rel!r}")
    return target


async def _store_files(
    files: List[UploadFile], paths: List[str], dest_root: Path
) -> Tuple[int, int]:
    if not files:
        raise ValueError("Upload contained zero files — select a folder, not an empty dir")
    if len(files) != len(paths):
        raise ValueError(
            f"File/path count mismatch: {len(files)} files vs {len(paths)} paths"
        )
    total = 0
    for up_file, rel in zip(files, paths):
        target = _safe_join(dest_root, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(target, "wb") as fh:
            while True:
                chunk = await up_file.read(CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise ValueError(
                        f"Upload exceeds {MAX_UPLOAD_BYTES / 1024**3:.0f} GiB cap"
                    )
                await fh.write(chunk)
        await up_file.close()
    return len(files), total


def _unique_movie_id(base: str, kind: str) -> str:
    candidate = base
    n = 0
    while (LOCAL_DATA_ROOT / f"{candidate}.{kind}").exists():
        n += 1
        candidate = f"{base}_u{n}"
    return candidate


def validate_zarr_store(path: Path) -> Dict[str, Any]:
    if not (path / "zarr.json").exists():
        raise ValueError(f"{path.name} is not a Zarr v3 store (missing zarr.json)")
    try:
        volume = inspect_zarr(path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Zarr store failed to open: {exc}") from exc
    if len(volume.shape) < 3:
        raise ValueError(
            "Store does not look like a microscopy volume (need a ≥3D array) — "
            "did you swap the .zarr and .geff folders?"
        )
    return {
        "array_key": volume.array_key,
        "shape": list(volume.shape),
        "dtype": str(volume.dtype),
        "chunks": list(volume.chunks) if volume.chunks is not None else None,
    }


def validate_geff_store(path: Path) -> Dict[str, Any]:
    if not (path / "zarr.json").exists():
        raise ValueError(f"{path.name} is not a GEFF store (missing zarr.json)")
    try:
        graph = load_geff(path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"GEFF store failed to open: {exc}") from exc
    times = [int(node.t) for node in graph.nodes]
    return {
        "n_nodes": len(graph.nodes),
        "n_edges": len(graph.edges),
        "t_min": min(times) if times else None,
        "t_max": max(times) if times else None,
    }


def pair_status(movie_id: str) -> Dict[str, Any]:
    has_zarr = (LOCAL_DATA_ROOT / f"{movie_id}.zarr").is_dir()
    has_geff = (LOCAL_DATA_ROOT / f"{movie_id}.geff").is_dir()
    return {
        "movie_id": movie_id,
        "has_zarr": has_zarr,
        "has_geff": has_geff,
        "ready": has_zarr and has_geff,
    }


async def commit_folder(
    kind: str, movie_name: str, files: List[UploadFile], paths: List[str]
) -> Dict[str, Any]:
    """Stage, validate and install one uploaded folder. Returns a real report."""
    if kind not in ("zarr", "geff"):
        raise ValueError(f"kind must be 'zarr' or 'geff', got {kind!r}")
    _ensure_dirs()
    base = sanitize_name(movie_name)
    upload_id = uuid.uuid4().hex[:12]
    staging = STAGING_ROOT / upload_id
    staging.mkdir(parents=True, exist_ok=True)
    try:
        n_files, n_bytes = await _store_files(files, paths, staging)
        tops = sorted(p for p in staging.iterdir() if p.is_dir())
        if len(tops) != 1 or not tops[0].name.endswith(f".{kind}"):
            raise ValueError(
                f"Expected exactly one .{kind} folder, found: "
                + ", ".join(p.name for p in tops)
            )
        staged = tops[0]
        if kind == "zarr":
            validation = validate_zarr_store(staged)
        else:
            validation = validate_geff_store(staged)
        movie_id = _unique_movie_id(base, kind)
        dest = LOCAL_DATA_ROOT / f"{movie_id}.{kind}"
        shutil.move(str(staged), str(dest))
        registry = load_registry()
        entry = registry.get(movie_id, {"source": "upload"})
        entry.update(
            {
                "source": "upload",
                kind: {
                    "n_files": n_files,
                    "bytes": n_bytes,
                    "validation": validation,
                },
                "updated_at": time.time(),
            }
        )
        registry[movie_id] = entry
        _save_registry(registry)
        status = pair_status(movie_id)
        LOGGER.info(
            "upload %s: %s (%d files, %.1f MiB) ready=%s",
            kind, movie_id, n_files, n_bytes / 1024**2, status["ready"],
        )
        return {
            "movie_id": movie_id,
            "requested_name": base,
            "renamed": movie_id != base,
            "kind": kind,
            "n_files": n_files,
            "bytes": n_bytes,
            "validation": validation,
            "pair": status,
        }
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def delete_movie(movie_id: str) -> Dict[str, Any]:
    """Delete an *uploaded* movie only. Vault data is never touched."""
    registry = load_registry()
    entry = registry.get(movie_id)
    if entry is None or entry.get("source") != "upload":
        raise ValueError(
            f"Refusing to delete {movie_id!r}: only uploaded movies can be deleted"
        )
    removed = []
    for kind in ("zarr", "geff"):
        target = LOCAL_DATA_ROOT / f"{movie_id}.{kind}"
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            removed.append(target.name)
    registry.pop(movie_id, None)
    _save_registry(registry)
    return {"movie_id": movie_id, "removed": removed}


def train_contents() -> Dict[str, Any]:
    """Real listing of the server train directory (diagnostics, fast)."""
    _ensure_dirs()
    entries = []
    for p in sorted(LOCAL_DATA_ROOT.iterdir(), key=lambda x: x.name.lower()):
        if p.name.startswith("."):
            continue
        n_children = 0
        if p.is_dir():
            try:
                n_children = sum(1 for _ in p.iterdir())
            except OSError:
                n_children = -1
        entries.append({
            "name": p.name,
            "is_dir": p.is_dir(),
            "has_zarr_json": (p / "zarr.json").exists() if p.is_dir() else False,
            "n_children": n_children,
        })
    return {"root": str(LOCAL_DATA_ROOT), "entries": entries}
