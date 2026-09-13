"""
Zarr volume discovery and lazy slice / crop I/O for CellTrace Module 0.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import zarr
except ImportError:  # pragma: no cover
    zarr = None  # type: ignore


@dataclass
class ZarrVolumeInfo:
    path: Path
    array_key: str
    shape: Tuple[int, ...]
    dtype: str
    chunks: Optional[Tuple[int, ...]]
    attrs: Dict[str, Any] = field(default_factory=dict)
    axis_labels: Tuple[str, ...] = ()

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "array_key": self.array_key,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "chunks": list(self.chunks) if self.chunks is not None else None,
            "attrs": self.attrs,
            "axis_labels": list(self.axis_labels),
            "ndim": self.ndim,
        }


class ZarrIOError(RuntimeError):
    pass


def _require_zarr() -> None:
    if zarr is None:
        raise ZarrIOError(
            "The 'zarr' package is not installed. Run: pip install -r requirements.txt"
        )


def find_zarr_stores(movie_dir: Path) -> List[Path]:
    movie_dir = Path(movie_dir)
    if not movie_dir.exists():
        return []
    hits: List[Path] = []
    # Directory stores ending in .zarr
    for p in movie_dir.rglob("*"):
        if p.is_dir() and p.suffix == ".zarr":
            hits.append(p)
        elif p.is_file() and p.name in (".zarray", "zarr.json"):
            # parent may be an array; climb to store root
            parent = p.parent
            if parent.suffix == ".zarr" and parent not in hits:
                hits.append(parent)
    # Also accept a movie folder that IS a zarr store
    if (movie_dir / ".zgroup").exists() or (movie_dir / "zarr.json").exists():
        if movie_dir not in hits:
            hits.append(movie_dir)
    return sorted(set(hits), key=lambda x: str(x).lower())


def _open_group_or_array(path: Path):
    _require_zarr()
    try:
        return zarr.open(str(path), mode="r")
    except Exception as exc:  # noqa: BLE001
        raise ZarrIOError(f"Failed to open zarr at {path}: {exc}") from exc


def _iter_arrays(node, prefix: str = "") -> List[Tuple[str, Any]]:
    arrays: List[Tuple[str, Any]] = []
    if hasattr(node, "shape") and hasattr(node, "dtype") and not hasattr(node, "keys"):
        arrays.append((prefix or "/", node))
        return arrays
    if hasattr(node, "keys"):
        for key in node.keys():
            child = node[key]
            child_prefix = f"{prefix}/{key}" if prefix else key
            if hasattr(child, "shape") and hasattr(child, "dtype") and not hasattr(child, "keys"):
                arrays.append((child_prefix, child))
            else:
                arrays.extend(_iter_arrays(child, child_prefix))
    return arrays


def inspect_zarr(path: Path) -> ZarrVolumeInfo:
    path = Path(path)
    root = _open_group_or_array(path)
    arrays = _iter_arrays(root)
    if not arrays:
        raise ZarrIOError(f"No arrays found in zarr store: {path}")

    # Prefer largest ND array (likely the microscopy volume)
    def score(item: Tuple[str, Any]) -> Tuple[int, int]:
        arr = item[1]
        return (len(getattr(arr, "shape", ())), int(np.prod(arr.shape)))

    array_key, arr = max(arrays, key=score)
    attrs = {}
    try:
        attrs = dict(arr.attrs)
    except Exception:  # noqa: BLE001
        attrs = {}
    axis_labels: Tuple[str, ...] = ()
    for k in ("axes", "axis_names", "dimension_names"):
        if k in attrs:
            raw = attrs[k]
            if isinstance(raw, (list, tuple)):
                axis_labels = tuple(str(x) for x in raw)
            break
    chunks = tuple(arr.chunks) if getattr(arr, "chunks", None) is not None else None
    return ZarrVolumeInfo(
        path=path,
        array_key=array_key,
        shape=tuple(int(s) for s in arr.shape),
        dtype=str(arr.dtype),
        chunks=chunks,
        attrs=attrs,
        axis_labels=axis_labels,
    )


def infer_tzyx_axes(shape: Sequence[int], axis_labels: Sequence[str] = ()) -> Dict[str, int]:
    """
    Map logical axes T,Z,Y,X onto array dimensions.
    Supports common layouts: (T,Z,Y,X), (T,C,Z,Y,X), (Z,Y,X), (T,Y,X).
    """
    labels = [str(a).lower() for a in axis_labels]
    mapping: Dict[str, int] = {}
    if labels:
        for i, lab in enumerate(labels):
            if lab in ("t", "time", "frame"):
                mapping["t"] = i
            elif lab in ("z", "depth"):
                mapping["z"] = i
            elif lab in ("y", "row"):
                mapping["y"] = i
            elif lab in ("x", "col", "column"):
                mapping["x"] = i
            elif lab in ("c", "channel"):
                mapping["c"] = i
        if {"z", "y", "x"}.issubset(mapping.keys()) or {"y", "x"}.issubset(mapping.keys()):
            return mapping

    ndim = len(shape)
    if ndim == 5:
        # Assume T,C,Z,Y,X
        return {"t": 0, "c": 1, "z": 2, "y": 3, "x": 4}
    if ndim == 4:
        # Assume T,Z,Y,X (challenge default)
        return {"t": 0, "z": 1, "y": 2, "x": 3}
    if ndim == 3:
        return {"z": 0, "y": 1, "x": 2}
    if ndim == 2:
        return {"y": 0, "x": 1}
    raise ZarrIOError(f"Unsupported array ndim={ndim}, shape={shape}")


def open_array(info: ZarrVolumeInfo):
    root = _open_group_or_array(info.path)
    if info.array_key in ("", "/"):
        return root
    node = root
    for part in info.array_key.strip("/").split("/"):
        if not part:
            continue
        node = node[part]
    return node


def read_slice_yx(
    info: ZarrVolumeInfo,
    t: int = 0,
    z: int = 0,
    channel: int = 0,
) -> np.ndarray:
    arr = open_array(info)
    axes = infer_tzyx_axes(info.shape, info.axis_labels)
    index: List[Any] = [slice(None)] * info.ndim

    def _clip(axis: str, val: int) -> int:
        if axis not in axes:
            return val
        dim = axes[axis]
        return int(np.clip(val, 0, info.shape[dim] - 1))

    if "t" in axes:
        index[axes["t"]] = _clip("t", t)
    if "z" in axes:
        index[axes["z"]] = _clip("z", z)
    if "c" in axes:
        index[axes["c"]] = _clip("c", channel)
    data = np.asarray(arr[tuple(index)])
    data = np.squeeze(data)
    if data.ndim != 2:
        # Fallback: take first 2 trailing dims
        while data.ndim > 2:
            data = data[0]
    return np.asarray(data, dtype=np.float32)


def read_crop_zyx(
    info: ZarrVolumeInfo,
    t: int,
    z0: int,
    z1: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    channel: int = 0,
) -> np.ndarray:
    arr = open_array(info)
    axes = infer_tzyx_axes(info.shape, info.axis_labels)
    index: List[Any] = [0] * info.ndim

    def set_axis(name: str, value: Any) -> None:
        if name in axes:
            index[axes[name]] = value

    set_axis("t", int(np.clip(t, 0, info.shape[axes["t"]] - 1)) if "t" in axes else 0)
    set_axis("c", channel)
    if "z" in axes:
        zs, ze = sorted((z0, z1))
        zs = int(np.clip(zs, 0, info.shape[axes["z"]]))
        ze = int(np.clip(ze, 0, info.shape[axes["z"]]))
        if ze <= zs:
            ze = min(info.shape[axes["z"]], zs + 1)
        set_axis("z", slice(zs, ze))
    if "y" in axes:
        ys, ye = sorted((y0, y1))
        ys = int(np.clip(ys, 0, info.shape[axes["y"]]))
        ye = int(np.clip(ye, 0, info.shape[axes["y"]]))
        if ye <= ys:
            ye = min(info.shape[axes["y"]], ys + 1)
        set_axis("y", slice(ys, ye))
    if "x" in axes:
        xs, xe = sorted((x0, x1))
        xs = int(np.clip(xs, 0, info.shape[axes["x"]]))
        xe = int(np.clip(xe, 0, info.shape[axes["x"]]))
        if xe <= xs:
            xe = min(info.shape[axes["x"]], xs + 1)
        set_axis("x", slice(xs, xe))

    # For axes not set (rare), use full slice
    for i, v in enumerate(index):
        if isinstance(v, int) and i not in axes.values():
            index[i] = slice(None)

    crop = np.asarray(arr[tuple(index)], dtype=np.float32)
    crop = np.squeeze(crop)
    if crop.ndim == 2:
        crop = crop[None, ...]
    if crop.ndim != 3:
        raise ZarrIOError(f"Expected 3D crop ZYX, got shape {crop.shape}")
    return crop


def volume_bounds(info: ZarrVolumeInfo) -> Dict[str, int]:
    axes = infer_tzyx_axes(info.shape, info.axis_labels)
    out = {
        "t_max": int(info.shape[axes["t"]]) if "t" in axes else 1,
        "z_max": int(info.shape[axes["z"]]) if "z" in axes else 1,
        "y_max": int(info.shape[axes["y"]]) if "y" in axes else int(info.shape[-2]),
        "x_max": int(info.shape[axes["x"]]) if "x" in axes else int(info.shape[-1]),
        "c_max": int(info.shape[axes["c"]]) if "c" in axes else 1,
    }
    return out


def dump_hierarchy(path: Path, max_entries: int = 64) -> Dict[str, Any]:
    path = Path(path)
    root = _open_group_or_array(path)
    arrays = _iter_arrays(root)
    entries = []
    for key, arr in arrays[:max_entries]:
        entries.append(
            {
                "key": key,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "chunks": list(arr.chunks) if getattr(arr, "chunks", None) else None,
            }
        )
    return {"path": str(path), "n_arrays": len(arrays), "arrays": entries}


def write_meta_sidecar(info: ZarrVolumeInfo, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = info.as_dict()
    payload["bounds"] = volume_bounds(info)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
