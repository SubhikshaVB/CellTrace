"""
GEFF graph annotation loaders for CellTrace Module 2 (and Module 0 cell centers).

The Biohub challenge ships cell lineage / detection graphs as `.geff` artifacts.
This loader tolerates several on-disk layouts commonly seen in challenge exports:
  - directory stores with nodes/edges arrays
  - JSON / JSON-lines graphs
  - NPZ bundles with positions and track ids
  - Zarr-backed geff folders
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
class CellNode:
    node_id: int
    track_id: int
    t: int
    z: float
    y: float
    x: float
    parent_id: Optional[int] = None
    attrs: Dict[str, Any] = field(default_factory=dict)

    @property
    def zyx(self) -> Tuple[float, float, float]:
        return (float(self.z), float(self.y), float(self.x))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "track_id": self.track_id,
            "t": self.t,
            "z": self.z,
            "y": self.y,
            "x": self.x,
            "parent_id": self.parent_id,
            "attrs": self.attrs,
        }


@dataclass
class GeffGraph:
    path: Path
    nodes: List[CellNode]
    edges: List[Tuple[int, int]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def nodes_at_time(self, t: int) -> List[CellNode]:
        return [n for n in self.nodes if int(n.t) == int(t)]

    def track_ids(self) -> List[int]:
        return sorted({int(n.track_id) for n in self.nodes})

    def time_range(self) -> Tuple[int, int]:
        if not self.nodes:
            return (0, 0)
        ts = [int(n.t) for n in self.nodes]
        return (min(ts), max(ts))

    def as_dict(self) -> Dict[str, Any]:
        t0, t1 = self.time_range()
        return {
            "path": str(self.path),
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "n_tracks": len(self.track_ids()),
            "t_min": t0,
            "t_max": t1,
            "meta": self.meta,
        }


class GeffIOError(RuntimeError):
    pass


def find_geff_files(movie_dir: Path) -> List[Path]:
    movie_dir = Path(movie_dir)
    if not movie_dir.exists():
        return []
    hits: List[Path] = []
    for p in movie_dir.rglob("*"):
        name = p.name.lower()
        if p.is_file() and (name.endswith(".geff") or name.endswith(".geff.json")):
            hits.append(p)
        elif p.is_dir() and (name.endswith(".geff") or name == "geff"):
            hits.append(p)
    return sorted(set(hits), key=lambda x: str(x).lower())


def _load_json_graph(path: Path) -> GeffGraph:
    raw = json.loads(path.read_text(encoding="utf-8"))
    nodes_raw = raw.get("nodes") or raw.get("cells") or raw.get("detections") or []
    edges_raw = raw.get("edges") or raw.get("links") or []
    nodes: List[CellNode] = []
    for i, item in enumerate(nodes_raw):
        if isinstance(item, dict):
            nid = int(item.get("id", item.get("node_id", i)))
            tid = int(item.get("track_id", item.get("track", item.get("label", nid))))
            t = int(item.get("t", item.get("time", item.get("frame", 0))))
            z = float(item.get("z", item.get("pos_z", 0.0)))
            y = float(item.get("y", item.get("pos_y", 0.0)))
            x = float(item.get("x", item.get("pos_x", 0.0)))
            parent = item.get("parent_id", item.get("parent"))
            parent_id = int(parent) if parent is not None else None
            nodes.append(
                CellNode(
                    node_id=nid,
                    track_id=tid,
                    t=t,
                    z=z,
                    y=y,
                    x=x,
                    parent_id=parent_id,
                    attrs={k: v for k, v in item.items() if k not in {
                        "id", "node_id", "track_id", "track", "label", "t", "time",
                        "frame", "z", "y", "x", "pos_z", "pos_y", "pos_x", "parent_id", "parent",
                    }},
                )
            )
    edges: List[Tuple[int, int]] = []
    for e in edges_raw:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            edges.append((int(e[0]), int(e[1])))
        elif isinstance(e, dict):
            edges.append((int(e.get("source", e.get("u"))), int(e.get("target", e.get("v")))))
    return GeffGraph(path=path, nodes=nodes, edges=edges, meta={"format": "json"})


def _load_npz(path: Path) -> GeffGraph:
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)

    def col(*names: str) -> Optional[np.ndarray]:
        for n in names:
            if n in keys:
                return np.asarray(data[n])
        return None

    pos = col("positions", "pos", "coords", "xyz", "zyx")
    t = col("t", "time", "frame", "frames")
    track = col("track_id", "track", "tracks", "label", "labels")
    node_id = col("node_id", "id", "ids")
    parent = col("parent_id", "parent", "parents")

    if pos is None:
        raise GeffIOError(f"NPZ GEFF missing positions: {path}")

    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] not in (3, 4):
        raise GeffIOError(f"Unexpected position array shape {pos.shape} in {path}")

    n = pos.shape[0]
    if pos.shape[1] == 4:
        # t,z,y,x
        if t is None:
            t = pos[:, 0]
            pos = pos[:, 1:4]
        else:
            pos = pos[:, -3:]

    if t is None:
        t = np.zeros(n, dtype=np.int64)
    if track is None:
        track = np.arange(n, dtype=np.int64)
    if node_id is None:
        node_id = np.arange(n, dtype=np.int64)

    nodes: List[CellNode] = []
    for i in range(n):
        pid = None
        if parent is not None and i < len(parent):
            raw_p = parent[i]
            if raw_p is not None and not (isinstance(raw_p, float) and np.isnan(raw_p)):
                pid = int(raw_p)
        zyx = pos[i]
        # Accept either zyx or xyz — heuristically treat as z,y,x (challenge)
        nodes.append(
            CellNode(
                node_id=int(node_id[i]),
                track_id=int(track[i]),
                t=int(t[i]),
                z=float(zyx[0]),
                y=float(zyx[1]),
                x=float(zyx[2]),
                parent_id=pid,
            )
        )
    edges: List[Tuple[int, int]] = []
    if "edges" in keys:
        e = np.asarray(data["edges"])
        for row in e:
            edges.append((int(row[0]), int(row[1])))
    return GeffGraph(path=path, nodes=nodes, edges=edges, meta={"format": "npz", "keys": sorted(keys)})


# def _load_zarr_geff(path: Path) -> GeffGraph:
#     if zarr is None:
#         raise GeffIOError("zarr is required to read zarr-backed .geff stores")
#     root = zarr.open(str(path), mode="r")

#     def find_arr(names: Sequence[str]):
#         if hasattr(root, "keys"):
#             for n in names:
#                 if n in root:
#                     return np.asarray(root[n])
#             # nested search
#             for key in root.keys():
#                 child = root[key]
#                 if hasattr(child, "shape"):
#                     if key in names:
#                         return np.asarray(child)
#         elif hasattr(root, "shape"):
#             return np.asarray(root)
#         return None

#     pos = find_arr(["positions", "pos", "coords", "zyx", "nodes/pos"])
#     t = find_arr(["t", "time", "frame", "nodes/t"])
#     track = find_arr(["track_id", "track", "label", "nodes/track_id"])
#     node_id = find_arr(["node_id", "id", "nodes/id"])

#     # Fallback: concatenate columns from a table-like array
#     if pos is None:
#         raise GeffIOError(f"Could not locate positions in geff zarr: {path}")

#     pos = np.asarray(pos, dtype=np.float64)
#     n = pos.shape[0]
#     if pos.ndim == 2 and pos.shape[1] >= 3:
#         zyx = pos[:, -3:]
#     else:
#         raise GeffIOError(f"Bad position shape {pos.shape}")

#     if t is None:
#         t = np.zeros(n, dtype=np.int64)
#     if track is None:
#         track = np.arange(n)
#     if node_id is None:
#         node_id = np.arange(n)

#     nodes = [
#         CellNode(
#             node_id=int(node_id[i]),
#             track_id=int(track[i]),
#             t=int(t[i]),
#             z=float(zyx[i, 0]),
#             y=float(zyx[i, 1]),
#             x=float(zyx[i, 2]),
#         )
#         for i in range(n)
#     ]
#     return GeffGraph(path=path, nodes=nodes, edges=[], meta={"format": "zarr-geff"})


def _load_zarr_geff(path: Path) -> GeffGraph:
    if zarr is None:
        raise GeffIOError("zarr is required to read Zarr-backed GEFF files")

    root = zarr.open(str(path), mode="r")

    # BioHub GEFF coordinates are separate Zarr columns, not a positions matrix.
    def read_path(*names: str) -> Optional[np.ndarray]:
        for name in names:
            try:
                return np.asarray(root[name]).reshape(-1)
            except Exception:
                continue
        return None

    t_column = read_path("nodes/props/t/values")
    z_column = read_path("nodes/props/z/values")
    y_column = read_path("nodes/props/y/values")
    x_column = read_path("nodes/props/x/values")
    if all(column is not None for column in (t_column, z_column, y_column, x_column)):
        node_ids = read_path("nodes/ids")
        n = min(len(t_column), len(z_column), len(y_column), len(x_column))
        if node_ids is None:
            node_ids = np.arange(n, dtype=np.int64)
        nodes = [
            CellNode(int(node_ids[i]), int(node_ids[i]), int(t_column[i]), float(z_column[i]), float(y_column[i]), float(x_column[i]))
            for i in range(n)
        ]
        edges = []
        try:
            edge_ids = np.asarray(root["edges/ids"])
            if edge_ids.ndim == 2 and edge_ids.shape[1] >= 2:
                edges = [(int(row[0]), int(row[1])) for row in edge_ids]
        except Exception:
            pass
        return GeffGraph(path=path, nodes=nodes, edges=edges, meta={"format": "biohub-zarr-geff"})

    def read_first(*paths: str) -> Optional[np.ndarray]:
        for item_path in paths:
            try:
                return np.asarray(root[item_path])
            except Exception:
                pass
        return None

    # BioHub GEFF hierarchy:
    # nodes/ids
    # nodes/props/t/values
    # nodes/props/z/values
    # nodes/props/y/values
    # nodes/props/x/values
    node_ids = read_first("nodes/ids", "node_id", "ids")
    times = read_first("nodes/props/t/values", "nodes/t", "t", "time")
    zs = read_first("nodes/props/z/values", "nodes/z", "z")
    ys = read_first("nodes/props/y/values", "nodes/y", "y")
    xs = read_first("nodes/props/x/values", "nodes/x", "x")

    if any(value is None for value in (times, zs, ys, xs)):
        raise GeffIOError(
            "GEFF is missing one or more coordinate arrays: t, z, y, x"
        )

    n = min(len(times), len(zs), len(ys), len(xs))

    if node_ids is None:
        node_ids = np.arange(n, dtype=np.int64)
    else:
        node_ids = np.asarray(node_ids).reshape(-1)[:n]

    nodes = [
        CellNode(
            node_id=int(node_ids[i]),
            # The dataset does not expose separate tracks here;
            # use node ID until lineage handling is added in Module 4.
            track_id=int(node_ids[i]),
            t=int(times[i]),
            z=float(zs[i]),
            y=float(ys[i]),
            x=float(xs[i]),
        )
        for i in range(n)
    ]

    edges: List[Tuple[int, int]] = []
    edge_ids = read_first("edges/ids", "edges")
    if edge_ids is not None:
        edge_ids = np.asarray(edge_ids)
        if edge_ids.ndim == 2 and edge_ids.shape[1] >= 2:
            edges = [
                (int(row[0]), int(row[1]))
                for row in edge_ids
            ]

    return GeffGraph(
        path=path,
        nodes=nodes,
        edges=edges,
        meta={
            "format": "biohub-zarr-geff",
            "n_nodes": len(nodes),
            "n_edges": len(edges),
        },
    )

def load_geff(path: Path) -> GeffGraph:
    path = Path(path)
    if not path.exists():
        raise GeffIOError(f"GEFF path does not exist: {path}")

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".json" or path.name.lower().endswith(".geff.json"):
            return _load_json_graph(path)
        if suffix == ".npz":
            return _load_npz(path)
        if suffix == ".geff":
            # Could be JSON text saved as .geff
            try:
                return _load_json_graph(path)
            except Exception:
                try:
                    return _load_npz(path)
                except Exception as exc:  # noqa: BLE001
                    raise GeffIOError(f"Unrecognized .geff file format: {path} ({exc})") from exc
        raise GeffIOError(f"Unsupported GEFF file type: {path}")

    # Directory store
    # A BioHub .geff directory is a Zarr store. Its zarr.json metadata is not
    # annotation JSON, so it must be read as Zarr before looking for JSON files.
    if (path / "zarr.json").exists():
        return _load_zarr_geff(path)
    json_candidates = list(path.glob("*.json")) + list(path.glob("**/*.json"))
    for jc in json_candidates[:5]:
        try:
            return _load_json_graph(jc)
        except Exception:
            continue
    npz_candidates = list(path.glob("*.npz"))
    for nc in npz_candidates[:5]:
        try:
            return _load_npz(nc)
        except Exception:
            continue
    try:
        return _load_zarr_geff(path)
    except Exception as exc:  # noqa: BLE001
        raise GeffIOError(f"Failed to load GEFF directory {path}: {exc}") from exc


def positions_by_time(graph: GeffGraph) -> Dict[int, np.ndarray]:
    buckets: Dict[int, List[Tuple[float, float, float]]] = {}
    for n in graph.nodes:
        buckets.setdefault(int(n.t), []).append(n.zyx)
    return {t: np.asarray(v, dtype=np.float64) for t, v in buckets.items()}


def track_polyline(graph: GeffGraph, track_id: int) -> np.ndarray:
    pts = [n for n in graph.nodes if int(n.track_id) == int(track_id)]
    pts.sort(key=lambda n: int(n.t))
    if not pts:
        return np.zeros((0, 4), dtype=np.float64)
    return np.asarray([[n.t, n.z, n.y, n.x] for n in pts], dtype=np.float64)
