"""
Guided Journey backend (Review-2 UI support).

Small, honest data endpoints for the narrative Journey view:
  - geff_cloud: raw GEFF annotation points (pre-pipeline ground truth)
  - artifact summaries: previously computed run summaries (loss histories…)
  - focus_attention: REAL trained-GAT attention weights for one cell's
    neighborhood, recomputed live from the saved checkpoint with the exact
    training-time math (no invented weights).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    ARTIFACTS_ROOT,
    EMBEDDING_DIR,
    FEATURE_DIR,
    LOCAL_DATA_ROOT,
    TRACK_DIR,
    spacing_dict,
)
from modules.module3_gat import ContrastiveGAT
from utils.io_geff import load_geff
from utils.spacing import DEFAULT_SPACING


# ----------------------------------------------------------------------------
# Raw GEFF cloud (Chapter 1 hero: what is .geff?)
# ----------------------------------------------------------------------------
def geff_cloud(movie_id: str, max_points: int = 3000) -> Dict[str, Any]:
    path = Path(LOCAL_DATA_ROOT) / f"{movie_id}.geff"
    graph = load_geff(path)  # raises GeffIOError with a clear message
    nodes = sorted(graph.nodes, key=lambda n: (int(n.t), int(n.node_id)))
    total = len(nodes)
    step = max(1, total // max(int(max_points), 1))
    sampled = nodes[::step][: int(max_points)]
    ts = [int(n.t) for n in nodes] or [0]
    return {
        "movie_id": movie_id,
        "format": graph.meta.get("format", "geff"),
        "n_total_nodes": total,
        "n_total_edges": len(graph.edges),
        "n_shown": len(sampled),
        "t_min": min(ts),
        "t_max": max(ts),
        "links": [[int(s), int(d)] for s, d in graph.edges],
        "spacing_um": spacing_dict(),
        "points": [{"id": int(n.node_id), "t": int(n.t),
                    "x": round(float(n.x), 2), "y": round(float(n.y), 2),
                    "z": round(float(n.z), 2)} for n in sampled],
    }


# ----------------------------------------------------------------------------
# Allow-listed artifact summaries (loss curves, weights, counts…)
# ----------------------------------------------------------------------------
_SUMMARIES = {
    "module2_summary": lambda m: Path(FEATURE_DIR) / m / "module2_fusion_summary.json",
    "module3_summary": lambda m: Path(FEATURE_DIR) / m / "module3_gat_summary.json",
    "tracking_summary": lambda m: Path(TRACK_DIR) / m / "tracking_half_summary.json",
    "embedding_meta": lambda m: Path(EMBEDDING_DIR) / m / "appearance_embeddings.json",
    "review_manifest": lambda m: Path(ARTIFACTS_ROOT) / "review2_manifest.json",
}


def read_summary(name: str, movie_id: str = "") -> Dict[str, Any]:
    if name not in _SUMMARIES:
        raise ValueError(f"Unknown summary '{name}'. Allowed: {sorted(_SUMMARIES)}")
    path = _SUMMARIES[name](movie_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Summary '{name}' not computed yet"
            + (f" for '{movie_id}'" if movie_id else "")
            + ". Run the pipeline first.")
    return {"name": name, "movie_id": movie_id,
            "data": json.loads(path.read_text(encoding="utf-8"))}


# ----------------------------------------------------------------------------
# Real trained-GAT attention for one focus cell (Chapter 4 lens)
# ----------------------------------------------------------------------------
@torch.no_grad()
def focus_attention(movie_id: str, node_id: int, k: int = 8) -> Dict[str, Any]:
    feat_dir = Path(FEATURE_DIR) / movie_id
    m2_path = feat_dir / "fused_features.npz"
    ckpt_path = feat_dir / "neighbour_aware_gat.pt"
    if not m2_path.exists():
        raise FileNotFoundError(f"Module-2 features missing for '{movie_id}'.")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Trained GAT missing for '{movie_id}'. Run Module 3.")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    config = ckpt.get("config", {}) or {}
    if config.get("type") not in ("ContrastiveGAT",):
        raise RuntimeError(
            "Saved GAT is a legacy self-supervised model without multi-head "
            "attention. Re-run Module 3 (contrastive) for this movie.")

    data = np.load(m2_path)
    node_ids = np.asarray(data["node_ids"], dtype=np.int64)
    times = np.asarray(data["times"], dtype=np.int64)
    positions = np.asarray(data["positions_zyx"], dtype=np.float32)
    fused = np.asarray(data["fused_features"], dtype=np.float32)
    where = np.where(node_ids == int(node_id))[0]
    if len(where) == 0:
        raise ValueError(f"Node {node_id} not found in '{movie_id}'.")
    focus = int(where[0])
    frame = int(times[focus])

    # Same-frame neighborhood: self + k nearest (training-time graph rule).
    frame_idx = np.where(times == frame)[0]
    spacing = np.asarray(DEFAULT_SPACING.as_tuple, dtype=np.float32)
    d = np.linalg.norm((positions[frame_idx] - positions[focus]) * spacing, axis=1)
    order = np.argsort(d, kind="stable")[: max(int(k), 0) + 1]
    nbr_global = frame_idx[order]

    sd = ckpt["model_state_dict"]
    _1, heads, hidden = sd["layer1.att_src"].shape
    out_dim = sd["layer2.lin.weight"].shape[0]
    model = ContrastiveGAT(int(fused.shape[1]), hidden_dim=int(hidden),
                           heads=int(heads), out_dim=int(out_dim))
    model.load_state_dict(sd)
    model.eval()
    layer = model.layer1
    heads, dim = layer.heads, layer.out_dim
    x = torch.from_numpy(fused[nbr_global])
    proj = layer.lin(x).view(len(nbr_global), heads, dim)  # (K+1, H, D)
    focus_local = int(np.where(nbr_global == focus)[0][0])
    a_src = (proj * layer.att_src).sum(-1)                       # (K+1, H)
    a_dst = (proj[focus_local] * layer.att_dst).sum(-1)          # (H,)
    e = F.leaky_relu(a_src + a_dst, negative_slope=0.2)  # (K+1, H): a_dst (1, H) broadcasts
    alpha = torch.softmax(e, dim=0).cpu().numpy()                # (K+1, H)

    neighbors = []
    for rank, (gi, dd) in enumerate(zip(nbr_global.tolist(), d[order].tolist())):
        ah = alpha[rank]
        neighbors.append({
            "node_id": int(node_ids[gi]),
            "is_self": bool(gi == focus),
            "distance_um": round(float(dd), 2),
            "alpha_mean": round(float(ah.mean()), 4),
            "alpha_heads": [round(float(v), 4) for v in ah.tolist()],
        })
    neighbors.sort(key=lambda r: r["alpha_mean"], reverse=True)
    return {"movie_id": movie_id, "focus_node_id": int(node_id), "frame": frame,
            "heads": int(heads),
            "note": ("Exact layer-1 attention of the saved contrastive GAT "
                     "(softmax over same-frame KNN + self)."),
            "neighbors": neighbors}
