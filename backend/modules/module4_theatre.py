"""
Module 4 — Tracking Theatre data bundles (Review-2 UI support).

Builds ONE JSON bundle per movie for the Tracking Theatre frontend:
cell positions, predicted matches with confidences, group-cost breakdowns,
fitted weights, and GAT-impact stats. Everything is read from real
artifacts on disk (Module 2/3/4 outputs) — nothing is invented here.

Sources:
  - "fitted": artifacts/tracks/{movie}/associations.npz  (pipeline run)
  - "custom": artifacts/tracks/{movie}/associations_custom.npz (UI mixer run)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from config import FEATURE_DIR, TRACK_DIR, TRACK_GROUP_NAMES


def associations_path(movie_id: str, source: str = "fitted") -> Path:
    name = "associations.npz" if source == "fitted" else "associations_custom.npz"
    return Path(TRACK_DIR) / movie_id / name


def summary_path(movie_id: str, source: str = "fitted") -> Path:
    name = ("tracking_half_summary.json" if source == "fitted"
            else "tracking_custom_summary.json")
    return Path(TRACK_DIR) / movie_id / name


def _cos_sim_matrix_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return (an * bn).sum(axis=1)


def build_theatre_bundle(movie_id: str, source: str = "fitted") -> Dict[str, Any]:
    m2_path = Path(FEATURE_DIR) / movie_id / "fused_features.npz"
    m3_path = Path(FEATURE_DIR) / movie_id / "context_enhanced_features.npz"
    if not m2_path.exists():
        raise FileNotFoundError(
            f"Module-2 features missing for '{movie_id}'. Run the pipeline first.")
    if not m3_path.exists():
        raise FileNotFoundError(
            f"Module-3 features missing for '{movie_id}'. Run the pipeline first.")
    apath = associations_path(movie_id, source)
    if not apath.exists():
        if source == "custom":
            raise FileNotFoundError(
                f"No custom mixer run yet for '{movie_id}'. Move the sliders and Remix.")
        raise FileNotFoundError(
            f"No tracking yet for '{movie_id}'. Run Module 4 first.")

    m2d = np.load(m2_path)
    m3d = np.load(m3_path)
    node_ids = np.asarray(m2d["node_ids"], dtype=np.int64)
    times = np.asarray(m2d["times"], dtype=np.int64)
    pos = np.asarray(m2d["positions_zyx"], dtype=np.float32)  # voxels: z,y,x
    appearance = np.asarray(m2d["appearance"], dtype=np.float32)
    ctx_ids = np.asarray(m3d["node_ids"], dtype=np.int64)
    ctx = np.asarray(m3d["context_enhanced_features"], dtype=np.float32)
    if not np.array_equal(node_ids, ctx_ids):
        order = np.asarray([{int(n): i for i, n in enumerate(ctx_ids)}[int(n)]
                            for n in node_ids])
        ctx = ctx[order]

    a = np.load(apath, allow_pickle=True)
    src = np.asarray(a["src_node"]).tolist()
    dst = np.asarray(a["dst_node"]).tolist()
    t_src = np.asarray(a["t_src"]).tolist()
    t_dst = np.asarray(a["t_dst"]).tolist()
    cost = np.asarray(a["cost"]).tolist()
    conf = np.asarray(a["confidence"]).tolist()
    gc = np.asarray(a["group_costs"], dtype=np.float32)

    index_of = {int(n): i for i, n in enumerate(node_ids)}
    matches = []
    app_sims, ctx_sims = [], []
    for k in range(len(src)):
        si = index_of.get(int(src[k]))
        di = index_of.get(int(dst[k]))
        if si is not None and di is not None:
            app_sims.append(float(_cos_sim_matrix_rows(
                appearance[si:si + 1], appearance[di:di + 1])[0]))
            ctx_sims.append(float(_cos_sim_matrix_rows(ctx[si:si + 1], ctx[di:di + 1])[0]))
        matches.append({
            "src": int(src[k]), "dst": int(dst[k]),
            "t_src": int(t_src[k]), "t_dst": int(t_dst[k]),
            "cost": round(float(cost[k]), 4),
            "confidence": round(float(conf[k]), 4),
            "group_costs": [round(float(v), 4) for v in gc[k].tolist()],
        })

    spath = summary_path(movie_id, source)
    summary = json.loads(spath.read_text(encoding="utf-8")) if spath.exists() else {}

    cells = [{"id": int(n), "t": int(t),
              "x": round(float(p[2]), 2), "y": round(float(p[1]), 2),
              "z": round(float(p[0]), 2)}
             for n, t, p in zip(node_ids.tolist(), times.tolist(), pos.tolist())]
    frames = sorted(set(int(t) for t in times.tolist()))
    return {
        "movie_id": movie_id,
        "source": source,
        "frames": frames,
        "bounds": {"x_max": round(float(pos[:, 2].max()), 1) if len(pos) else 256.0,
                   "y_max": round(float(pos[:, 1].max()), 1) if len(pos) else 256.0,
                   "z_max": round(float(pos[:, 0].max()), 1) if len(pos) else 64.0},
        "cells": cells,
        "matches": matches,
        "group_names": list(TRACK_GROUP_NAMES),
        "weights": summary.get("group_weights", {}),
        "weight_source": summary.get("group_weight_source", "unknown"),
        "mean_confidence": summary.get("mean_confidence", 0.0),
        "n_associations": len(matches),
        "per_frame_pair": summary.get("per_frame_pair", []),
        "gat_impact": {
            "n_matched_pairs": len(app_sims),
            "mean_cosine_appearance": round(float(np.mean(app_sims)), 4) if app_sims else 0.0,
            "mean_cosine_gat_context": round(float(np.mean(ctx_sims)), 4) if ctx_sims else 0.0,
        },
    }
