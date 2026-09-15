"""
Module 4 — Confidence-Aware Tracking, FIRST HALF (Review 2).

Implements (per movie, consecutive frame pairs):
  1. Feature-group separation:
       appearance (256) / spatial (3) / kinematic (6) / neighborhood (3) / gat_context (128)
  2. Learnable group weighting: small MLP + softmax (weights sum to 1).
     Fitted on TRAIN movies only (BCE on GT-labelled candidate pairs);
     uniform prior when no fitted weights exist.
  3. Temporal candidate generation: consecutive (t, t+1) pairs, top-K + max-dist gate.
  4. Cost matrix from weighted group distances.
  5. Hungarian one-to-one assignment per frame pair.
  6. Per-association confidence from match quality + runner-up margin.

STOPPING POINT (Review 2): associations + costs + confidences are saved.
Deferred to Review 3: confidence thresholding + trajectory stitching,
division handling, MOTA/MOTP evaluation.

Ground-truth policy: run() NEVER touches GEFF edges. fit_weights() uses GT
labels on train movies only (supervised fitting — legitimate training).
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from config import (
    EMBEDDING_DIR,
    FEATURE_DIR,
    LOCAL_DATA_ROOT,
    SPACING_ZYX_UM,
    TRACK_CONFIDENCE_TEMP,
    TRACK_GROUP_NAMES,
    TRACK_MAX_DIST_UM,
    TRACK_TOP_K_CANDIDATES,
    TRACK_WEIGHT_EPOCHS,
    TRACK_WEIGHT_LR,
    TRACK_WEIGHT_MLP_HIDDEN,
    TRACK_WEIGHT_SEED,
    TRACK_DIR,
)
from utils.io_geff import load_geff

GATED_COST = 2.0  # penalty cost for pairs beyond the geometric gate


# ----------------------------------------------------------------------------
# Feature groups
# ----------------------------------------------------------------------------
def load_tracking_bundle(movie_id: str) -> Dict[str, Any]:
    """Load M2 groups + M3 context features, aligned by node_id. No GT used."""
    feat_dir = Path(FEATURE_DIR) / movie_id
    m2_path = feat_dir / "fused_features.npz"
    m3_path = feat_dir / "context_enhanced_features.npz"
    if not m2_path.exists():
        raise FileNotFoundError(f"Run Module 2 for '{movie_id}' first: {m2_path} missing")
    if not m3_path.exists():
        raise FileNotFoundError(f"Run Module 3 for '{movie_id}' first: {m3_path} missing")
    m2d = np.load(m2_path)
    m3d = np.load(m3_path)
    node_ids = np.asarray(m2d["node_ids"], dtype=np.int64)
    ctx_ids = np.asarray(m3d["node_ids"], dtype=np.int64)
    ctx_feat = np.asarray(m3d["context_enhanced_features"], dtype=np.float32)
    if not np.array_equal(node_ids, ctx_ids):
        # Align M3 rows onto M2 order via node_id.
        pos = {int(n): i for i, n in enumerate(ctx_ids)}
        order = np.asarray([pos[int(n)] for n in node_ids])
        ctx_feat = ctx_feat[order]
    spacing = np.asarray(SPACING_ZYX_UM, dtype=np.float32)
    positions_um = np.asarray(m2d["positions_zyx"], dtype=np.float32) * spacing
    context10 = np.asarray(m2d["context"], dtype=np.float32)  # M2 standardized context
    return {
        "movie_id": movie_id,
        "node_ids": node_ids,
        "times": np.asarray(m2d["times"], dtype=np.int64),
        "appearance": np.asarray(m2d["appearance"], dtype=np.float32),
        "spatial_um": positions_um,
        # M2 context columns: [vel*3, speed, gap, incoming, ncount, mean_d, min_d]
        "kinematic": context10[:, 0:6],
        "neighborhood": context10[:, 6:9],
        "gat_context": ctx_feat,
    }


def _cos_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise (1 - cosine)/2 in [0, 1]. a: (N,D), b: (M,D)."""
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return np.clip((1.0 - an @ bn.T) / 2.0, 0.0, 1.0)


def _euc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d2 = (np.square(a).sum(1, keepdims=True)
          + np.square(b).sum(1)[None, :] - 2.0 * a @ b.T)
    return np.sqrt(np.maximum(d2, 0.0))


def _zscore_rows(v: np.ndarray) -> np.ndarray:
    mu = v.mean(axis=0, keepdims=True)
    sd = v.std(axis=0, keepdims=True)
    return ((v - mu) / np.maximum(sd, 1e-6)).astype(np.float32)


def group_cost_matrices(bundle: Dict[str, Any], src_idx: np.ndarray,
                        dst_idx: np.ndarray) -> Dict[str, np.ndarray]:
    """Five (Ns, Nd) cost matrices, each in [0, 1]."""
    d_app = _cos_dist(bundle["appearance"][src_idx], bundle["appearance"][dst_idx])
    d_spa = np.clip(_euc(bundle["spatial_um"][src_idx], bundle["spatial_um"][dst_idx])
                    / TRACK_MAX_DIST_UM, 0.0, 1.0)
    zk = _zscore_rows(bundle["kinematic"])
    zn = _zscore_rows(bundle["neighborhood"])
    d_kin = np.clip(np.abs(zk[src_idx, None, :] - zk[dst_idx][None, :, :]).mean(-1) / 4.0,
                    0.0, 1.0)
    d_nb = np.clip(np.abs(zn[src_idx, None, :] - zn[dst_idx][None, :, :]).mean(-1) / 4.0,
                   0.0, 1.0)
    d_ctx = _cos_dist(bundle["gat_context"][src_idx], bundle["gat_context"][dst_idx])
    return {"appearance": d_app, "spatial": d_spa, "kinematic": d_kin,
            "neighborhood": d_nb, "gat_context": d_ctx}


# ----------------------------------------------------------------------------
# Learnable group weighting: small MLP + softmax
# ----------------------------------------------------------------------------
class GroupWeightMLP(nn.Module):
    """Global descriptor (mean+std of group costs on candidates, 10-D)
    -> hidden -> 5 logits -> softmax. Plus a learnable match bias."""

    def __init__(self, hidden: int = TRACK_WEIGHT_MLP_HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 5))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, desc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.softmax(self.net(desc), dim=-1), self.bias


def _global_descriptor(costs: Dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray([[costs[g].mean(), costs[g].std()] for g in TRACK_GROUP_NAMES],
                       dtype=np.float32).reshape(-1)


def weights_path() -> Path:
    return Path(TRACK_DIR) / "group_weight_mlp.pt"


def load_weights() -> Tuple[Dict[str, float], str]:
    """Returns (weights dict, source). Uniform prior if nothing fitted."""
    wp = weights_path()
    if wp.exists():
        try:
            ckpt = torch.load(wp, map_location="cpu", weights_only=True)
            w = np.asarray(ckpt["weights"], dtype=float)
            w = w / w.sum()
            return {g: round(float(v), 4) for g, v in zip(TRACK_GROUP_NAMES, w)}, \
                f"fitted:{ckpt.get('train_movies', '?')}"
        except Exception:
            pass
    u = round(1.0 / len(TRACK_GROUP_NAMES), 4)
    return {g: u for g in TRACK_GROUP_NAMES}, "uniform-prior"


def fit_weights(train_movie_ids: Sequence[str],
                epochs: int = TRACK_WEIGHT_EPOCHS,
                lr: float = TRACK_WEIGHT_LR) -> Dict[str, Any]:
    """Supervised fitting of the group-weight MLP on TRAIN movies only.

    Builds real candidate pairs, labels them with GEFF edges (GT), and
    optimizes BCE: P(match) = sigmoid(-(w . d - b)/temp).
    """
    t0 = time.time()
    torch.manual_seed(TRACK_WEIGHT_SEED)
    np.random.seed(TRACK_WEIGHT_SEED % (2 ** 32 - 1))
    if not train_movie_ids:
        raise ValueError("fit_weights needs >=1 train movie.")

    # Collect candidate group-costs + GT labels across train movies.
    stack_d, stack_y = [], []
    descs = []
    per_movie: Dict[str, Any] = {}
    for movie_id in train_movie_ids:
        bundle = load_tracking_bundle(movie_id)
        geff_path = Path(LOCAL_DATA_ROOT) / f"{movie_id}.geff"
        gt = set(load_geff(geff_path).edges) if geff_path.exists() else set()
        frames = sorted(set(int(t) for t in bundle["times"]))
        n_pos = n_tot = 0
        for t0f, t1f in zip(frames[:-1], frames[1:]):
            if t1f != t0f + 1:
                continue
            s_idx = np.where(bundle["times"] == t0f)[0]
            d_idx = np.where(bundle["times"] == t1f)[0]
            if len(s_idx) == 0 or len(d_idx) == 0:
                continue
            costs = group_cost_matrices(bundle, s_idx, d_idx)
            descs.append(_global_descriptor(costs))
            s_nodes = bundle["node_ids"][s_idx]
            d_nodes = bundle["node_ids"][d_idx]
            for i, sn in enumerate(s_nodes):
                # Top-K spatial pre-filter keeps fitting fast on big movies.
                order = np.argsort(costs["spatial"][i])[:TRACK_TOP_K_CANDIDATES]
                for j in order:
                    stack_d.append([costs[g][i, j] for g in TRACK_GROUP_NAMES])
                    y = 1.0 if (int(sn), int(d_nodes[j])) in gt else 0.0
                    stack_y.append(y)
                    n_tot += 1
                    n_pos += int(y)
        per_movie[movie_id] = {"candidates": n_tot, "positives": n_pos}
    if not stack_d or sum(stack_y) == 0:
        raise RuntimeError("No GT-labelled positive candidates; cannot fit weights.")
    D = torch.tensor(np.asarray(stack_d, dtype=np.float32))
    Y = torch.tensor(np.asarray(stack_y, dtype=np.float32))
    desc = torch.tensor(np.mean(descs, axis=0), dtype=torch.float32)

    mlp = GroupWeightMLP()
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    hist = []
    for ep in range(1, epochs + 1):
        mlp.train()
        w, b = mlp(desc)
        score = torch.sigmoid(-((D * w).sum(1) - b) / TRACK_CONFIDENCE_TEMP)
        loss = nn.functional.binary_cross_entropy(score, Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        hist.append({"epoch": ep, "loss": round(float(loss.detach()), 6)})
    mlp.eval()
    with torch.no_grad():
        w, b = mlp(desc)
        wv = w.detach().cpu().numpy()
    weights = {g: round(float(v), 4) for g, v in zip(TRACK_GROUP_NAMES, wv)}
    Path(TRACK_DIR).mkdir(parents=True, exist_ok=True)
    torch.save({"weights": torch.from_numpy(wv.astype(np.float32)), "bias": float(b.detach().cpu()),
                "train_movies": list(train_movie_ids), "history": hist,
                "per_movie": per_movie}, weights_path())
    return {"status": "complete", "train_movies": list(train_movie_ids),
            "weights": weights, "bias": round(float(b.detach().cpu()), 4),
            "n_candidates": int(len(Y)), "n_positives": int(Y.sum().item()),
            "final_loss": hist[-1]["loss"], "history": hist,
            "per_movie": per_movie, "elapsed_sec": round(time.time() - t0, 1),
            "checkpoint": str(weights_path())}


# ----------------------------------------------------------------------------
# Matching + confidence
# ----------------------------------------------------------------------------
def _match_frame_pair(bundle: Dict[str, Any], t0f: int, t1f: int,
                      w: np.ndarray) -> List[Dict[str, Any]]:
    s_idx = np.where(bundle["times"] == t0f)[0]
    d_idx = np.where(bundle["times"] == t1f)[0]
    if len(s_idx) == 0 or len(d_idx) == 0:
        return []
    costs = group_cost_matrices(bundle, s_idx, d_idx)
    stacked = np.stack([costs[g] for g in TRACK_GROUP_NAMES], axis=-1)  # (Ns,Nd,5)
    dist_um = _euc(bundle["spatial_um"][s_idx], bundle["spatial_um"][d_idx])
    gated = dist_um > TRACK_MAX_DIST_UM
    C = (stacked * w.reshape(1, 1, -1)).sum(-1)
    C = np.where(gated, GATED_COST, C)
    rows, cols = linear_sum_assignment(C)
    out = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        best = float(C[r, c])
        row_others = np.delete(C[r], c)
        runner = float(row_others.min()) if row_others.size else 1.0
        quality = 1.0 - min(best, 1.0)
        margin = float(np.clip(runner - best, 0.0, 1.0))
        conf = round(0.5 * quality + 0.5 * margin, 4)
        out.append({
            "src_node_id": int(bundle["node_ids"][s_idx[r]]),
            "dst_node_id": int(bundle["node_ids"][d_idx[c]]),
            "t_src": int(t0f), "t_dst": int(t1f),
            "cost": round(best, 4),
            "confidence": conf,
            "gated": bool(gated[r, c]),
            "group_costs": [round(float(v), 4) for v in stacked[r, c].tolist()],
        })
    return out


class TrackingEngine:
    def __init__(self) -> None:
        self.state: Dict[str, Any] = {"status": "idle", "progress": 0.0}

    def run(self, movie_id: str,
            group_weights: Optional[Dict[str, float]] = None,
            tag: str = "") -> Dict[str, Any]:
        """tag="custom" writes separate artifacts (UI mixer experiments,
        fitted run untouched)."""
        t0 = time.time()
        self.state = {"status": "running", "progress": 0.05,
                      "message": f"Loading tracking bundle for {movie_id}"}
        bundle = load_tracking_bundle(movie_id)
        if group_weights:
            weights = {g: float(group_weights.get(g, 0.0)) for g in TRACK_GROUP_NAMES}
            s = sum(weights.values())
            weights = {g: round(v / s if s > 0 else 1.0 / len(TRACK_GROUP_NAMES), 4)
                       for g, v in weights.items()}
            w_source = "custom-mixer"
        else:
            weights, w_source = load_weights()
        w = np.asarray([weights[g] for g in TRACK_GROUP_NAMES], dtype=np.float64)
        w = w / w.sum()
        suffix = "_custom" if tag == "custom" else ""

        frames = sorted(set(int(t) for t in bundle["times"]))
        pairs = [(a, b) for a, b in zip(frames[:-1], frames[1:]) if b == a + 1]
        associations: List[Dict[str, Any]] = []
        per_pair = []
        for k, (a, b) in enumerate(pairs):
            matches = _match_frame_pair(bundle, a, b, w)
            associations.extend(matches)
            per_pair.append({"t_src": a, "t_dst": b, "n_matches": len(matches),
                             "mean_confidence": round(float(np.mean(
                                 [m["confidence"] for m in matches])) if matches else 0.0, 4),
                             "n_gated": sum(1 for m in matches if m["gated"])})
            self.state["progress"] = 0.05 + 0.9 * (k + 1) / max(len(pairs), 1)

        out_dir = Path(TRACK_DIR) / movie_id
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / f"associations{suffix}.npz"
        np.savez_compressed(
            npz_path,
            src_node=np.asarray([m["src_node_id"] for m in associations], dtype=np.int64),
            dst_node=np.asarray([m["dst_node_id"] for m in associations], dtype=np.int64),
            t_src=np.asarray([m["t_src"] for m in associations], dtype=np.int64),
            t_dst=np.asarray([m["t_dst"] for m in associations], dtype=np.int64),
            cost=np.asarray([m["cost"] for m in associations], dtype=np.float32),
            confidence=np.asarray([m["confidence"] for m in associations], dtype=np.float32),
            group_costs=np.asarray([m["group_costs"] for m in associations],
                                   dtype=np.float32).reshape(-1, len(TRACK_GROUP_NAMES)),
            group_names=np.asarray(TRACK_GROUP_NAMES),
        )
        result = {
            "movie_id": movie_id, "status": "complete-half",
            "scope": ("matching+similarity+Hungarian+confidence. "
                      "Thresholding and trajectory stitching deferred to Review 3."),
            "n_frames": len(frames), "n_frame_pairs": len(pairs),
            "n_cells": int(len(bundle["node_ids"])),
            "n_associations": len(associations),
            "group_weights": weights, "group_weight_source": w_source,
            "mean_confidence": round(float(np.mean(
                [m["confidence"] for m in associations])) if associations else 0.0, 4),
            "n_gated_matches": sum(1 for m in associations if m["gated"]),
            "per_frame_pair": per_pair,
            "preview": associations[:8],
            "elapsed_sec": round(time.time() - t0, 1),
            "artifacts": {"associations": str(npz_path)},
        }
        (out_dir / f"tracking{suffix or '_half'}_summary.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        self.state = {"status": "complete-half", "progress": 1.0,
                      "message": "Tracking first-half complete", "result": result}
        return result


ENGINE = TrackingEngine()


def run_tracking(movie_id: str) -> Dict[str, Any]:
    return ENGINE.run(movie_id)


def module_status() -> Dict[str, Any]:
    return ENGINE.state


def safe_call(fn, *args, **kwargs) -> Dict[str, Any]:
    try:
        return {"ok": True, "result": fn(*args, **kwargs)}
    except Exception as exc:
        return {"ok": False, "error": str(exc),
                "traceback": traceback.format_exc()}
