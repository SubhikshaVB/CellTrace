"""
Module 3 — Neighbour-Aware Graph Attention Network (Review 2).

Inputs (per movie):
  - Module 2 fused features:  artifacts/features/{movie}/fused_features.npz
      {node_ids, times, positions_zyx, appearance, context, alpha, fused_features}
  - GEFF ground-truth temporal links (for contrastive supervision only)

What this module does:
  1. Rebuilds a per-frame KNN spatial graph (K configurable, same-frame only).
  2. Trains a sparse multi-head Graph Attention Network with a contrastive
     objective: same-cell pairs across consecutive frames are pulled together,
     non-matching pairs are pushed apart (margin loss).
  3. Encodes every cell into a 128-D context-aware embedding (L2-normalized).
  4. Falls back to self-supervised gated attention when a movie has no
     ground-truth links (e.g. test movies).

Movie-level discipline: use train_gat_multimovie() with disjoint train/val
movie lists. Single-movie run() splits *pairs* within that movie for
monitoring only and says so explicitly in its summary.

Artifacts (same names as Review 1, extended contents):
  - artifacts/features/{movie}/context_enhanced_features.npz
  - artifacts/features/{movie}/neighbour_aware_gat.pt
  - artifacts/features/{movie}/module3_gat_summary.json
"""
from __future__ import annotations

import json
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# index_reduce(amax) emits a PyTorch "beta API" notice; its behavior is pinned
# by tests/test_review2_math.py::test_gat_forward_normalized, so silence only
# that notice (not other warnings).
warnings.filterwarnings(
    "ignore", message=".*index_reduce.*beta.*", category=UserWarning
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    FEATURE_DIR,
    GAT_EPOCHS,
    GAT_HIDDEN_DIM,
    GAT_K_NEIGHBORS,
    GAT_LR,
    GAT_MARGIN,
    GAT_MAX_PAIRS_PER_MOVIE,
    GAT_NEG_PER_POS,
    GAT_NUM_HEADS,
    GAT_OUT_DIM,
    GAT_SEED,
    GAT_VAL_FRACTION,
    GAT_WEIGHT_DECAY,
    LOCAL_DATA_ROOT,
)
from modules import module2_spatiotemporal as m2
from utils.io_geff import load_geff
from utils.spacing import DEFAULT_SPACING, batch_distances_um


# ----------------------------------------------------------------------------
# Legacy Review-1 model (kept for compatibility + test-movie fallback)
# ----------------------------------------------------------------------------
class GatedGraphAttention(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key = nn.Linear(feature_dim, feature_dim, bias=False)
        self.value = nn.Linear(feature_dim, feature_dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid(),
        )
        self.decoder = nn.Linear(feature_dim, feature_dim)

    def forward(self, features: torch.Tensor, neighbour_mask: torch.Tensor):
        scale = float(features.shape[1]) ** -0.5
        q = self.query(features)
        k = self.key(features)
        v = self.value(features)
        scores = (q @ k.T) * scale
        scores = scores.masked_fill(~neighbour_mask, -1e9)
        attention = torch.softmax(scores, dim=1)
        neighbour_features = attention @ v
        gate = self.gate(torch.cat([features, neighbour_features], dim=1))
        context_enhanced = gate * features + (1.0 - gate) * neighbour_features
        reconstruction = self.decoder(context_enhanced)
        return context_enhanced, attention, gate, reconstruction


# ----------------------------------------------------------------------------
# Review-2 sparse multi-head GAT
# ----------------------------------------------------------------------------
class SparseGATLayer(nn.Module):
    """Single sparse graph-attention layer.

    Each node attends over itself + its K spatial neighbours (edge list).
    No N x N dense matrix is ever built, so large movies fit in memory.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4,
                 dropout: float = 0.0, concat: bool = True) -> None:
        super().__init__()
        self.heads = int(heads)
        self.out_dim = int(out_dim)
        self.concat = bool(concat)
        self.lin = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        self.bias = nn.Parameter(torch.empty(heads * out_dim if concat else out_dim))
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: (N, in_dim); edge_index: (2, E) with (src, dst), includes self-loops.
        n = x.shape[0]
        h = self.heads
        d = self.out_dim
        proj = self.lin(x).view(n, h, d)                      # (N, H, D)
        src, dst = edge_index[0], edge_index[1]               # (E,)
        a_src = (proj[src] * self.att_src).sum(-1)            # (E, H)
        a_dst = (proj[dst] * self.att_dst).sum(-1)            # (E, H)
        e = self.leaky(a_src + a_dst)                         # (E, H)

        # Segment softmax over incoming edges per destination node.
        e_max = torch.full((n, h), -1e9, device=x.device, dtype=e.dtype)
        e_max.index_reduce_(0, dst, e, "amax", include_self=False)
        alpha = torch.exp(e - e_max[dst])
        denom = torch.zeros(n, h, device=x.device, dtype=e.dtype)
        denom.index_add_(0, dst, alpha)
        alpha = alpha / denom[dst].clamp_min(1e-12)
        alpha = self.dropout(alpha)

        out = torch.zeros(n, h, d, device=x.device, dtype=proj.dtype)
        out.index_add_(0, dst, alpha.unsqueeze(-1) * proj[src])
        if self.concat:
            out = out.reshape(n, h * d)
        else:
            out = out.mean(dim=1)
        return out + self.bias


class ContrastiveGAT(nn.Module):
    """Two-layer sparse GAT producing L2-normalized context embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int = GAT_HIDDEN_DIM,
                 heads: int = GAT_NUM_HEADS, out_dim: int = GAT_OUT_DIM) -> None:
        super().__init__()
        self.layer1 = SparseGATLayer(in_dim, hidden_dim, heads=heads, concat=True)
        self.layer2 = SparseGATLayer(hidden_dim * heads, out_dim, heads=1, concat=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.layer1(x, edge_index))
        x = self.layer2(x, edge_index)
        return F.normalize(x, p=2, dim=1, eps=1e-12)


# ----------------------------------------------------------------------------
# Graph + pair construction (pure functions — unit-testable)
# ----------------------------------------------------------------------------
def build_frame_knn_graph(positions_zyx: np.ndarray, times: np.ndarray,
                          k: int = GAT_K_NEIGHBORS) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Per-frame KNN graph. Returns edge_index (2, E) with self-loops + stats.

    Raises if any edge ever connects two different frames (must never happen).
    """
    n = int(positions_zyx.shape[0])
    src_list: List[int] = list(range(n))
    dst_list: List[int] = list(range(n))  # self-loops
    n_frame_edges = 0
    for frame in np.unique(times):
        idx = np.where(times == frame)[0]
        if len(idx) < 2:
            continue
        dist = batch_distances_um(positions_zyx[idx], positions_zyx[idx], DEFAULT_SPACING)
        np.fill_diagonal(dist, np.inf)
        kk = min(int(k), len(idx) - 1)
        for li, gi in enumerate(idx):
            for lj in np.argsort(dist[li])[:kk]:
                src_list.append(int(gi))
                dst_list.append(int(idx[lj]))
                n_frame_edges += 1
    edge_index = np.asarray([src_list, dst_list], dtype=np.int64)
    # Safety: same-frame only.
    if edge_index.shape[1]:
        bad = int(np.sum(times[edge_index[0]] != times[edge_index[1]]))
        if bad:
            raise RuntimeError(f"KNN graph leaked across frames: {bad} edges")
    stats = {"n_nodes": n, "n_edges_incl_self": int(edge_index.shape[1]),
             "n_frame_edges": int(n_frame_edges), "k": int(k)}
    return edge_index, stats


def build_gt_pairs(node_ids: np.ndarray, times: np.ndarray,
                   geff_edges: Sequence[Tuple[int, int]],
                   neg_per_pos: int = GAT_NEG_PER_POS,
                   seed: int = GAT_SEED,
                   max_pairs: int = GAT_MAX_PAIRS_PER_MOVIE
                   ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Contrastive pairs from GEFF temporal links.

    Positive: (src, dst) with dt == 1 (same cell, consecutive frames).
    Negatives: for each positive, `neg_per_pos` distractors sampled from the
    same frame-pair: (src, dst') and (src', dst).

    Returns (pairs_idx (P,2), labels (P,) 1=pos/0=neg, info).
    """
    rng = np.random.default_rng(seed)
    index_of = {int(nid): i for i, nid in enumerate(node_ids)}
    by_frame: Dict[int, List[int]] = {}
    for i, t in enumerate(times):
        by_frame.setdefault(int(t), []).append(i)

    positives: List[Tuple[int, int]] = []
    skipped_dt = 0
    for s_id, d_id in geff_edges:
        if int(s_id) not in index_of or int(d_id) not in index_of:
            continue
        si, di = index_of[int(s_id)], index_of[int(d_id)]
        if int(times[di]) - int(times[si]) != 1:
            skipped_dt += 1
            continue
        positives.append((si, di))

    if len(positives) > max_pairs:
        sel = rng.choice(len(positives), size=max_pairs, replace=False)
        positives = [positives[i] for i in sel]

    pairs: List[Tuple[int, int]] = []
    labels: List[int] = []
    for si, di in positives:
        pairs.append((si, di))
        labels.append(1)
        t_s, t_d = int(times[si]), int(times[di])
        src_pool = [j for j in by_frame[t_s] if j != si]
        dst_pool = [j for j in by_frame[t_d] if j != di]
        for r in range(neg_per_pos):
            if r % 2 == 0 and dst_pool:
                pairs.append((si, int(rng.choice(dst_pool))))
                labels.append(0)
            elif src_pool:
                pairs.append((int(rng.choice(src_pool)), di))
                labels.append(0)
    info = {"n_positive": int(sum(labels)),
            "n_negative": int(len(labels) - sum(labels)),
            "n_gt_edges_total": int(len(geff_edges)),
            "n_gt_skipped_non_consecutive": int(skipped_dt)}
    return (np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
            np.asarray(labels, dtype=np.float32), info)


def contrastive_margin_loss(emb: torch.Tensor, pairs: torch.Tensor,
                            labels: torch.Tensor,
                            margin: float = GAT_MARGIN) -> torch.Tensor:
    """L = mean( y*d^2 + (1-y)*max(0, m-d)^2 ), d = euclidean distance."""
    d = torch.norm(emb[pairs[:, 0]] - emb[pairs[:, 1]], p=2, dim=1)
    pos = labels * d * d
    neg = (1.0 - labels) * torch.clamp(margin - d, min=0.0) ** 2
    return (pos + neg).mean()


# ----------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------
def _ensure_module2(movie_id: str) -> Path:
    feature_dir = Path(FEATURE_DIR) / movie_id
    if not (feature_dir / "fused_features.npz").exists():
        m2.run_full(movie_id)
    return feature_dir


def _load_movie_bundle(movie_id: str) -> Dict[str, Any]:
    feature_dir = _ensure_module2(movie_id)
    data = np.load(feature_dir / "fused_features.npz")
    bundle = {
        "movie_id": movie_id,
        "feature_dir": feature_dir,
        "node_ids": np.asarray(data["node_ids"], dtype=np.int64),
        "times": np.asarray(data["times"], dtype=np.int64),
        "positions": np.asarray(data["positions_zyx"], dtype=np.float32),
        "fused": np.asarray(data["fused_features"], dtype=np.float32),
    }
    geff_path = Path(LOCAL_DATA_ROOT) / f"{movie_id}.geff"
    bundle["geff_edges"] = load_geff(geff_path).edges if geff_path.exists() else []
    return bundle


class GATEngine:
    def __init__(self) -> None:
        self.feature_root = Path(FEATURE_DIR)
        self.device = torch.device("cpu")
        self.state: Dict[str, Any] = {"status": "idle", "progress": 0.0}

    # -- training ------------------------------------------------------
    def _train_contrastive(self, model: nn.Module,
                           bundles: List[Dict[str, Any]],
                           val_pairs: Optional[Dict[str, Any]],
                           epochs: int, lr: float) -> Dict[str, Any]:
        # Pre-encode static graphs per movie.
        prepared = []
        for b in bundles:
            edge_index, gstats = build_frame_knn_graph(b["positions"], b["times"])
            pairs, labels, pinfo = build_gt_pairs(b["node_ids"], b["times"], b["geff_edges"])
            n_val = int(len(pairs) * GAT_VAL_FRACTION)
            if val_pairs is None and n_val > 4 and len(pairs) > 8:
                # Within-movie monitoring split (single-movie mode only).
                perm = np.random.default_rng(GAT_SEED).permutation(len(pairs))
                vp, vl = pairs[perm[:n_val]], labels[perm[:n_val]]
                pairs, labels = pairs[perm[n_val:]], labels[perm[n_val:]]
                val_here = {"emb_movie": b["movie_id"], "pairs": vp, "labels": vl}
            else:
                val_here = None
            prepared.append({"bundle": b, "edge_index": edge_index,
                             "pairs": pairs, "labels": labels,
                             "graph_stats": gstats, "pair_info": pinfo,
                             "val_here": val_here})

        # External movie-level validation pairs (encoded with the same model).
        ext_val = None
        if val_pairs is not None:
            ext_val = val_pairs

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=GAT_WEIGHT_DECAY)
        history: List[Dict[str, float]] = []
        best_val = float("inf")
        best_state = None
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss, total_n = 0.0, 0
            for p in prepared:
                if len(p["pairs"]) == 0:
                    continue
                x = torch.from_numpy(p["bundle"]["fused"]).to(self.device)
                ei = torch.from_numpy(p["edge_index"]).to(self.device)
                pr = torch.from_numpy(p["pairs"]).to(self.device)
                lb = torch.from_numpy(p["labels"]).to(self.device)
                emb = model(x, ei)
                loss = contrastive_margin_loss(emb, pr, lb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(p["pairs"])
                total_n += len(p["pairs"])
            train_loss = total_loss / max(total_n, 1)

            # Validation (no grad).
            model.eval()
            with torch.no_grad():
                v_losses = []
                for p in prepared:
                    vh = p["val_here"]
                    if vh is None:
                        continue
                    x = torch.from_numpy(p["bundle"]["fused"]).to(self.device)
                    ei = torch.from_numpy(p["edge_index"]).to(self.device)
                    emb = model(x, ei)
                    pr = torch.from_numpy(vh["pairs"]).to(self.device)
                    lb = torch.from_numpy(vh["labels"]).to(self.device)
                    v_losses.append(float(contrastive_margin_loss(emb, pr, lb).cpu()))
                if ext_val is not None:
                    for key, bundle in ext_val["bundles"].items():
                        x = torch.from_numpy(bundle["fused"]).to(self.device)
                        ei = torch.from_numpy(bundle["edge_index"]).to(self.device)
                        emb = model(x, ei)
                        pr = torch.from_numpy(bundle["pairs"]).to(self.device)
                        lb = torch.from_numpy(bundle["labels"]).to(self.device)
                        v_losses.append(float(contrastive_margin_loss(emb, pr, lb).cpu()))
            val_loss = float(np.mean(v_losses)) if v_losses else float("nan")
            if v_losses and val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            history.append({"epoch": float(epoch),
                            "train_loss": round(train_loss, 6),
                            "val_loss": round(val_loss, 6) if v_losses else None})
            self.state["progress"] = 0.1 + 0.8 * epoch / epochs
            self.state["message"] = f"Contrastive GAT: epoch {epoch}/{epochs}"

        if best_state is not None:
            model.load_state_dict(best_state)
        return {"history": history, "prepared": prepared,
                "best_val_loss": best_val if best_state is not None else None}

    @torch.no_grad()
    def _encode(self, model: nn.Module, bundle: Dict[str, Any],
                edge_index: np.ndarray) -> np.ndarray:
        model.eval()
        x = torch.from_numpy(bundle["fused"]).to(self.device)
        ei = torch.from_numpy(np.asarray(edge_index)).to(self.device)
        return model(x, ei).detach().cpu().numpy().astype(np.float32)

    def _save_movie(self, movie_id: str, context: np.ndarray,
                    node_ids: np.ndarray, model: nn.Module,
                    extra: Dict[str, Any]) -> Dict[str, str]:
        feature_dir = Path(FEATURE_DIR) / movie_id
        feature_dir.mkdir(parents=True, exist_ok=True)
        out_path = feature_dir / "context_enhanced_features.npz"
        model_path = feature_dir / "neighbour_aware_gat.pt"
        # Mean self-attention as a diagnostic (recompute cheaply is costly;
        # store norms instead — honest, no invented attention values).
        norms = np.linalg.norm(context, axis=1)
        np.savez_compressed(out_path, node_ids=node_ids,
                            context_enhanced_features=context,
                            embedding_norms=norms.astype(np.float32))
        torch.save({"model_state_dict": model.state_dict(),
                    "config": extra.get("model_config", {}),
                    "mode": extra.get("mode", "contrastive")}, model_path)
        return {"context_enhanced_features": str(out_path),
                "gat_model": str(model_path)}

    # -- public: single movie ------------------------------------------
    def run(self, movie_id: str, epochs: int = GAT_EPOCHS,
            mode: str = "auto", lr: float = GAT_LR) -> Dict[str, Any]:
        t0 = time.time()
        self.state = {"status": "training", "progress": 0.05,
                      "message": f"Loading Module-2 features for {movie_id}"}
        torch.manual_seed(GAT_SEED)
        np.random.seed(GAT_SEED % (2 ** 32 - 1))

        bundle = _load_movie_bundle(movie_id)
        n_nodes, in_dim = bundle["fused"].shape
        if n_nodes == 0:
            raise RuntimeError("Module 2 returned no fused features.")

        _, _, pinfo_probe = build_gt_pairs(bundle["node_ids"], bundle["times"],
                                           bundle["geff_edges"])
        use_contrastive = (mode in ("contrastive", "auto") and pinfo_probe["n_positive"] > 0)
        if mode == "auto" and not use_contrastive:
            return self._run_selfsup(bundle, epochs)

        model = ContrastiveGAT(in_dim).to(self.device)
        fit = self._train_contrastive(model, [bundle], None, epochs, lr)
        prep = fit["prepared"][0]
        context = self._encode(model, bundle, prep["edge_index"])

        model_config = {"type": "ContrastiveGAT", "in_dim": int(in_dim),
                        "hidden_dim": GAT_HIDDEN_DIM, "heads": GAT_NUM_HEADS,
                        "out_dim": GAT_OUT_DIM, "margin": GAT_MARGIN,
                        "k_neighbors": GAT_K_NEIGHBORS,
                        "neg_per_pos": GAT_NEG_PER_POS, "epochs": int(epochs)}
        artifacts = self._save_movie(movie_id, context, bundle["node_ids"],
                                     model, {"mode": "contrastive",
                                             "model_config": model_config})
        result = {
            "movie_id": movie_id, "status": "complete", "mode": "contrastive",
            "n_cell_nodes": int(n_nodes),
            "graph": prep["graph_stats"], "pairs": prep["pair_info"],
            "val_scope": ("within-movie pair split (monitoring only); "
                          "use train_gat_multimovie() for movie-level validation"),
            "training_history": fit["history"],
            "final_train_loss": fit["history"][-1]["train_loss"],
            "best_val_loss": fit["best_val_loss"],
            "mean_embedding_norm": round(float(np.linalg.norm(context, axis=1).mean()), 4),
            "elapsed_sec": round(time.time() - t0, 1),
            "artifacts": artifacts,
        }
        (bundle["feature_dir"] / "module3_gat_summary.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8")
        self.state = {"status": "complete", "progress": 1.0,
                      "message": "Contrastive GAT complete", "result": result}
        return result

    def _run_selfsup(self, bundle: Dict[str, Any], epochs: int) -> Dict[str, Any]:
        """Legacy gated self-supervised path (movies without GT links)."""
        from modules.module2_spatiotemporal import ENGINE as m2engine  # noqa
        movie_id = bundle["movie_id"]
        fused = bundle["fused"]
        n_nodes, feature_dim = fused.shape
        edge_index, gstats = build_frame_knn_graph(bundle["positions"], bundle["times"])
        mask = np.zeros((n_nodes, n_nodes), dtype=bool)
        mask[edge_index[0], edge_index[1]] = True
        x = torch.from_numpy(fused).to(self.device)
        nmask = torch.from_numpy(mask).to(self.device)
        model = GatedGraphAttention(feature_dim).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        history = []
        for epoch in range(1, epochs + 1):
            model.train()
            out, _, _, recon = model(x, nmask)
            loss = F.mse_loss(recon, x) + 0.15 * F.mse_loss(out, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            history.append({"epoch": float(epoch),
                            "train_loss": round(float(loss.detach().cpu()), 6),
                            "val_loss": None})
        model.eval()
        with torch.no_grad():
            out, _, gate, _ = model(x, nmask)
        context = F.normalize(out, p=2, dim=1).cpu().numpy().astype(np.float32)
        artifacts = self._save_movie(movie_id, context, bundle["node_ids"], model,
                                     {"mode": "selfsup-gated",
                                      "model_config": {"type": "GatedGraphAttention",
                                                       "feature_dim": feature_dim}})
        result = {"movie_id": movie_id, "status": "complete", "mode": "selfsup-gated",
                  "n_cell_nodes": int(n_nodes), "graph": gstats,
                  "note": "No consecutive-frame GT links; self-supervised fallback used.",
                  "training_history": history,
                  "final_train_loss": history[-1]["train_loss"],
                  "mean_self_gate": round(float(gate.mean().cpu()), 4),
                  "artifacts": artifacts}
        (bundle["feature_dir"] / "module3_gat_summary.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8")
        self.state = {"status": "complete", "progress": 1.0,
                      "message": "Self-supervised GAT fallback complete",
                      "result": result}
        return result

    # -- public: movie-level training ----------------------------------
    def train_multimovie(self, train_movie_ids: Sequence[str],
                         val_movie_ids: Sequence[str],
                         epochs: int = GAT_EPOCHS,
                         lr: float = GAT_LR) -> Dict[str, Any]:
        t0 = time.time()
        overlap = set(train_movie_ids) & set(val_movie_ids)
        if overlap:
            raise ValueError(f"Train/val movie overlap (leakage): {sorted(overlap)}")
        if not train_movie_ids or not val_movie_ids:
            raise ValueError("Need >=1 train and >=1 val movie for movie-level training.")
        torch.manual_seed(GAT_SEED)
        self.state = {"status": "training", "progress": 0.05,
                      "message": "Movie-level contrastive GAT training"}

        train_bundles = [_load_movie_bundle(m) for m in train_movie_ids]
        in_dim = train_bundles[0]["fused"].shape[1]
        val_bundles: Dict[str, Any] = {}
        for m in val_movie_ids:
            b = _load_movie_bundle(m)
            ei, _ = build_frame_knn_graph(b["positions"], b["times"])
            pr, lb, _ = build_gt_pairs(b["node_ids"], b["times"], b["geff_edges"])
            val_bundles[m] = {"fused": b["fused"], "edge_index": ei,
                              "pairs": pr, "labels": lb}
        model = ContrastiveGAT(in_dim).to(self.device)
        fit = self._train_contrastive(model, train_bundles,
                                      {"bundles": val_bundles}, epochs, lr)

        # Encode every movie with the shared trained model.
        encoded: Dict[str, Any] = {}
        for b in train_bundles + [_load_movie_bundle(m) for m in val_movie_ids]:
            ei, _ = build_frame_knn_graph(b["positions"], b["times"])
            ctx = self._encode(model, b, ei)
            arts = self._save_movie(b["movie_id"], ctx, b["node_ids"], model,
                                    {"mode": "contrastive-multimovie",
                                     "model_config": {"type": "ContrastiveGAT",
                                                      "in_dim": int(in_dim)}})
            encoded[b["movie_id"]] = arts
        result = {"status": "complete", "mode": "contrastive-multimovie",
                  "train_movies": list(train_movie_ids),
                  "val_movies": list(val_movie_ids),
                  "training_history": fit["history"],
                  "best_val_loss": fit["best_val_loss"],
                  "per_movie_pairs": {b["movie_id"]: p["pair_info"]
                                      for b, p in zip(train_bundles, fit["prepared"])},
                  "elapsed_sec": round(time.time() - t0, 1),
                  "artifacts": encoded}
        self.state = {"status": "complete", "progress": 1.0,
                      "message": "Movie-level GAT training complete",
                      "result": result}
        return result


ENGINE = GATEngine()


def run_gat(movie_id: str, epochs: int = GAT_EPOCHS,
            mode: str = "auto") -> Dict[str, Any]:
    return ENGINE.run(movie_id, epochs=epochs, mode=mode)


def train_gat_multimovie(train_movie_ids: Sequence[str],
                         val_movie_ids: Sequence[str],
                         epochs: int = GAT_EPOCHS) -> Dict[str, Any]:
    return ENGINE.train_multimovie(train_movie_ids, val_movie_ids, epochs=epochs)


def module_status() -> Dict[str, Any]:
    return ENGINE.state


def safe_call(fn, *args, **kwargs) -> Dict[str, Any]:
    try:
        return {"ok": True, "result": fn(*args, **kwargs)}
    except Exception as exc:
        return {"ok": False, "error": str(exc),
                "traceback": traceback.format_exc()}
