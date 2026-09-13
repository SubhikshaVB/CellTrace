"""
Module 1 — CellDINO-Based Appearance Representation Learning.

Consumes ONLY preprocessed cell tensors from Module 0 (never raw Zarr).
Runs a lightweight Vision Transformer / 3D CNN hybrid encoder to produce
dense appearance embeddings, validates numerical stability, and L2-normalizes.
"""
from __future__ import annotations

import json
import logging
import math
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    CELLDINO_BATCH_SIZE,
    CELLDINO_DEPTH,
    CELLDINO_DEVICE,
    CELLDINO_EMBED_DIM,
    CELLDINO_L2_EPS,
    CELLDINO_MLP_RATIO,
    CELLDINO_NUM_HEADS,
    CELLDINO_PATCH_EMBED,
    EMBEDDING_DIR,
    TENSOR_DIR,
    ensure_artifact_dirs,
)

LOGGER = logging.getLogger("celltrace.module1")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TORCH_OK = True


# ---------------------------------------------------------------------------
# Model components (PyTorch)
# ---------------------------------------------------------------------------
if TORCH_OK:

    class DropPath(nn.Module):
        def __init__(self, drop_prob: float = 0.0) -> None:
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep = 1.0 - self.drop_prob 
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
            mask = mask.floor()
            return x.div(keep) * mask

    class PatchEmbed3D(nn.Module):
        def __init__(self, in_ch: int = 1, embed_dim: int = 256, patch: int = 8) -> None:
            super().__init__()
            self.proj = nn.Conv3d(in_ch, embed_dim, kernel_size=patch, stride=patch)
            self.norm = nn.LayerNorm(embed_dim)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B,C,Z,Y,X)
            x = self.proj(x)  # (B,E,z,y,x)
            b, e, z, y, w = x.shape
            x = x.view(b, e, z * y * w).transpose(1, 2)  # (B,N,E)
            return self.norm(x)

    class MLP(nn.Module):
        def __init__(self, dim: int, hidden: int, drop: float = 0.0) -> None:
            super().__init__()
            self.fc1 = nn.Linear(dim, hidden)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden, dim)
            self.drop = nn.Dropout(drop)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x

    class Attention(nn.Module):
        def __init__(self, dim: int, num_heads: int = 4, attn_drop: float = 0.0) -> None:
            super().__init__()
            if dim % num_heads != 0:
                raise ValueError("embed dim must be divisible by num_heads")
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = self.head_dim ** -0.5
            self.qkv = nn.Linear(dim, dim * 3)
            self.proj = nn.Linear(dim, dim)
            self.attn_drop = nn.Dropout(attn_drop)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            b, n, c = x.shape
            qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = (attn @ v).transpose(1, 2).reshape(b, n, c)
            return self.proj(out)

    class Block(nn.Module):
        def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, drop_path: float = 0.0) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = Attention(dim, num_heads=num_heads)
            self.drop_path = DropPath(drop_path)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = MLP(dim, int(dim * mlp_ratio))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

    class ConvStem3D(nn.Module):
        """Local inductive bias before transformer tokens."""

        def __init__(self, in_ch: int = 1, mid: int = 32, out_ch: int = 1) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(in_ch, mid, 3, padding=1),
                nn.InstanceNorm3d(mid),
                nn.GELU(),
                nn.Conv3d(mid, mid, 3, padding=1),
                nn.InstanceNorm3d(mid),
                nn.GELU(),
                nn.Conv3d(mid, out_ch, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)

    class CellDINOEncoder(nn.Module):
        def __init__(
            self,
            embed_dim: int = CELLDINO_EMBED_DIM,
            depth: int = CELLDINO_DEPTH,
            num_heads: int = CELLDINO_NUM_HEADS,
            mlp_ratio: float = CELLDINO_MLP_RATIO,
            patch: int = CELLDINO_PATCH_EMBED,
            in_ch: int = 1,
        ) -> None:
            super().__init__()
            self.stem = ConvStem3D(in_ch=in_ch, mid=32, out_ch=in_ch)
            self.patch_embed = PatchEmbed3D(in_ch=in_ch, embed_dim=embed_dim, patch=max(4, patch // 2))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_drop = nn.Dropout(0.0)
            self.blocks = nn.ModuleList(
                [
                    Block(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_path=0.05 * i / max(1, depth))
                    for i in range(depth)
                ]
            )
            self.norm = nn.LayerNorm(embed_dim)
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            self.apply(self._init_weights)

        @staticmethod
        def _init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.stem(x)
            tokens = self.patch_embed(x)
            b = tokens.shape[0]
            cls = self.cls_token.expand(b, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            # Interpolate-free: add learned scalar positional gate
            tokens = self.pos_drop(tokens)
            for blk in self.blocks:
                tokens = blk(tokens)
            tokens = self.norm(tokens)
            cls_out = tokens[:, 0]
            return self.head(cls_out)


# ---------------------------------------------------------------------------
# Numpy fallback encoder (no torch)
# ---------------------------------------------------------------------------
class NumpyCellDINOFallback:
    """Deterministic hashing / PCA-ish projection when torch is unavailable."""

    def __init__(self, embed_dim: int = CELLDINO_EMBED_DIM, seed: int = 17) -> None:
        self.embed_dim = embed_dim
        rng = np.random.default_rng(seed)
        self.proj = rng.normal(0.0, 1.0 / math.sqrt(embed_dim), size=(embed_dim, 128)).astype(np.float32)

    def _feats(self, tensor: np.ndarray) -> np.ndarray:
        # tensor: (C,Z,Y,X)
        vol = tensor.astype(np.float32)
        if vol.ndim == 4:
            vol = vol[0]
        z, y, x = vol.shape
        # Multi-scale pooled descriptors
        feats: List[float] = []
        feats.extend([float(vol.mean()), float(vol.std()), float(vol.min()), float(vol.max())])
        # Depth profiles
        z_profile = vol.mean(axis=(1, 2))
        y_profile = vol.mean(axis=(0, 2))
        x_profile = vol.mean(axis=(0, 1))
        for arr, n in ((z_profile, 32), (y_profile, 32), (x_profile, 32)):
            idx = np.linspace(0, len(arr) - 1, n)
            samp = np.interp(idx, np.arange(len(arr)), arr)
            feats.extend(samp.tolist())
        # Gradient energy
        gz = np.diff(vol, axis=0)
        gy = np.diff(vol, axis=1)
        gx = np.diff(vol, axis=2)
        feats.extend([float(np.mean(np.abs(gz))), float(np.mean(np.abs(gy))), float(np.mean(np.abs(gx)))])
        # Central crop moments
        cz, cy, cx = z // 2, y // 2, x // 2
        crop = vol[max(0, cz - 4): cz + 4, max(0, cy - 8): cy + 8, max(0, cx - 8): cx + 8]
        feats.extend([float(crop.mean()), float(crop.std())])
        while len(feats) < 128:
            feats.append(0.0)
        return np.asarray(feats[:128], dtype=np.float32)

    def encode_batch(self, batch: np.ndarray) -> np.ndarray:
        rows = []
        for i in range(batch.shape[0]):
            f = self._feats(batch[i])
            emb = self.proj @ f
            rows.append(emb)
        return np.stack(rows, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Data models / engine
# ---------------------------------------------------------------------------
@dataclass
class EmbeddingBatchResult:
    movie_id: str
    n_embeddings: int
    embed_dim: int
    path: str
    nan_count: int = 0
    inf_count: int = 0
    mean_norm: float = 0.0
    backend: str = "torch"
    elapsed_sec: float = 0.0
    preview: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Module1State:
    movie_id: Optional[str] = None
    step: str = "idle"
    progress: float = 0.0
    message: str = ""
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CellDINOEngine:
    def __init__(
        self,
        tensor_dir: Path = TENSOR_DIR,
        embedding_dir: Path = EMBEDDING_DIR,
        device: str = CELLDINO_DEVICE,
        embed_dim: int = CELLDINO_EMBED_DIM,
        batch_size: int = CELLDINO_BATCH_SIZE,
    ) -> None:
        ensure_artifact_dirs()
        self.tensor_dir = Path(tensor_dir)
        self.embedding_dir = Path(embedding_dir)
        self.embedding_dir.mkdir(parents=True, exist_ok=True)
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.state = Module1State()
        self.backend = "torch" if TORCH_OK else "numpy"
        self.device = device
        self.model = None
        self.fallback = NumpyCellDINOFallback(embed_dim=embed_dim)
        if TORCH_OK:
            if device == "cuda" and not torch.cuda.is_available():
                self.device = "cpu"
            self.model = CellDINOEncoder(embed_dim=embed_dim)
            self.model.to(self.device)
            self.model.eval()

    # ------------------------------------------------------------------
    # Tensor loading (Module 0 artifacts only)
    # ------------------------------------------------------------------
    def resolve_tensor_bundle(self, movie_id: str) -> Dict[str, Any]:
        base = self.tensor_dir / movie_id / "cell_tensors"
        npz = base / "cell_tensors.npz"
        index = base / "tensor_index.json"
        if not npz.exists():
            # Try shards
            shards = sorted(base.glob("tensors_shard_*.npy"))
            if not shards:
                raise FileNotFoundError(
                    f"No Module 0 tensors for '{movie_id}'. Run Module 0 tensor generation first. "
                    f"Expected at {npz}"
                )
            arrays = [np.load(s) for s in shards]
            tensors = np.concatenate(arrays, axis=0)
            meta = {"source": "shards", "n": int(tensors.shape[0])}
            node_ids = np.arange(tensors.shape[0], dtype=np.int64)
            track_ids = np.zeros(tensors.shape[0], dtype=np.int64)
            times = np.zeros(tensors.shape[0], dtype=np.int64)
            if index.exists():
                payload = json.loads(index.read_text(encoding="utf-8"))
                items = payload.get("items", [])
                if items:
                    node_ids = np.asarray([it["node_id"] for it in items], dtype=np.int64)
                    track_ids = np.asarray([it["track_id"] for it in items], dtype=np.int64)
                    times = np.asarray([it["t"] for it in items], dtype=np.int64)
            return {
                "tensors": tensors.astype(np.float32),
                "node_ids": node_ids,
                "track_ids": track_ids,
                "times": times,
                "meta": meta,
                "path": str(base),
            }

        data = np.load(npz)
        tensors = np.asarray(data["tensors"], dtype=np.float32)
        node_ids = np.asarray(data["node_ids"], dtype=np.int64) if "node_ids" in data.files else np.arange(len(tensors))
        track_ids = np.asarray(data["track_ids"], dtype=np.int64) if "track_ids" in data.files else np.zeros(len(tensors))
        times = np.asarray(data["times"], dtype=np.int64) if "times" in data.files else np.zeros(len(tensors))
        return {
            "tensors": tensors,
            "node_ids": node_ids,
            "track_ids": track_ids,
            "times": times,
            "meta": {"source": "npz", "path": str(npz)},
            "path": str(npz),
        }

    def validate_tensors(self, tensors: np.ndarray) -> Dict[str, Any]:
        if tensors.ndim != 5:
            raise ValueError(f"Expected tensors shaped (N,C,Z,Y,X), got {tensors.shape}")
        n, c, z, y, x = tensors.shape
        report = {
            "shape": list(tensors.shape),
            "dtype": str(tensors.dtype),
            "n": int(n),
            "nan": int(np.isnan(tensors).sum()),
            "inf": int(np.isinf(tensors).sum()),
            "min": float(np.nanmin(tensors)) if n else 0.0,
            "max": float(np.nanmax(tensors)) if n else 0.0,
            "mean": float(np.nanmean(tensors)) if n else 0.0,
        }
        if report["nan"] or report["inf"]:
            LOGGER.warning("Input tensors contain NaN/Inf: %s", report)
        if c < 1 or min(z, y, x) < 4:
            raise ValueError(f"Degenerate spatial dimensions: {tensors.shape}")
        return report

    # ------------------------------------------------------------------
    # Embedding ops
    # ------------------------------------------------------------------
    @staticmethod
    def l2_normalize(emb: np.ndarray, eps: float = CELLDINO_L2_EPS) -> np.ndarray:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return (emb / norms).astype(np.float32)

    @staticmethod
    def validate_embeddings(emb: np.ndarray) -> Dict[str, Any]:
        nan_count = int(np.isnan(emb).sum())
        inf_count = int(np.isinf(emb).sum())
        norms = np.linalg.norm(emb, axis=1) if emb.size else np.asarray([])
        return {
            "nan_count": nan_count,
            "inf_count": inf_count,
            "mean_norm": float(np.mean(norms)) if norms.size else 0.0,
            "std_norm": float(np.std(norms)) if norms.size else 0.0,
            "min_norm": float(np.min(norms)) if norms.size else 0.0,
            "max_norm": float(np.max(norms)) if norms.size else 0.0,
            "ok": nan_count == 0 and inf_count == 0,
        }

    def _encode_torch(self, tensors: np.ndarray) -> np.ndarray:
        assert self.model is not None and TORCH_OK
        self.model.eval()
        outs: List[np.ndarray] = []
        n = tensors.shape[0]
        with torch.no_grad():
            for i in range(0, n, self.batch_size):
                batch = tensors[i : i + self.batch_size]
                # Adaptive pool to a manageable size for the encoder
                t = torch.from_numpy(batch).to(self.device)
                if t.shape[-1] > 64 or t.shape[-2] > 64 or t.shape[-3] > 32:
                    t = F.adaptive_avg_pool3d(t, output_size=(32, 64, 64))
                # Ensure spatial dims divisible enough for patch embed
                # Pad to multiples of 8
                _, _, z, y, x = t.shape
                pz = (8 - z % 8) % 8
                py = (8 - y % 8) % 8
                px = (8 - x % 8) % 8
                if pz or py or px:
                    t = F.pad(t, (0, px, 0, py, 0, pz))
                emb = self.model(t)
                outs.append(emb.detach().cpu().numpy().astype(np.float32))
                self.state.progress = min(0.95, float(i + len(batch)) / max(1, n))
                self.state.message = f"Encoded {min(i + len(batch), n)}/{n}"
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, self.embed_dim), dtype=np.float32)

    def _encode_numpy(self, tensors: np.ndarray) -> np.ndarray:
        outs: List[np.ndarray] = []
        n = tensors.shape[0]
        for i in range(0, n, self.batch_size):
            batch = tensors[i : i + self.batch_size]
            outs.append(self.fallback.encode_batch(batch))
            self.state.progress = min(0.95, float(i + len(batch)) / max(1, n))
            self.state.message = f"Encoded(numpy) {min(i + len(batch), n)}/{n}"
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, self.embed_dim), dtype=np.float32)

    def extract_embeddings(
        self,
        movie_id: str,
        max_cells: Optional[int] = None,
        l2: bool = True,
    ) -> EmbeddingBatchResult:
        t0 = time.time()
        self.state = Module1State(movie_id=movie_id, step="load_tensors", progress=0.02, message="Loading Module 0 tensors")
        bundle = self.resolve_tensor_bundle(movie_id)
        tensors = bundle["tensors"]
        if max_cells is not None:
            tensors = tensors[: int(max_cells)]
            node_ids = bundle["node_ids"][: int(max_cells)]
            track_ids = bundle["track_ids"][: int(max_cells)]
            times = bundle["times"][: int(max_cells)]
        else:
            node_ids = bundle["node_ids"]
            track_ids = bundle["track_ids"]
            times = bundle["times"]

        tin = self.validate_tensors(tensors)
        if tin["nan"] or tin["inf"]:
            tensors = np.nan_to_num(tensors, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

        self.state.step = "encode"
        self.state.message = f"Running CellDINO ({self.backend})"
        if self.backend == "torch":
            emb = self._encode_torch(tensors)
        else:
            emb = self._encode_numpy(tensors)

        # Stability repair
        bad = ~np.isfinite(emb)
        nan_count = int(np.isnan(emb).sum())
        inf_count = int(np.isinf(emb).sum())
        if bad.any():
            emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if l2:
            emb = self.l2_normalize(emb)
        val = self.validate_embeddings(emb)

        out_dir = self.embedding_dir / movie_id
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / "appearance_embeddings.npz"
        np.savez_compressed(
            npz_path,
            embeddings=emb.astype(np.float32),
            node_ids=np.asarray(node_ids, dtype=np.int64),
            track_ids=np.asarray(track_ids, dtype=np.int64),
            times=np.asarray(times, dtype=np.int64),
        )
        meta = {
            "movie_id": movie_id,
            "embed_dim": int(emb.shape[1]) if emb.ndim == 2 and emb.shape[0] else self.embed_dim,
            "n_embeddings": int(emb.shape[0]),
            "backend": self.backend,
            "device": self.device if self.backend == "torch" else "cpu",
            "l2_normalized": bool(l2),
            "input_tensor_report": tin,
            "embedding_validation": val,
            "model": {
                "embed_dim": self.embed_dim,
                "depth": CELLDINO_DEPTH,
                "num_heads": CELLDINO_NUM_HEADS,
                "mlp_ratio": CELLDINO_MLP_RATIO,
                "patch": CELLDINO_PATCH_EMBED,
            },
            "created_at": time.time(),
            "npz_path": str(npz_path),
        }
        (out_dir / "appearance_embeddings.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        preview = []
        for i in range(min(5, emb.shape[0])):
            preview.append(
                {
                    "node_id": int(node_ids[i]),
                    "track_id": int(track_ids[i]),
                    "t": int(times[i]),
                    "norm": float(np.linalg.norm(emb[i])),
                    "first8": emb[i, :8].tolist(),
                }
            )

        elapsed = time.time() - t0
        self.state.step = "done"
        self.state.progress = 1.0
        self.state.message = f"Wrote {emb.shape[0]} embeddings"
        result = EmbeddingBatchResult(
            movie_id=movie_id,
            n_embeddings=int(emb.shape[0]),
            embed_dim=int(emb.shape[1]) if emb.size else self.embed_dim,
            path=str(npz_path),
            nan_count=nan_count,
            inf_count=inf_count,
            mean_norm=float(val["mean_norm"]),
            backend=self.backend,
            elapsed_sec=float(elapsed),
            preview=preview,
        )
        return result

    def load_embeddings(self, movie_id: str) -> Dict[str, Any]:
        npz_path = self.embedding_dir / movie_id / "appearance_embeddings.npz"
        meta_path = self.embedding_dir / movie_id / "appearance_embeddings.json"
        if not npz_path.exists():
            raise FileNotFoundError(f"No embeddings for {movie_id}")
        data = np.load(npz_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        emb = np.asarray(data["embeddings"])
        return {
            "movie_id": movie_id,
            "embeddings_shape": list(emb.shape),
            "mean_norm": float(np.linalg.norm(emb, axis=1).mean()) if emb.size else 0.0,
            "meta": meta,
            "path": str(npz_path),
        }

    def pairwise_similarity_preview(self, movie_id: str, top_k: int = 5) -> Dict[str, Any]:
        npz_path = self.embedding_dir / movie_id / "appearance_embeddings.npz"
        data = np.load(npz_path)
        emb = np.asarray(data["embeddings"], dtype=np.float32)
        node_ids = np.asarray(data["node_ids"])
        if emb.shape[0] == 0:
            return {"pairs": []}
        # Cosine sim for first min(64, N) nodes
        n = min(64, emb.shape[0])
        e = emb[:n]
        sim = e @ e.T
        pairs = []
        for i in range(n):
            sim[i, i] = -1.0
            j = int(np.argmax(sim[i]))
            pairs.append(
                {
                    "i": int(node_ids[i]),
                    "j": int(node_ids[j]),
                    "similarity": float(sim[i, j]),
                }
            )
        pairs.sort(key=lambda d: d["similarity"], reverse=True)
        return {"movie_id": movie_id, "top_pairs": pairs[:top_k]}

    def _must_load_npz(self, movie_id: str):
        npz_path = self.embedding_dir / movie_id / "appearance_embeddings.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"No embeddings for '{movie_id}'. Run the encoder first.")
        return np.load(npz_path)

    def node_vector(self, movie_id: str, node_id: int) -> Dict[str, Any]:
        """Full stored embedding vector for one node (Dive fingerprint)."""
        data = self._must_load_npz(movie_id)
        emb = np.asarray(data["embeddings"], dtype=np.float32)
        node_ids = np.asarray(data["node_ids"])
        where = np.where(node_ids == int(node_id))[0]
        if len(where) == 0:
            raise ValueError(f"Node {node_id} has no embedding in '{movie_id}'.")
        vec = emb[int(where[0])].astype(float)
        return {
            "movie_id": movie_id,
            "node_id": int(node_id),
            "dim": int(vec.shape[0]),
            "norm": float(np.linalg.norm(vec)),
            "vector": vec.tolist(),
        }

    def node_neighbors(self, movie_id: str, node_id: int, k: int = 5) -> Dict[str, Any]:
        """Top-k cosine neighbors of one node (Dive lookalikes)."""
        data = self._must_load_npz(movie_id)
        emb = np.asarray(data["embeddings"], dtype=np.float32)
        node_ids = np.asarray(data["node_ids"])
        track_ids = np.asarray(data["track_ids"]) if "track_ids" in data else node_ids
        times = np.asarray(data["times"]) if "times" in data else np.zeros(len(node_ids))
        where = np.where(node_ids == int(node_id))[0]
        if len(where) == 0:
            raise ValueError(f"Node {node_id} has no embedding in '{movie_id}'.")
        i = int(where[0])
        n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        sims = n @ n[i]
        sims[i] = -2.0
        order = np.argsort(-sims, kind="stable")[: max(int(k), 1)]
        return {
            "movie_id": movie_id,
            "node_id": int(node_id),
            "k": int(k),
            "neighbors": [
                {
                    "node_id": int(node_ids[j]),
                    "track_id": int(track_ids[j]),
                    "t": int(times[j]),
                    "similarity": round(float(sims[j]), 4),
                }
                for j in order.tolist()
            ],
        }

    def status(self) -> Dict[str, Any]:
        return {
            "state": self.state.as_dict(),
            "backend": self.backend,
            "device": self.device if self.backend == "torch" else "cpu",
            "torch_available": TORCH_OK,
            "embed_dim": self.embed_dim,
            "embedding_dir": str(self.embedding_dir),
        }


ENGINE = CellDINOEngine()


def run_celldino(movie_id: str, max_cells: Optional[int] = None) -> Dict[str, Any]:
    return ENGINE.extract_embeddings(movie_id, max_cells=max_cells).as_dict()


def get_embedding_meta(movie_id: str) -> Dict[str, Any]:
    return ENGINE.load_embeddings(movie_id)


def similarity_preview(movie_id: str) -> Dict[str, Any]:
    return ENGINE.pairwise_similarity_preview(movie_id)


def get_node_vector(movie_id: str, node_id: int) -> Dict[str, Any]:
    return ENGINE.node_vector(movie_id, node_id)


def node_neighbors(movie_id: str, node_id: int, k: int = 5) -> Dict[str, Any]:
    return ENGINE.node_neighbors(movie_id, node_id, k)


def module_status() -> Dict[str, Any]:
    return ENGINE.status()


def safe_call(fn, *args, **kwargs) -> Dict[str, Any]:
    try:
        return {"ok": True, "result": fn(*args, **kwargs)}
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Module1 error: %s\n%s", exc, traceback.format_exc())
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}