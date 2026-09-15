"""
CellTrace global configuration.

Centralizes physical voxel spacing (anisotropy), dataset roots, artifact paths,
and model / batch hyperparameters used by Modules 0–2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Repository roots
# ---------------------------------------------------------------------------
BACKEND_ROOT: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = BACKEND_ROOT.parent
LOCAL_DATA_ROOT: Path = BACKEND_ROOT / "local_data" / "train"
ARTIFACTS_ROOT: Path = BACKEND_ROOT / "artifacts"

TENSOR_DIR: Path = ARTIFACTS_ROOT / "tensors"
EMBEDDING_DIR: Path = ARTIFACTS_ROOT / "embeddings"
FEATURE_DIR: Path = ARTIFACTS_ROOT / "features"
TRACK_DIR: Path = ARTIFACTS_ROOT / "tracks"
EVAL_DIR: Path = ARTIFACTS_ROOT / "evaluations"

for _d in (TENSOR_DIR, EMBEDDING_DIR, FEATURE_DIR, TRACK_DIR, EVAL_DIR, LOCAL_DATA_ROOT):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Physical voxel spacing (µm) — Biohub / Cell Tracking challenge convention
# ---------------------------------------------------------------------------
VOXEL_SIZE_Z_UM: float = 1.625
VOXEL_SIZE_Y_UM: float = 0.40625
VOXEL_SIZE_X_UM: float = 0.40625
ANISOTROPY_RATIO_Z_OVER_XY: float = VOXEL_SIZE_Z_UM / VOXEL_SIZE_X_UM  # ~4.0

SPACING_ZYX_UM: Tuple[float, float, float] = (
    VOXEL_SIZE_Z_UM,
    VOXEL_SIZE_Y_UM,
    VOXEL_SIZE_X_UM,
)

# ---------------------------------------------------------------------------
# Module 0 — preprocessing
# ---------------------------------------------------------------------------
PERCENTILE_LOW: float = 1.0
PERCENTILE_HIGH: float = 99.0
DEFAULT_PATCH_MARGIN_UM: float = 8.0
MIN_PATCH_SIZE_ZYX: Tuple[int, int, int] = (8, 16, 16)
MAX_PATCH_SIZE_ZYX: Tuple[int, int, int] = (64, 128, 128)
TARGET_TENSOR_SIZE_ZYX: Tuple[int, int, int] = (32, 64, 64)
SLICE_JPEG_QUALITY: int = 85

# ---------------------------------------------------------------------------
# Module 1 — CellDINO appearance
# ---------------------------------------------------------------------------
CELLDINO_EMBED_DIM: int = 256
CELLDINO_PATCH_EMBED: int = 16
CELLDINO_DEPTH: int = 4
CELLDINO_NUM_HEADS: int = 4
CELLDINO_MLP_RATIO: float = 2.0
CELLDINO_BATCH_SIZE: int = 8
CELLDINO_L2_EPS: float = 1e-12
CELLDINO_DEVICE: str = "cpu"  # override to "cuda" when available

# ---------------------------------------------------------------------------
# Module 2 — spatiotemporal (partial for Review 1)
# ---------------------------------------------------------------------------
KNN_NEIGHBORS: int = 8
NEIGHBOR_RADIUS_UM: float = 25.0
VELOCITY_SMOOTH_WINDOW: int = 3

# ---------------------------------------------------------------------------
# Module 3 — contrastive neighbour-aware GAT (Review 2)
#
# Graph: per-frame KNN rebuilt from Module-2 positions (same frame only).
# Training: contrastive loss on GEFF ground-truth pairs, movie-level splits.
# ---------------------------------------------------------------------------
GAT_K_NEIGHBORS: int = 8
GAT_HIDDEN_DIM: int = 128
GAT_NUM_HEADS: int = 4
GAT_OUT_DIM: int = 128
GAT_MARGIN: float = 1.0
GAT_NEG_PER_POS: int = 3
GAT_LR: float = 1e-3
GAT_WEIGHT_DECAY: float = 1e-4
GAT_EPOCHS: int = 50
GAT_SEED: int = 2026
GAT_VAL_FRACTION: float = 0.2  # pair-level val split *within* train movies only
GAT_MAX_PAIRS_PER_MOVIE: int = 20000  # cap for very large movies

# ---------------------------------------------------------------------------
# Module 4 — confidence-aware tracking, first half (Review 2)
#
# Matching + similarity + Hungarian + confidence.
# Thresholding and trajectory stitching are deferred to Review 3.
# ---------------------------------------------------------------------------
TRACK_MAX_DIST_UM: float = 15.0
TRACK_TOP_K_CANDIDATES: int = 8
TRACK_GROUP_NAMES: Tuple[str, ...] = (
    "appearance",
    "spatial",
    "kinematic",
    "neighborhood",
    "gat_context",
)
TRACK_WEIGHT_MLP_HIDDEN: int = 32
TRACK_WEIGHT_LR: float = 1e-3
TRACK_WEIGHT_EPOCHS: int = 100
TRACK_WEIGHT_SEED: int = 2026
TRACK_CONFIDENCE_TEMP: float = 1.0

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_TITLE: str = "CellTrace API"
API_VERSION: str = "0.1.0-review1"
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def spacing_dict() -> Dict[str, float]:
    return {
        "z_um": VOXEL_SIZE_Z_UM,
        "y_um": VOXEL_SIZE_Y_UM,
        "x_um": VOXEL_SIZE_X_UM,
        "anisotropy_z_over_xy": ANISOTROPY_RATIO_Z_OVER_XY,
    }


def ensure_artifact_dirs() -> None:
    for d in (TENSOR_DIR, EMBEDDING_DIR, FEATURE_DIR, TRACK_DIR, EVAL_DIR):
        d.mkdir(parents=True, exist_ok=True)
