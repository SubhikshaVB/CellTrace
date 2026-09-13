"""
Self-supervised CellDINO training for CellTrace Review 2.

Training is split by complete embryo movie, never by individual patches:
- 70% train movies
- 15% validation movies
- 15% held-out test movies

GEFF is used here only to obtain cell-centred crops during preparation.
DINO training itself receives only augmented image tensors, not cell labels.
"""
from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ARTIFACTS_ROOT, LOCAL_DATA_ROOT, TENSOR_DIR
from modules import module0_data_prep as m0
from modules import module1_celldino as m1


MODEL_DIR = ARTIFACTS_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_STATE: Dict[str, Any] = {
    "status": "idle",
    "message": "Training has not started.",
    "progress": 0.0,
}


class DINOProjectionHead(nn.Module):
    """Maps the 256-value encoder representation to DINO prototypes."""

    def __init__(self, in_dim: int = 256, hidden_dim: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def discover_complete_movies(data_root: Path = LOCAL_DATA_ROOT) -> List[str]:
    """
    Return only movies with both required raw sources present:
    <movie>.zarr and <movie>.geff.
    """
    root = Path(data_root)
    movies: List[str] = []

    for zarr_path in sorted(root.glob("*.zarr")):
        movie_id = zarr_path.name.removesuffix(".zarr")
        geff_path = root / f"{movie_id}.geff"

        if (
            zarr_path.is_dir()
            and geff_path.is_dir()
            and (zarr_path / "zarr.json").exists()
            and (geff_path / "zarr.json").exists()
        ):
            movies.append(movie_id)

    return movies


def movie_level_split(movie_ids: List[str], seed: int = 42) -> Dict[str, List[str]]:
    """
    A whole embryo belongs to exactly one split.
    This prevents information leakage across train/validation/test.
    """
    ids = list(sorted(movie_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    if n < 3:
        return {"train": ids, "validation": [], "test": []}

    n_test = max(1, round(n * 0.15))
    n_val = max(1, round(n * 0.15))
    n_train = max(1, n - n_val - n_test)

    return {
        "train": ids[:n_train],
        "validation": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }


def _tensor_npz_path(movie_id: str) -> Path:
    return TENSOR_DIR / movie_id / "cell_tensors" / "cell_tensors.npz"


def ensure_movie_tensors(movie_id: str, source_cells_per_movie: int) -> Path:
    """
    Creates Module 0 tensors when this movie has not been prepared already.
    It never prepares held-out test movies during training.
    """
    path = _tensor_npz_path(movie_id)
    if path.exists():
        return path

    m0.ENGINE.generate_tensors(
        movie_id=movie_id,
        max_cells=source_cells_per_movie,
    )

    if not path.exists():
        raise RuntimeError(f"Tensor generation failed for {movie_id}")

    return path


def load_evenly_sampled_tensors(
    movie_ids: List[str],
    source_cells_per_movie: int,
    samples_per_movie: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Takes evenly spaced observations from each embryo rather than only
    consecutive early frames of one cell.
    """
    arrays: List[np.ndarray] = []
    counts: Dict[str, int] = {}

    for movie_id in movie_ids:
        path = ensure_movie_tensors(movie_id, source_cells_per_movie)
        bundle = np.load(path)

        tensors = np.asarray(bundle["tensors"], dtype=np.float32)
        if tensors.shape[0] == 0:
            continue

        take = min(samples_per_movie, tensors.shape[0])
        indices = np.linspace(0, tensors.shape[0] - 1, take, dtype=int)

        arrays.append(tensors[indices])
        counts[movie_id] = int(take)

    if not arrays:
        raise RuntimeError("No valid tensors were available for training.")

    return np.concatenate(arrays, axis=0), counts


def augment_3d_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Two different realistic views of the same microscopy crop.
    These augmentations provide the self-supervised DINO learning signal.
    """
    out = x.clone()

    # Random flips preserve a cell's identity but alter orientation.
    if torch.rand(()) < 0.5:
        out = torch.flip(out, dims=[2])
    if torch.rand(()) < 0.5:
        out = torch.flip(out, dims=[3])
    if torch.rand(()) < 0.5:
        out = torch.flip(out, dims=[4])

    # Fluorescence intensity and detector noise robustness.
    gain = 0.85 + 0.30 * torch.rand(
        (out.shape[0], 1, 1, 1, 1), device=out.device
    )
    bias = -0.04 + 0.08 * torch.rand(
        (out.shape[0], 1, 1, 1, 1), device=out.device
    )
    noise = 0.025 * torch.randn_like(out)

    return torch.clamp(out * gain + bias + noise, 0.0, 1.0)


class DINOLoss:
    def __init__(
        self,
        n_prototypes: int = 128,
        student_temperature: float = 0.10,
        teacher_temperature: float = 0.04,
        center_momentum: float = 0.90,
    ) -> None:
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.center_momentum = center_momentum
        self.center = torch.zeros(1, n_prototypes)

    def __call__(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        update_center: bool = True,
    ) -> torch.Tensor:
        center = self.center.to(teacher_logits.device)

        teacher_probs = F.softmax(
            (teacher_logits - center) / self.teacher_temperature,
            dim=-1,
        ).detach()

        student_log_probs = F.log_softmax(
            student_logits / self.student_temperature,
            dim=-1,
        )

        loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()

        if update_center:
            batch_center = teacher_logits.detach().mean(dim=0, keepdim=True).cpu()
            self.center = (
                self.center * self.center_momentum
                + batch_center * (1.0 - self.center_momentum)
            )

        return loss


def _mean_loss(
    encoder: nn.Module,
    head: nn.Module,
    teacher_encoder: nn.Module,
    teacher_head: nn.Module,
    tensors: torch.Tensor,
    batch_size: int,
    loss_fn: DINOLoss,
    training: bool,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    if tensors.shape[0] == 0:
        return float("nan")

    permutation = torch.randperm(tensors.shape[0]) if training else torch.arange(tensors.shape[0])
    losses: List[float] = []

    for start in range(0, tensors.shape[0], batch_size):
        idx = permutation[start:start + batch_size]
        batch = tensors[idx]

        view_student = augment_3d_batch(batch)
        view_teacher = augment_3d_batch(batch)

        with torch.no_grad():
            teacher_logits = teacher_head(teacher_encoder(view_teacher))

        student_logits = head(encoder(view_student))
        loss = loss_fn(
            student_logits,
            teacher_logits,
            update_center=training,
        )

        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses))


def train_self_supervised(
    epochs: int = 4,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    source_cells_per_movie: int = 64,
    samples_per_movie: int = 16,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train CellDINO using only local complete movies.

    Recommended first Review-2 run on CPU:
    epochs=4, batch_size=4, source_cells_per_movie=64,
    samples_per_movie=16.
    """
    if not m1.TORCH_OK or m1.ENGINE.model is None:
        raise RuntimeError("PyTorch CellDINO encoder is unavailable.")

    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    movie_ids = discover_complete_movies()
    split = movie_level_split(movie_ids, seed=seed)

    if len(split["train"]) < 1:
        raise RuntimeError("No complete local movie pairs were found.")

    TRAINING_STATE.update(
        {
            "status": "preparing",
            "progress": 0.02,
            "message": "Preparing training and validation cell tensors.",
        }
    )

    train_np, train_counts = load_evenly_sampled_tensors(
        split["train"],
        source_cells_per_movie,
        samples_per_movie,
    )

    val_np = np.zeros((0, *train_np.shape[1:]), dtype=np.float32)
    val_counts: Dict[str, int] = {}

    if split["validation"]:
        val_np, val_counts = load_evenly_sampled_tensors(
            split["validation"],
            source_cells_per_movie,
            samples_per_movie,
        )

    device = torch.device(m1.ENGINE.device)
    student_encoder = m1.ENGINE.model.to(device)
    student_encoder.train()

    student_head = DINOProjectionHead(
        in_dim=m1.CELLDINO_EMBED_DIM,
        out_dim=128,
    ).to(device)

    teacher_encoder = copy.deepcopy(student_encoder).to(device).eval()
    teacher_head = copy.deepcopy(student_head).to(device).eval()

    for parameter in teacher_encoder.parameters():
        parameter.requires_grad = False
    for parameter in teacher_head.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        list(student_encoder.parameters()) + list(student_head.parameters()),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    loss_fn = DINOLoss(n_prototypes=128)
    train_tensor = torch.from_numpy(train_np).float().to(device)
    val_tensor = torch.from_numpy(val_np).float().to(device)

    history: List[Dict[str, float]] = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        TRAINING_STATE.update(
            {
                "status": "training",
                "progress": 0.10 + 0.80 * ((epoch - 1) / epochs),
                "message": f"CellDINO self-supervised training: epoch {epoch}/{epochs}",
            }
        )

        student_encoder.train()
        student_head.train()

        train_loss = _mean_loss(
            student_encoder,
            student_head,
            teacher_encoder,
            teacher_head,
            train_tensor,
            batch_size,
            loss_fn,
            training=True,
            optimizer=optimizer,
        )

        # EMA teacher: slow, stable copy of the student.
        momentum = 0.996 + 0.004 * (epoch / epochs)
        with torch.no_grad():
            for teacher, student in zip(
                teacher_encoder.parameters(),
                student_encoder.parameters(),
            ):
                teacher.data.mul_(momentum).add_(
                    student.data,
                    alpha=1.0 - momentum,
                )

            for teacher, student in zip(
                teacher_head.parameters(),
                student_head.parameters(),
            ):
                teacher.data.mul_(momentum).add_(
                    student.data,
                    alpha=1.0 - momentum,
                )

        student_encoder.eval()
        student_head.eval()

        with torch.no_grad():
            val_loss = (
                _mean_loss(
                    student_encoder,
                    student_head,
                    teacher_encoder,
                    teacher_head,
                    val_tensor,
                    batch_size,
                    loss_fn,
                    training=False,
                )
                if val_tensor.shape[0] > 0
                else float("nan")
            )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": val_loss,
            }
        )

    movie_tag = f"{len(movie_ids)}movies_seed{seed}"
    checkpoint_path = MODEL_DIR / f"celldino_dino_{movie_tag}.pt"
    history_path = MODEL_DIR / f"celldino_dino_{movie_tag}_history.json"

    torch.save(
        {
            "encoder_state_dict": student_encoder.state_dict(),
            "projection_head_state_dict": student_head.state_dict(),
            "split": split,
            "history": history,
            "settings": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "source_cells_per_movie": source_cells_per_movie,
                "samples_per_movie": samples_per_movie,
                "seed": seed,
            },
        },
        checkpoint_path,
    )

    result = {
        "status": "complete",
        "training_method": "self_supervised_dino_student_teacher",
        "complete_movies_found": len(movie_ids),
        "split": split,
        "n_train_samples": int(train_np.shape[0]),
        "n_validation_samples": int(val_np.shape[0]),
        "train_samples_by_movie": train_counts,
        "validation_samples_by_movie": val_counts,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "held_out_test_note": (
            "Test movies were excluded from model training and validation. "
            "They remain reserved for later tracking evaluation."
        ),
    }

    history_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    TRAINING_STATE.update(
        {
            "status": "complete",
            "progress": 1.0,
            "message": "CellDINO training complete. Checkpoint saved.",
            "result": result,
        }
    )

    return result


def training_status() -> Dict[str, Any]:
    return dict(TRAINING_STATE)