"""
Module 0 — Data Preparation & Preprocessing for CellTrace.

Pipeline:
  1) Dataset discovery & validation (Zarr hierarchy + GEFF presence)
  2) Physical anisotropy checks (Z=1.625 µm, Y/X=0.40625 µm)
  3) Variable-sized 3D cell patch extraction
  4) 1-99% percentile clipping + intensity normalization
  5) Standardized cell tensor export
  6) 2D optical-slice rendering (base64) for the React viewport
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from config import (
    DEFAULT_PATCH_MARGIN_UM,
    LOCAL_DATA_ROOT,
    MAX_PATCH_SIZE_ZYX,
    MIN_PATCH_SIZE_ZYX,
    PERCENTILE_HIGH,
    PERCENTILE_LOW,
    SLICE_JPEG_QUALITY,
    TARGET_TENSOR_SIZE_ZYX,
    TENSOR_DIR,
    ensure_artifact_dirs,
    spacing_dict,
)
from utils.io_geff import CellNode, GeffGraph, find_geff_files, load_geff
from utils.io_zarr import (
    ZarrIOError,
    ZarrVolumeInfo,
    dump_hierarchy,
    find_zarr_stores,
    inspect_zarr,
    read_crop_zyx,
    read_slice_yx,
    volume_bounds,
    write_meta_sidecar,
)
from utils.spacing import (
    DEFAULT_SPACING,
    VoxelSpacing,
    calibrate_bbox_zyx,
    clamp_bbox,
    physical_volume_um3,
    summarize_spacing,
)

LOGGER = logging.getLogger("celltrace.module0")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class DatasetRecord:
    movie_id: str
    movie_dir: Path
    zarr_path: Optional[Path] = None
    geff_path: Optional[Path] = None
    volume: Optional[ZarrVolumeInfo] = None
    bounds: Dict[str, int] = field(default_factory=dict)
    spacing: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    loaded_at: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "movie_id": self.movie_id,
            "movie_dir": str(self.movie_dir),
            "zarr_path": str(self.zarr_path) if self.zarr_path else None,
            "geff_path": str(self.geff_path) if self.geff_path else None,
            "volume": self.volume.as_dict() if self.volume else None,
            "bounds": self.bounds,
            "spacing": self.spacing,
            "validation": self.validation,
            "loaded_at": self.loaded_at,
            "dimensions": {
                "t": self.bounds.get("t_max"),
                "z": self.bounds.get("z_max"),
                "y": self.bounds.get("y_max"),
                "x": self.bounds.get("x_max"),
            },
        }


@dataclass
class CellPatch:
    movie_id: str
    node_id: int
    track_id: int
    t: int
    center_zyx: Tuple[float, float, float]
    bbox_start: Tuple[int, int, int]
    bbox_end: Tuple[int, int, int]
    raw_shape: Tuple[int, int, int]
    tensor_shape: Tuple[int, int, int]
    stats: Dict[str, float] = field(default_factory=dict)
    tensor_path: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineState:
    movie_id: Optional[str] = None
    step: str = "idle"
    progress: float = 0.0
    message: str = ""
    errors: List[str] = field(default_factory=list)
    n_patches: int = 0
    n_tensors: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------
# Add these helpers inside DataPreparationEngine

class DataPreparationEngine:
        # ------------------------------------------------------------------
    # Dataset Explorer — raw Zarr + GEFF inspection
    # ------------------------------------------------------------------
    def inspect_raw_dataset(
        self,
        movie_id: str,
        t: int = 0,
        z: int = 0,
    ) -> Dict[str, Any]:
        """Return real, presentation-friendly Zarr and GEFF metadata."""
        record = self.get_dataset(movie_id)

        if record.volume is None:
            raise RuntimeError("Volume is not loaded")

        graph = self._ensure_graph(movie_id)

        t = int(np.clip(t, 0, record.bounds["t_max"] - 1))
        z = int(np.clip(z, 0, record.bounds["z_max"] - 1))

        plane = read_slice_yx(record.volume, t=t, z=z)
        sample = np.asarray(plane[:4, :6])

        nodes = [
            {
                "node_id": int(node.node_id),
                "t": int(node.t),
                "z": float(node.z),
                "y": float(node.y),
                "x": float(node.x),
            }
            for node in graph.nodes
        ]

        return {
            "movie_id": movie_id,
            "t": t,
            "z": z,
            "zarr": {
                "path": str(record.zarr_path),
                "array_key": record.volume.array_key,
                "shape_tzyx": list(record.volume.shape),
                "dtype": record.volume.dtype,
                "chunks": list(record.volume.chunks)
                if record.volume.chunks is not None
                else None,
                "raw_intensity_sample": sample.astype(int).tolist(),
                "meaning": (
                    "A 4-D microscopy array: time × depth × height × width. "
                    "Each value is fluorescence intensity."
                ),
            },
            "geff": {
                "path": str(record.geff_path),
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "sample_nodes": nodes[:8],
                "sample_edges": [
                    {"source": int(source), "target": int(target)}
                    for source, target in graph.edges[:8]
                ],
                "meaning": (
                    "A graph annotation: each node stores one cell observation "
                    "at one time and position; each edge links related observations."
                ),
            },
            "nodes": nodes,
            "edges": [
                {"source": int(source), "target": int(target)}
                for source, target in graph.edges
            ],
        }

    def render_geff_overlay(
        self,
        movie_id: str,
        t: int = 0,
        z: int = 0,
    ) -> Dict[str, Any]:
        """Render the real Zarr slice with GEFF centres and patch boxes."""
        record = self.get_dataset(movie_id)

        if record.volume is None:
            raise RuntimeError("Volume is not loaded")

        graph = self._ensure_graph(movie_id)

        t = int(np.clip(t, 0, record.bounds["t_max"] - 1))
        z = int(np.clip(z, 0, record.bounds["z_max"] - 1))

        plane = read_slice_yx(record.volume, t=t, z=z)
        clipped, percentiles = self.percentile_clip(plane)
        normalized = self.normalize_intensity(
            clipped,
            percentiles["p_low"],
            percentiles["p_high"],
        )

        u8 = (normalized * 255.0).astype(np.uint8)
        rgb = np.stack([u8, u8, u8], axis=-1)
        image = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(image)

        visible_nodes = []

        shape_zyx = (
            int(record.bounds["z_max"]),
            int(record.bounds["y_max"]),
            int(record.bounds["x_max"]),
        )

        for node in graph.nodes_at_time(t):
            # Show only annotations whose patch intersects this Z slice.
            if abs(int(round(node.z)) - z) > 2:
                continue

            start, end = self._patch_bounds_for_node(
                node=node,
                shape_zyx=shape_zyx,
            )

            left = int(start[2])
            top = int(start[1])
            right = int(end[2] - 1)
            bottom = int(end[1] - 1)

            # Yellow = extracted patch boundary
            draw.rectangle(
                [left, top, right, bottom],
                outline=(255, 215, 0),
                width=2,
            )

            # Red = GEFF cell centre
            cx = int(round(node.x))
            cy = int(round(node.y))
            draw.ellipse(
                [cx - 4, cy - 4, cx + 4, cy + 4],
                fill=(255, 70, 70),
                outline=(255, 255, 255),
            )

            visible_nodes.append(
                {
                    "node_id": int(node.node_id),
                    "t": int(node.t),
                    "z": float(node.z),
                    "y": float(node.y),
                    "x": float(node.x),
                }
            )

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        return {
            "movie_id": movie_id,
            "t": t,
            "z": z,
            "mime": "image/png",
            "image_base64": base64.b64encode(
                buffer.getvalue()
            ).decode("ascii"),
            "visible_count": len(visible_nodes),
            "visible_nodes": visible_nodes,
        }
    @staticmethod
    def _image_base64(image: np.ndarray) -> str:
        """Render a 2-D float/uint image as a base64 PNG."""
        image = np.asarray(image, dtype=np.float32)
        finite = image[np.isfinite(image)]

        if finite.size == 0:
            image = np.zeros_like(image, dtype=np.float32)
        else:
            lo, hi = np.percentile(finite, (1, 99))
            if hi <= lo:
                image = np.zeros_like(image, dtype=np.float32)
            else:
                image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)

        u8 = (image * 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(u8, mode="L").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


    @staticmethod
    def _pad_center_crop(
        crop: np.ndarray,
        target_shape: Sequence[int],
    ) -> np.ndarray:
        """Centre-pad a Z,Y,X crop to its requested shape."""
        target_z, target_y, target_x = map(int, target_shape)
        out = np.zeros((target_z, target_y, target_x), dtype=crop.dtype)

        copy_z = min(crop.shape[0], target_z)
        copy_y = min(crop.shape[1], target_y)
        copy_x = min(crop.shape[2], target_x)

        src_z = max(0, (crop.shape[0] - copy_z) // 2)
        src_y = max(0, (crop.shape[1] - copy_y) // 2)
        src_x = max(0, (crop.shape[2] - copy_x) // 2)

        dst_z = max(0, (target_z - copy_z) // 2)
        dst_y = max(0, (target_y - copy_y) // 2)
        dst_x = max(0, (target_x - copy_x) // 2)

        out[
            dst_z:dst_z + copy_z,
            dst_y:dst_y + copy_y,
            dst_x:dst_x + copy_x,
        ] = crop[
            src_z:src_z + copy_z,
            src_y:src_y + copy_y,
            src_x:src_x + copy_x,
        ]
        return out


    def _read_node_patch(
        self,
        movie_id: str,
        node: CellNode,
        margin_um: float = DEFAULT_PATCH_MARGIN_UM,
    ) -> Tuple[np.ndarray, Tuple[int, int, int], Tuple[int, int, int]]:
        """Read one real Z,Y,X cell-centred patch from the source Zarr volume."""
        record = self.get_dataset(movie_id)
        if record.volume is None:
            raise RuntimeError("Dataset volume is not loaded")

        shape_zyx = (
            int(record.bounds["z_max"]),
            int(record.bounds["y_max"]),
            int(record.bounds["x_max"]),
        )

        start, end = self._patch_bounds_for_node(
            node=node,
            shape_zyx=shape_zyx,
            margin_um=margin_um,
        )

        crop = read_crop_zyx(
            record.volume,
            t=int(node.t),
            z0=int(start[0]),
            z1=int(end[0]),
            y0=int(start[1]),
            y1=int(end[1]),
            x0=int(start[2]),
            x1=int(end[2]),
        )

        if crop.ndim != 3 or crop.size == 0:
            raise RuntimeError(
                f"Empty patch for node {node.node_id}; "
                f"bbox={start}->{end}, crop_shape={crop.shape}"
            )

        return np.asarray(crop), start, end
    def __init__(
        self,
        data_root: Path = LOCAL_DATA_ROOT,
        tensor_dir: Path = TENSOR_DIR,
        spacing: VoxelSpacing = DEFAULT_SPACING,
    ) -> None:
        ensure_artifact_dirs()
        self.data_root = Path(data_root)
        self.tensor_dir = Path(tensor_dir)
        self.spacing = spacing
        self._cache: Dict[str, DatasetRecord] = {}
        self._graphs: Dict[str, GeffGraph] = {}
        self._patches: Dict[str, List[CellPatch]] = {}
        self.state = PipelineState()
        self.tensor_dir.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 0.1 Dataset discovery / loading
    # ------------------------------------------------------------------
    def list_movies(self) -> List[Dict[str, Any]]:
        if not self.data_root.exists():
            return []

        movies: List[Dict[str, Any]] = []

        for zarr_path in sorted(self.data_root.glob("*.zarr")):
            movie_id = zarr_path.name.removesuffix(".zarr")
            geff_path = self.data_root / f"{movie_id}.geff"

            movies.append(
                {
                    "movie_id": movie_id,
                    "path": str(self.data_root),
                    "has_zarr": zarr_path.is_dir() and (zarr_path / "zarr.json").exists(),
                    "has_geff": geff_path.is_dir() and (geff_path / "zarr.json").exists(),
                    "n_zarr": 1,
                    "n_geff": 1 if geff_path.exists() else 0,
                    "zarr_candidates": [str(zarr_path)],
                    "geff_candidates": [str(geff_path)] if geff_path.exists() else [],
                }
            )

        return movies

    def resolve_movie_dir(self, movie_id: str) -> Path:
        candidate = self.data_root / movie_id
        if candidate.exists():
            return candidate
        zarr_candidate = self.data_root / f"{movie_id}.zarr"
        if zarr_candidate.exists():
            return zarr_candidate
        # fuzzy match
        for child in self.data_root.iterdir():
            if movie_id.lower() in child.name.lower():
                return child
        raise FileNotFoundError(
            f"Movie '{movie_id}' not found under {self.data_root}. "
            "Paste your dataset folders into backend/local_data/train/."
        )

    def load_dataset(self, movie_id: str) -> DatasetRecord:
    # """
    # Fast loader for a paired flat Zarr + GEFF movie.
    # """
        self.state = PipelineState(
            movie_id=movie_id,
            step="0.1_load",
            progress=0.10,
            message="Resolving Zarr and GEFF paths",
        )

        zarr_path = self.data_root / f"{movie_id}.zarr"
        geff_path = self.data_root / f"{movie_id}.geff"

        if not zarr_path.is_dir() or not (zarr_path / "zarr.json").exists():
            raise FileNotFoundError(
                f"Zarr movie '{movie_id}.zarr' was not found under {self.data_root}"
            )

        if not geff_path.is_dir() or not (geff_path / "zarr.json").exists():
            geff_path = None

        self.state.progress = 0.35
        self.state.message = f"Opening microscopy movie: {zarr_path.name}"

        volume = inspect_zarr(zarr_path)
        bounds = volume_bounds(volume)

        record = DatasetRecord(
            movie_id=movie_id,
            movie_dir=self.data_root,
            zarr_path=zarr_path,
            geff_path=geff_path,
            volume=volume,
            bounds=bounds,
            spacing=spacing_dict(),
            loaded_at=time.time(),
        )

        self._cache[movie_id] = record

        if geff_path is not None:
            try:
                self._graphs[movie_id] = load_geff(geff_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("GEFF load deferred for %s: %s", movie_id, exc)

        write_meta_sidecar(
            volume,
            self.tensor_dir / movie_id / "zarr_meta.json",
        )

        self.state.progress = 1.0
        self.state.message = "Dataset loaded"
        self.state.step = "0.1_done"

        return record

    def get_dataset(self, movie_id: str) -> DatasetRecord:
        if movie_id not in self._cache:
            return self.load_dataset(movie_id)
        return self._cache[movie_id]

    # ------------------------------------------------------------------
    # 0.2 Validation
    # ------------------------------------------------------------------
    def validate_dataset(self, movie_id: str) -> Dict[str, Any]:
        record = self.get_dataset(movie_id)
        self.state = PipelineState(
            movie_id=movie_id, step="0.2_validate", progress=0.2, message="Validating volume"
        )
        issues: List[str] = []
        warnings: List[str] = []
        checks: Dict[str, Any] = {}

        if record.volume is None:
            issues.append("Volume metadata missing")
        else:
            shape = record.volume.shape
            checks["ndim"] = len(shape)
            checks["shape"] = list(shape)
            checks["dtype"] = record.volume.dtype
            if len(shape) < 3:
                issues.append(f"Expected >=3D volume, got shape {shape}")
            if any(s <= 0 for s in shape):
                issues.append("Non-positive dimension detected")
            prod = int(np.prod(shape))
            checks["n_voxels"] = prod
            if prod > 2_000_000_000:
                warnings.append("Very large volume; prefer lazy slice access only")

            hierarchy = dump_hierarchy(record.zarr_path) if record.zarr_path else {}
            checks["hierarchy"] = hierarchy

            # Sample a center slice for finite intensity check
            try:
                b = record.bounds
                t_mid = max(0, (b.get("t_max", 1) // 2) - 1)
                z_mid = max(0, (b.get("z_max", 1) // 2) - 1)
                sample = read_slice_yx(record.volume, t=t_mid, z=z_mid)
                finite = bool(np.isfinite(sample).all())
                checks["sample_finite"] = finite
                checks["sample_min"] = float(np.nanmin(sample))
                checks["sample_max"] = float(np.nanmax(sample))
                checks["sample_mean"] = float(np.nanmean(sample))
                if not finite:
                    issues.append("Non-finite intensities in sample slice")
                if np.allclose(sample, sample.flat[0]):
                    warnings.append("Sample slice appears constant (blank?)")
            except Exception as exc:  # noqa: BLE001
                issues.append(f"Slice sample failed: {exc}")

        spacing_report = summarize_spacing(self.spacing)
        checks["spacing"] = spacing_report
        if not spacing_report["ok"]:
            warnings.extend(spacing_report["issues"])

        if record.geff_path is None:
            warnings.append("No GEFF annotation found — patch extraction will be limited")
        else:
            checks["geff_path"] = str(record.geff_path)
            if movie_id not in self._graphs:
                try:
                    self._graphs[movie_id] = load_geff(record.geff_path)
                except Exception as exc:  # noqa: BLE001
                    issues.append(f"GEFF parse failed: {exc}")
            if movie_id in self._graphs:
                g = self._graphs[movie_id]
                checks["geff"] = g.as_dict()
                if len(g.nodes) == 0:
                    issues.append("GEFF contains zero nodes")

        phys = None
        if record.volume is not None and len(record.volume.shape) >= 3:
            # Approximate physical volume using trailing ZYX
            zyx = record.volume.shape[-3:]
            phys = physical_volume_um3(zyx, self.spacing)
            checks["approx_physical_volume_um3"] = phys

        ok = len(issues) == 0
        report = {
            "movie_id": movie_id,
            "ok": ok,
            "issues": issues,
            "warnings": warnings,
            "checks": checks,
            "spacing_um": spacing_dict(),
        }
        record.validation = report
        self._cache[movie_id] = record
        self.state.progress = 1.0
        self.state.step = "0.2_done"
        self.state.message = "Validation complete" if ok else "Validation found issues"
        self.state.errors = issues
        return report

    # ------------------------------------------------------------------
    # Intensity standardization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def percentile_clip(
        volume: np.ndarray,
        low: float = PERCENTILE_LOW,
        high: float = PERCENTILE_HIGH,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        flat = volume.astype(np.float64).ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return volume.astype(np.float32), {"p_low": 0.0, "p_high": 1.0}
        p_low = float(np.percentile(flat, low))
        p_high = float(np.percentile(flat, high))
        if p_high <= p_low:
            p_high = p_low + 1e-6
        clipped = np.clip(volume.astype(np.float64), p_low, p_high)
        return clipped.astype(np.float32), {"p_low": p_low, "p_high": p_high}

    @staticmethod
    def normalize_intensity(volume: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
        scale = max(p_high - p_low, 1e-6)
        out = (volume.astype(np.float64) - p_low) / scale
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def resize_zyx(volume: np.ndarray, target: Sequence[int]) -> np.ndarray:
        """Trilinear-ish resize via zoom (scipy) with numpy fallback."""
        target = tuple(int(t) for t in target)
        if volume.shape == target:
            return volume.astype(np.float32)
        try:
            from scipy.ndimage import zoom

            factors = [t / s for t, s in zip(target, volume.shape)]
            out = zoom(volume.astype(np.float32), factors, order=1)
            # zoom can be off-by-one; crop/pad
            out = DataPreparationEngine._fit_shape(out, target)
            return out.astype(np.float32)
        except Exception:
            return DataPreparationEngine._nn_resize(volume, target)

    @staticmethod
    def _fit_shape(arr: np.ndarray, target: Sequence[int]) -> np.ndarray:
        target = tuple(target)
        out = np.zeros(target, dtype=arr.dtype)
        slices_src = []
        slices_dst = []
        for a, t in zip(arr.shape, target):
            n = min(a, t)
            slices_src.append(slice(0, n))
            slices_dst.append(slice(0, n))
        out[tuple(slices_dst)] = arr[tuple(slices_src)]
        return out

    @staticmethod
    def _nn_resize(volume: np.ndarray, target: Sequence[int]) -> np.ndarray:
        zz = np.linspace(0, volume.shape[0] - 1, target[0])
        yy = np.linspace(0, volume.shape[1] - 1, target[1])
        xx = np.linspace(0, volume.shape[2] - 1, target[2])
        zi = np.clip(np.round(zz).astype(int), 0, volume.shape[0] - 1)
        yi = np.clip(np.round(yy).astype(int), 0, volume.shape[1] - 1)
        xi = np.clip(np.round(xx).astype(int), 0, volume.shape[2] - 1)
        return volume[zi][:, yi][:, :, xi].astype(np.float32)

    # ------------------------------------------------------------------
    # 0.3 Cell patch extraction
    # ------------------------------------------------------------------
    def _ensure_graph(self, movie_id: str) -> GeffGraph:
        if movie_id in self._graphs:
            return self._graphs[movie_id]
        record = self.get_dataset(movie_id)
        if record.geff_path is None:
            raise FileNotFoundError(f"No GEFF for movie {movie_id}")
        graph = load_geff(record.geff_path)
        self._graphs[movie_id] = graph
        return graph

    def _patch_bounds_for_node(
        self,
        node: CellNode,
        shape_zyx: Sequence[int],
        margin_um: float = DEFAULT_PATCH_MARGIN_UM,
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        half = (margin_um, margin_um, margin_um)
        # Prefer slightly larger XY extent due to finer XY resolution
        half = (margin_um * 1.2, margin_um, margin_um)
        start, end = calibrate_bbox_zyx(node.zyx, half, self.spacing)
        start, end = clamp_bbox(start, end, shape_zyx)

        # Enforce min/max patch sizes by expanding/shrinking around center
        cz, cy, cx = [int(round(v)) for v in node.zyx]
        sizes = [end[i] - start[i] for i in range(3)]
        for i, (mn, mx) in enumerate(zip(MIN_PATCH_SIZE_ZYX, MAX_PATCH_SIZE_ZYX)):
            if sizes[i] < mn:
                deficit = mn - sizes[i]
                start_list = list(start)
                end_list = list(end)
                start_list[i] -= deficit // 2
                end_list[i] += deficit - deficit // 2
                start, end = clamp_bbox(start_list, end_list, shape_zyx)
                sizes[i] = end[i] - start[i]
            if sizes[i] > mx:
                center = (start[i] + end[i]) // 2
                start_list = list(start)
                end_list = list(end)
                start_list[i] = max(0, center - mx // 2)
                end_list[i] = min(shape_zyx[i], start_list[i] + mx)
                start, end = clamp_bbox(start_list, end_list, shape_zyx)
        # Keep center roughly inside
        _ = (cz, cy, cx)
        return start, end

    def extract_patches(
    self,
    movie_id: str,
    max_cells: Optional[int] = 256,
    time_filter: Optional[Sequence[int]] = None,
    margin_um: float = DEFAULT_PATCH_MARGIN_UM,
    persist_raw: bool = True,
) -> Dict[str, Any]:
        """
        0.3: Extract real 3-D cell-centred patches from the Zarr volume.

        Saves each raw patch as .npy and returns a patch preview image for the UI.
        """
        self.get_dataset(movie_id)
        graph = self._ensure_graph(movie_id)

        nodes = list(graph.nodes)
        if time_filter is not None:
            allowed = {int(t) for t in time_filter}
            nodes = [node for node in nodes if int(node.t) in allowed]

        if max_cells is not None:
            nodes = nodes[: int(max_cells)]

        raw_dir = self.tensor_dir / movie_id / "raw_patches"
        raw_dir.mkdir(parents=True, exist_ok=True)

        patches: List[CellPatch] = []
        failures: List[Dict[str, Any]] = []
        previews: List[Dict[str, Any]] = []

        self.state = PipelineState(
            movie_id=movie_id,
            step="0.3_extract",
            progress=0.0,
            message="Extracting real cell-centred Zarr patches",
        )

        for index, node in enumerate(nodes):
            try:
                crop, start, end = self._read_node_patch(
                    movie_id=movie_id,
                    node=node,
                    margin_um=margin_um,
                )

                raw_path = raw_dir / f"node_{int(node.node_id)}_t_{int(node.t)}.npy"
                if persist_raw:
                    np.save(raw_path, crop.astype(np.float32))

                patch = CellPatch(
                    movie_id=movie_id,
                    node_id=int(node.node_id),
                    track_id=int(node.track_id),
                    t=int(node.t),
                    center_zyx=(
                        float(node.z),
                        float(node.y),
                        float(node.x),
                    ),
                    bbox_start=tuple(map(int, start)),
                    bbox_end=tuple(map(int, end)),
                    raw_shape=tuple(map(int, crop.shape)),
                    tensor_shape=tuple(map(int, TARGET_TENSOR_SIZE_ZYX)),
                    stats={
                        "min": float(np.min(crop)),
                        "max": float(np.max(crop)),
                        "mean": float(np.mean(crop)),
                        "std": float(np.std(crop)),
                    },
                    tensor_path=str(raw_path),
                )
                patches.append(patch)

                # MIP = real extracted patch preview for your React screen.
                if len(previews) < 12:
                    previews.append(
                        {
                            **patch.as_dict(),
                            "preview_projection": "max_intensity_projection_z",
                            "patch_image_base64": self._image_base64(
                                np.max(crop, axis=0)
                            ),
                            "mime": "image/png",
                        }
                    )

            except Exception as exc:
                failures.append(
                    {
                        "node_id": int(node.node_id),
                        "track_id": int(node.track_id),
                        "error": str(exc),
                    }
                )

            self.state.progress = (index + 1) / max(len(nodes), 1)
            self.state.message = f"Extracted {len(patches)} / {len(nodes)} patches"
            self.state.n_patches = len(patches)

        self._patches[movie_id] = patches

        manifest_path = self.tensor_dir / movie_id / "patch_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps([patch.as_dict() for patch in patches], indent=2),
            encoding="utf-8",
        )

        self.state.step = "0.3_done"
        self.state.progress = 1.0

        return {
            "movie_id": movie_id,
            "n_requested": len(nodes),
            "n_extracted": len(patches),
            "n_failed": len(failures),
            "failures": failures[:50],
            "manifest": str(manifest_path),
            "patches_preview": previews,
        }

    # ------------------------------------------------------------------
    # 0.4 / 0.5 Standardization + tensor generation
    # ------------------------------------------------------------------
    def generate_tensors(
    self,
    movie_id: str,
    max_cells: Optional[int] = 256,
    target_size: Sequence[int] = TARGET_TENSOR_SIZE_ZYX,
    batch_write: int = 32,
) -> Dict[str, Any]:
        """
        0.5: Convert real extracted patches to normalized PyTorch tensors.

        Output shape per cell: (1, Z, Y, X)
        Combined output shape: (N, 1, Z, Y, X)
        """
        self.get_dataset(movie_id)

        if movie_id not in self._patches or not self._patches[movie_id]:
            self.extract_patches(movie_id, max_cells=max_cells)

        patches = self._patches[movie_id]
        if max_cells is not None:
            patches = patches[: int(max_cells)]

        graph = self._ensure_graph(movie_id)
        node_map = {int(node.node_id): node for node in graph.nodes}

        output_dir = self.tensor_dir / movie_id / "cell_tensors"
        output_dir.mkdir(parents=True, exist_ok=True)

        all_tensors: List[torch.Tensor] = []
        manifest: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []

        self.state = PipelineState(
            movie_id=movie_id,
            step="0.5_tensorize",
            progress=0.0,
            message="Generating normalized PyTorch tensors",
        )

        for index, patch in enumerate(patches):
            try:
                node = node_map.get(int(patch.node_id))
                if node is None:
                    raise RuntimeError(f"Node {patch.node_id} no longer exists in GEFF")

                crop, start, end = self._read_node_patch(movie_id, node)

                clipped, percentiles = self.percentile_clip(crop)
                normalized = self.normalize_intensity(
                    clipped,
                    percentiles["p_low"],
                    percentiles["p_high"],
                )

                source = torch.from_numpy(normalized).float()
                tensor = F.interpolate(
                    source.unsqueeze(0).unsqueeze(0),
                    size=tuple(map(int, target_size)),
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(0).contiguous()

                # Final safeguards.
                tensor = torch.nan_to_num(
                    tensor,
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                ).clamp_(0.0, 1.0)

                tensor_path = output_dir / (
                    f"node_{int(patch.node_id)}_t_{int(patch.t)}.pt"
                )

                torch.save(
                    {
                        # Batch size, color channel, depth, heiight, width
                        "tensor": tensor,  # (1, Z, Y, X)
                        "node_id": int(patch.node_id),
                        "track_id": int(patch.track_id),
                        "t": int(patch.t),
                        "center_zyx": list(patch.center_zyx),
                        "bbox_start": list(start),
                        "bbox_end": list(end),
                        "raw_shape": list(crop.shape),
                        "percentiles": percentiles,
                        "spacing_um": spacing_dict(),
                    },
                    tensor_path,
                )

                all_tensors.append(tensor)
                manifest.append(
                    {
                        "node_id": int(patch.node_id),
                        "track_id": int(patch.track_id),
                        "t": int(patch.t),
                        "tensor_path": str(tensor_path),
                        "tensor_shape": list(tensor.shape),
                        "raw_shape": list(crop.shape),
                        "min": float(tensor.min().item()),
                        "max": float(tensor.max().item()),
                        "mean": float(tensor.mean().item()),
                        "std": float(tensor.std().item()),
                        "p_low": float(percentiles["p_low"]),
                        "p_high": float(percentiles["p_high"]),
                    }
                )

            except Exception as exc:
                failures.append(
                    {
                        "node_id": int(patch.node_id),
                        "error": str(exc),
                    }
                )

            self.state.progress = (index + 1) / max(len(patches), 1)
            self.state.message = f"Generated {len(all_tensors)} tensors"
            self.state.n_tensors = len(all_tensors)

        combined_path = None
        if all_tensors:
            combined = torch.stack(all_tensors, dim=0).cpu()

            # Native PyTorch bundle.
            combined_path = output_dir / "cell_tensors.pt"
            torch.save(
                {
                    "tensors": combined,
                    "movie_id": movie_id,
                    "target_shape_czyx": list(combined.shape[1:]),
                    "node_ids": [row["node_id"] for row in manifest],
                    "track_ids": [row["track_id"] for row in manifest],
                    "times": [row["t"] for row in manifest],
                    "spacing_um": spacing_dict(),
                },
                combined_path,
            )

            # Compatibility bundle used by Module 1.
            npz_path = output_dir / "cell_tensors.npz"
            np.savez_compressed(
                npz_path,
                tensors=combined.numpy().astype(np.float32),
                node_ids=np.asarray(
                    [row["node_id"] for row in manifest],
                    dtype=np.int64,
                ),
                track_ids=np.asarray(
                    [row["track_id"] for row in manifest],
                    dtype=np.int64,
                ),
                times=np.asarray(
                    [row["t"] for row in manifest],
                    dtype=np.int64,
                ),
            )

        else:
            combined_path = None
            npz_path = None

        manifest_path = output_dir / "tensor_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        self.state.step = "0.5_done"
        self.state.progress = 1.0
        self.state.message = f"Generated {len(all_tensors)} valid tensors"

        return {
            "movie_id": movie_id,
            "n_tensors": len(all_tensors),
            "n_failed": len(failures),
            "failures": failures[:50],
            "tensor_shape": (
                list(all_tensors[0].shape) if all_tensors else None
            ),
            "combined_tensor_path": str(combined_path) if combined_path else None,
            "npz_path": str(npz_path) if npz_path else None,
            "manifest_path": str(manifest_path),
            "preview": manifest[:10],
        }

    # ------------------------------------------------------------------
    # Slice rendering for UI
    # ------------------------------------------------------------------
    def render_slice_base64(
        self,
        movie_id: str,
        t: int = 0,
        z: int = 0,
        channel: int = 0,
        colormap: str = "cyan",
    ) -> Dict[str, Any]:
        record = self.get_dataset(movie_id)
        if record.volume is None:
            raise RuntimeError("Volume not loaded")
        plane = read_slice_yx(record.volume, t=t, z=z, channel=channel)
        clipped, pr = self.percentile_clip(plane)
        norm = self.normalize_intensity(clipped, pr["p_low"], pr["p_high"])
        u8 = (norm * 255.0).astype(np.uint8)
        if colormap == "gray":
            img = Image.fromarray(u8, mode="L")
        else:
            # Cyan/electric microscopy tint for dark UI
            rgb = np.zeros((u8.shape[0], u8.shape[1], 3), dtype=np.uint8)
            rgb[..., 1] = u8  # G
            rgb[..., 2] = u8  # B
            rgb[..., 0] = (u8.astype(np.uint16) * 40 // 255).astype(np.uint8)
            img = Image.fromarray(rgb, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=SLICE_JPEG_QUALITY)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return {
            "movie_id": movie_id,
            "t": int(t),
            "z": int(z),
            "channel": int(channel),
            "shape": list(plane.shape),
            "p_low": pr["p_low"],
            "p_high": pr["p_high"],
            "mime": "image/jpeg",
            "image_base64": b64,
            "bounds": record.bounds,
        }

    def standardize_only(self, movie_id: str, t: int, z: int) -> Dict[str, Any]:
        """
        0.4: Show real raw slice, normalized slice, and nearest real cell patch.
        """
        record = self.get_dataset(movie_id)
        if record.volume is None:
            raise RuntimeError("Volume not loaded")

        t = int(np.clip(t, 0, record.bounds["t_max"] - 1))
        z = int(np.clip(z, 0, record.bounds["z_max"] - 1))

        plane = read_slice_yx(record.volume, t=t, z=z)
        clipped, percentiles = self.percentile_clip(plane)
        normalized = self.normalize_intensity(
            clipped,
            percentiles["p_low"],
            percentiles["p_high"],
        )

        graph = self._ensure_graph(movie_id)
        frame_nodes = [node for node in graph.nodes if int(node.t) == t]

        nearest_patch = None
        if frame_nodes:
            nearest = min(
                frame_nodes,
                key=lambda node: abs(float(node.z) - float(z)),
            )
            crop, start, end = self._read_node_patch(movie_id, nearest)

            patch_clipped, patch_percentiles = self.percentile_clip(crop)
            patch_normalized = self.normalize_intensity(
                patch_clipped,
                patch_percentiles["p_low"],
                patch_percentiles["p_high"],
            )

            nearest_patch = {
                "node_id": int(nearest.node_id),
                "track_id": int(nearest.track_id),
                "center_zyx": [
                    float(nearest.z),
                    float(nearest.y),
                    float(nearest.x),
                ],
                "bbox_start": list(start),
                "bbox_end": list(end),
                "raw_shape": list(crop.shape),
                "patch_image_base64": self._image_base64(np.max(crop, axis=0)),
                "normalized_patch_image_base64": self._image_base64(
                    np.max(patch_normalized, axis=0)
                ),
                "mime": "image/png",
            }

        return {
            "movie_id": movie_id,
            "t": t,
            "z": z,
            "shape": list(plane.shape),
            "raw": {
                "min": float(np.min(plane)),
                "max": float(np.max(plane)),
                "mean": float(np.mean(plane)),
                "std": float(np.std(plane)),
            },
            "standardization": {
                "percentile_low": float(PERCENTILE_LOW),
                "percentile_high": float(PERCENTILE_HIGH),
                "p_low": float(percentiles["p_low"]),
                "p_high": float(percentiles["p_high"]),
            },
            "normalized": {
                "min": float(np.min(normalized)),
                "max": float(np.max(normalized)),
                "mean": float(np.mean(normalized)),
                "std": float(np.std(normalized)),
            },
            "raw_image_base64": self._image_base64(plane),
            "standardized_image_base64": self._image_base64(normalized),
            "histogram": np.histogram(
                normalized,
                bins=32,
                range=(0.0, 1.0),
            )[0].tolist(),
            "nearest_cell_patch": nearest_patch,
            "mime": "image/png",
        }

    def run_full_preprocess(
        self,
        movie_id: str,
        max_cells: Optional[int] = 256,
    ) -> Dict[str, Any]:
        loaded = self.load_dataset(movie_id).as_dict()
        validation = self.validate_dataset(movie_id)
        extraction = self.extract_patches(movie_id, max_cells=max_cells)
        tensors = self.generate_tensors(movie_id, max_cells=max_cells)
        return {
            "movie_id": movie_id,
            "loaded": loaded,
            "validation": validation,
            "extraction": extraction,
            "tensors": tensors,
            "state": self.state.as_dict(),
        }

    def status(self) -> Dict[str, Any]:
        return {
            "state": self.state.as_dict(),
            "cached_movies": list(self._cache.keys()),
            "data_root": str(self.data_root),
            "tensor_dir": str(self.tensor_dir),
        }

    def fingerprint_movie(self, movie_id: str) -> str:
        record = self.get_dataset(movie_id)
        raw = f"{movie_id}|{record.zarr_path}|{record.geff_path}|{record.bounds}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


# Singleton used by API layer
ENGINE = DataPreparationEngine()


def list_datasets() -> List[Dict[str, Any]]:
    return ENGINE.list_movies()


def load_movie(movie_id: str) -> Dict[str, Any]:
    return ENGINE.load_dataset(movie_id).as_dict()


def validate_movie(movie_id: str) -> Dict[str, Any]:
    return ENGINE.validate_dataset(movie_id)


def extract_cell_patches(movie_id: str, max_cells: Optional[int] = None) -> Dict[str, Any]:
    return ENGINE.extract_patches(movie_id, max_cells=max_cells)


def standardize_slice(movie_id: str, t: int, z: int) -> Dict[str, Any]:
    return ENGINE.standardize_only(movie_id, t, z)


def tensorize_movie(movie_id: str, max_cells: Optional[int] = 256) -> Dict[str, Any]:
    return ENGINE.generate_tensors(movie_id, max_cells=max_cells)


def get_slice(movie_id: str, t: int, z: int) -> Dict[str, Any]:
    return ENGINE.render_slice_base64(movie_id, t=t, z=z)

def inspect_dataset(movie_id: str, t: int = 0, z: int = 0) -> Dict[str, Any]:
    return ENGINE.inspect_raw_dataset(movie_id, t=t, z=z)

def get_geff_overlay(movie_id: str, t: int = 0, z: int = 0) -> Dict[str, Any]:
    return ENGINE.render_geff_overlay(movie_id, t=t, z=z)

def run_module0(movie_id: str, max_cells: Optional[int] = 256) -> Dict[str, Any]:
    return ENGINE.run_full_preprocess(movie_id, max_cells=max_cells)


def module_status() -> Dict[str, Any]:
    return ENGINE.status()


def safe_call(fn, *args, **kwargs) -> Dict[str, Any]:
    try:
        result = fn(*args, **kwargs)
        return {"ok": True, "result": result}
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Module0 error: %s\n%s", exc, traceback.format_exc())
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
