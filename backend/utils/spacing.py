"""
Physical voxel spacing and anisotropy helpers for CellTrace.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from config import (
    ANISOTROPY_RATIO_Z_OVER_XY,
    SPACING_ZYX_UM,
    VOXEL_SIZE_X_UM,
    VOXEL_SIZE_Y_UM,
    VOXEL_SIZE_Z_UM,
)


@dataclass(frozen=True)
class VoxelSpacing:
    z_um: float = VOXEL_SIZE_Z_UM
    y_um: float = VOXEL_SIZE_Y_UM
    x_um: float = VOXEL_SIZE_X_UM

    @property
    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.z_um, self.y_um, self.x_um)

    @property
    def anisotropy(self) -> float:
        xy = 0.5 * (self.y_um + self.x_um)
        if xy <= 0:
            raise ValueError("XY spacing must be positive")
        return self.z_um / xy

    def validate(self, tol: float = 1e-3) -> List[str]:
        issues: List[str] = []
        if abs(self.z_um - VOXEL_SIZE_Z_UM) > tol:
            issues.append(
                f"Z spacing {self.z_um} µm differs from expected {VOXEL_SIZE_Z_UM} µm"
            )
        if abs(self.y_um - VOXEL_SIZE_Y_UM) > tol:
            issues.append(
                f"Y spacing {self.y_um} µm differs from expected {VOXEL_SIZE_Y_UM} µm"
            )
        if abs(self.x_um - VOXEL_SIZE_X_UM) > tol:
            issues.append(
                f"X spacing {self.x_um} µm differs from expected {VOXEL_SIZE_X_UM} µm"
            )
        if abs(self.anisotropy - ANISOTROPY_RATIO_Z_OVER_XY) > 0.05:
            issues.append(
                f"Anisotropy {self.anisotropy:.4f} differs from expected "
                f"{ANISOTROPY_RATIO_Z_OVER_XY:.4f}"
            )
        return issues

    def voxels_to_um(self, zyx: Sequence[float]) -> np.ndarray:
        arr = np.asarray(zyx, dtype=np.float64)
        return arr * np.asarray(self.as_tuple, dtype=np.float64)

    def um_to_voxels(self, zyx_um: Sequence[float]) -> np.ndarray:
        arr = np.asarray(zyx_um, dtype=np.float64)
        return arr / np.asarray(self.as_tuple, dtype=np.float64)

    def physical_distance_um(
        self, a_zyx: Sequence[float], b_zyx: Sequence[float]
    ) -> float:
        a = self.voxels_to_um(a_zyx)
        b = self.voxels_to_um(b_zyx)
        return float(np.linalg.norm(a - b))

    def margin_voxels(self, margin_um: float) -> Tuple[int, int, int]:
        mz = max(1, int(np.ceil(margin_um / self.z_um)))
        my = max(1, int(np.ceil(margin_um / self.y_um)))
        mx = max(1, int(np.ceil(margin_um / self.x_um)))
        return mz, my, mx


DEFAULT_SPACING = VoxelSpacing()


def calibrate_bbox_zyx(
    center_zyx: Sequence[float],
    half_extent_um: Sequence[float],
    spacing: VoxelSpacing = DEFAULT_SPACING,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Return integer (z0,y0,x0), (z1,y1,x1) crop bounds from physical extent."""
    c = np.asarray(center_zyx, dtype=np.float64)
    half_vox = spacing.um_to_voxels(half_extent_um)
    z0, y0, x0 = np.floor(c - half_vox).astype(int)
    z1, y1, x1 = np.ceil(c + half_vox).astype(int)
    return (int(z0), int(y0), int(x0)), (int(z1), int(y1), int(x1))


def clamp_bbox(
    start: Sequence[int],
    end: Sequence[int],
    shape_zyx: Sequence[int],
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    z0 = int(np.clip(start[0], 0, shape_zyx[0]))
    y0 = int(np.clip(start[1], 0, shape_zyx[1]))
    x0 = int(np.clip(start[2], 0, shape_zyx[2]))
    z1 = int(np.clip(end[0], 0, shape_zyx[0]))
    y1 = int(np.clip(end[1], 0, shape_zyx[1]))
    x1 = int(np.clip(end[2], 0, shape_zyx[2]))
    if z1 <= z0:
        z1 = min(shape_zyx[0], z0 + 1)
    if y1 <= y0:
        y1 = min(shape_zyx[1], y0 + 1)
    if x1 <= x0:
        x1 = min(shape_zyx[2], x0 + 1)
    return (z0, y0, x0), (z1, y1, x1)


def resample_isotropic_factors(
    spacing: VoxelSpacing = DEFAULT_SPACING,
) -> Tuple[float, float, float]:
    """Factors to resample anisotropic voxels toward isotropic XY pitch."""
    target = spacing.x_um
    return (spacing.z_um / target, spacing.y_um / target, spacing.x_um / target)


def physical_volume_um3(shape_zyx: Sequence[int], spacing: VoxelSpacing = DEFAULT_SPACING) -> float:
    z, y, x = shape_zyx
    return float(z * spacing.z_um * y * spacing.y_um * x * spacing.x_um)


def batch_distances_um(
    points_zyx: np.ndarray,
    query_zyx: np.ndarray,
    spacing: VoxelSpacing = DEFAULT_SPACING,
) -> np.ndarray:
    """Pairwise physical distances between query points and a point cloud."""
    pts = np.asarray(points_zyx, dtype=np.float64) * np.asarray(spacing.as_tuple)
    qry = np.asarray(query_zyx, dtype=np.float64) * np.asarray(spacing.as_tuple)
    if qry.ndim == 1:
        qry = qry[None, :]
    d = qry[:, None, :] - pts[None, :, :]
    return np.linalg.norm(d, axis=-1)


def expected_spacing_tuple() -> Tuple[float, float, float]:
    return SPACING_ZYX_UM


def summarize_spacing(spacing: VoxelSpacing = DEFAULT_SPACING) -> dict:
    return {
        "z_um": spacing.z_um,
        "y_um": spacing.y_um,
        "x_um": spacing.x_um,
        "anisotropy": spacing.anisotropy,
        "issues": spacing.validate(),
        "ok": len(spacing.validate()) == 0,
    }
