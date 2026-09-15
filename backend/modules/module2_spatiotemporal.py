"""
Module 2 — Complete Spatiotemporal Feature Fusion.

Input:
  - CellDINO appearance embeddings from Module 1
  - GEFF cell positions and supplied temporal relationships

Output:
  - aligned, fused 256-dimensional feature vector per cell observation
  - spatial k-NN graph for Module 3
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from config import EMBEDDING_DIR, FEATURE_DIR, KNN_NEIGHBORS, LOCAL_DATA_ROOT
from utils.io_geff import load_geff
from utils.spacing import DEFAULT_SPACING, batch_distances_um


class SpatiotemporalEngine:
    def __init__(self) -> None:
        self.data_root = Path(LOCAL_DATA_ROOT)
        self.feature_root = Path(FEATURE_DIR)
        self.feature_root.mkdir(parents=True, exist_ok=True)
        self.spacing = DEFAULT_SPACING
        self.knn = KNN_NEIGHBORS
        self.state: Dict[str, Any] = {"status": "idle", "progress": 0.0}

    def _geff_path(self, movie_id: str) -> Path:
        path = self.data_root / f"{movie_id}.geff"
        if not path.exists():
            raise FileNotFoundError(f"GEFF file not found: {path}")
        return path

    def _appearance(self, movie_id: str) -> Tuple[Dict[int, np.ndarray], int]:
        path = Path(EMBEDDING_DIR) / movie_id / "appearance_embeddings.npz"
        if not path.exists():
            raise FileNotFoundError(
                "No CellDINO embeddings found. Run Module 1 for this movie first."
            )

        data = np.load(path)
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        node_ids = np.asarray(data["node_ids"], dtype=np.int64)

        return {
            int(node_id): embeddings[index]
            for index, node_id in enumerate(node_ids)
        }, int(embeddings.shape[1])

    @staticmethod
    def _standardize(values: np.ndarray) -> np.ndarray:
        if values.shape[0] < 2:
            return values.astype(np.float32)

        mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True)
        return ((values - mean) / np.maximum(std, 1e-6)).astype(np.float32)

    def run_full(self, movie_id: str) -> Dict[str, Any]:
        self.state = {
            "status": "running",
            "progress": 0.05,
            "message": "Loading GEFF graph and CellDINO embeddings",
        }

        graph = load_geff(self._geff_path(movie_id))
        nodes = sorted(graph.nodes, key=lambda node: (int(node.t), int(node.node_id)))

        if not nodes:
            raise RuntimeError("The GEFF graph contains no cell observations.")

        appearance_by_node, embed_dim = self._appearance(movie_id)

        node_ids = np.asarray([int(node.node_id) for node in nodes], dtype=np.int64)
        times = np.asarray([int(node.t) for node in nodes], dtype=np.int64)
        positions = np.asarray([node.zyx for node in nodes], dtype=np.float32)
        positions_um = positions * np.asarray(self.spacing.as_tuple, dtype=np.float32)

        index_of = {int(node_id): index for index, node_id in enumerate(node_ids)}
        n = len(nodes)

        appearance = np.zeros((n, embed_dim), dtype=np.float32)
        appearance_available = np.zeros(n, dtype=np.float32)

        for index, node_id in enumerate(node_ids):
            if int(node_id) in appearance_by_node:
                appearance[index] = appearance_by_node[int(node_id)]
                appearance_available[index] = 1.0

        # Temporal evidence: velocity and displacement obtained from GEFF links.
        velocity = np.zeros((n, 3), dtype=np.float32)
        speed = np.zeros(n, dtype=np.float32)
        temporal_gap = np.zeros(n, dtype=np.float32)
        incoming_links = np.zeros(n, dtype=np.float32)

        for source_id, target_id in graph.edges:
            if source_id not in index_of or target_id not in index_of:
                continue

            source_index = index_of[source_id]
            target_index = index_of[target_id]
            dt = int(times[target_index] - times[source_index])

            if dt <= 0:
                continue

            displacement = positions_um[target_index] - positions_um[source_index]
            current_velocity = displacement / float(dt)

            velocity[target_index] += current_velocity
            speed[target_index] += float(np.linalg.norm(current_velocity))
            temporal_gap[target_index] += float(dt)
            incoming_links[target_index] += 1.0

        valid_temporal = incoming_links > 0
        velocity[valid_temporal] /= incoming_links[valid_temporal, None]
        speed[valid_temporal] /= incoming_links[valid_temporal]
        temporal_gap[valid_temporal] /= incoming_links[valid_temporal]

        # Spatial evidence: each cell connects to its k nearest cells in one frame.
        neighbour_count = np.zeros(n, dtype=np.float32)
        mean_neighbour_distance = np.zeros(n, dtype=np.float32)
        min_neighbour_distance = np.zeros(n, dtype=np.float32)
        spatial_edges: List[Tuple[int, int, float]] = []

        for frame in np.unique(times):
            frame_indices = np.where(times == frame)[0]

            if len(frame_indices) < 2:
                continue

            frame_positions = positions[frame_indices]
            distances = batch_distances_um(
                frame_positions,
                frame_positions,
                self.spacing,
            )
            np.fill_diagonal(distances, np.inf)

            k = min(self.knn, len(frame_indices) - 1)

            for local_source, global_source in enumerate(frame_indices):
                nearest = np.argsort(distances[local_source])[:k]
                nearest_distances = distances[local_source, nearest]

                neighbour_count[global_source] = float(len(nearest))
                mean_neighbour_distance[global_source] = float(
                    np.mean(nearest_distances)
                )
                min_neighbour_distance[global_source] = float(
                    np.min(nearest_distances)
                )

                for local_target, distance in zip(nearest, nearest_distances):
                    spatial_edges.append(
                        (
                            int(global_source),
                            int(frame_indices[local_target]),
                            float(distance),
                        )
                    )

        self.state.update(
            {
                "progress": 0.55,
                "message": "Aligning appearance, temporal, and neighbourhood features",
            }
        )

        # 10 interpretable context values per observation.
        context_raw = np.column_stack(
            [
                velocity,
                speed,
                temporal_gap,
                incoming_links,
                neighbour_count,
                mean_neighbour_distance,
                min_neighbour_distance,
            ]
        ).astype(np.float32)

        context = self._standardize(context_raw)

        # Deterministic alignment: 10 context values -> same 256-D space as CellDINO.
        rng = np.random.default_rng(2026)
        projection = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(context.shape[1]),
            size=(context.shape[1], embed_dim),
        ).astype(np.float32)

        aligned_context = np.tanh(context @ projection).astype(np.float32)

        # Adaptive Module-2 fusion weight.
        # Cells with temporal evidence and a stable local neighbourhood retain
        # more appearance detail; uncertain observations receive more context.
        alpha = (
            0.55
            + 0.20 * np.clip(incoming_links, 0.0, 1.0)
            + 0.15 * np.clip(neighbour_count / max(self.knn, 1), 0.0, 1.0)
            + 0.10 * appearance_available
        )
        alpha = np.clip(alpha, 0.50, 0.95).astype(np.float32)

        fused = (
            alpha[:, None] * appearance
            + (1.0 - alpha[:, None]) * aligned_context
        )
        fused /= np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-6)
        fused = fused.astype(np.float32)

        output_dir = self.feature_root / movie_id
        output_dir.mkdir(parents=True, exist_ok=True)

        features_path = output_dir / "fused_features.npz"
        edges_path = output_dir / "spatial_neighbour_graph.npz"
        summary_path = output_dir / "module2_fusion_summary.json"

        np.savez_compressed(
            features_path,
            node_ids=node_ids,
            times=times,
            positions_zyx=positions,
            appearance=appearance,
            context=context,
            alpha=alpha,
            fused_features=fused,
        )

        edge_array = (
            np.asarray(spatial_edges, dtype=np.float32)
            if spatial_edges
            else np.zeros((0, 3), dtype=np.float32)
        )

        np.savez_compressed(
            edges_path,
            edges=edge_array,
        )

        preview = [
            {
                "node_id": int(node_ids[index]),
                "frame": int(times[index]),
                "speed_um_per_frame": round(float(speed[index]), 3),
                "neighbours": int(neighbour_count[index]),
                "appearance_weight": round(float(alpha[index]), 3),
            }
            for index in range(min(8, n))
        ]

        result = {
            "movie_id": movie_id,
            "status": "complete",
            "n_cell_observations": int(n),
            "n_geff_temporal_links": int(np.sum(incoming_links)),
            "n_spatial_neighbour_links": int(len(spatial_edges)),
            "appearance_embeddings_available": int(np.sum(appearance_available)),
            "appearance_coverage_percent": round(
                100.0 * float(np.mean(appearance_available)),
                1,
            ),
            "context_feature_dimension": int(context.shape[1]),
            "fused_feature_dimension": int(fused.shape[1]),
            "mean_appearance_weight": round(float(np.mean(alpha)), 3),
            "preview": preview,
            "artifacts": {
                "fused_features": str(features_path),
                "spatial_graph": str(edges_path),
            },
            "feature_groups": [
                "CellDINO appearance representation",
                "GEFF temporal displacement and velocity",
                "local k-nearest-neighbour context",
                "aligned adaptive appearance-context fusion",
            ],
        }

        summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        self.state = {
            "status": "complete",
            "progress": 1.0,
            "message": "Spatiotemporal feature fusion complete",
            "result": result,
        }

        return result

def graph_preview(
    movie_id: str,
    frame: int | None = None,
    focus_node_id: int | None = None,
) -> Dict[str, Any]:
    """
    Returns real XY positions and real GEFF links for visualisation.
    Spatial edges are Module-2 k-nearest-neighbour links.
    Temporal path edges are supplied GEFF annotation relationships.
    """
    feature_dir = Path(FEATURE_DIR) / movie_id
    feature_path = feature_dir / "fused_features.npz"
    edge_path = feature_dir / "spatial_neighbour_graph.npz"

    if not feature_path.exists() or not edge_path.exists():
        ENGINE.run_full(movie_id)

    features = np.load(feature_path)
    graph_data = np.load(edge_path)

    node_ids = np.asarray(features["node_ids"], dtype=np.int64)
    times = np.asarray(features["times"], dtype=np.int64)
    positions = np.asarray(features["positions_zyx"], dtype=np.float32)
    spatial_edges = np.asarray(graph_data["edges"], dtype=np.float32)

    node_index = {int(node_id): index for index, node_id in enumerate(node_ids)}

    focus_index = node_index.get(int(focus_node_id)) if focus_node_id else None

    if focus_index is not None:
        selected_frame = int(times[focus_index])
    elif frame is not None:
        selected_frame = int(frame)
    else:
        unique_frames, counts = np.unique(times, return_counts=True)
        selected_frame = int(unique_frames[np.argmax(counts)])

    frame_indices = np.where(times == selected_frame)[0].tolist()
    frame_index_set = set(frame_indices)

    spatial_nodes = [
        {
            "node_id": int(node_ids[index]),
            "t": int(times[index]),
            "z": round(float(positions[index, 0]), 2),
            "y": round(float(positions[index, 1]), 2),
            "x": round(float(positions[index, 2]), 2),
            "is_focus": bool(index == focus_index),
        }
        for index in frame_indices
    ]

    visible_spatial_edges = [
        {
            "source_node_id": int(node_ids[int(source)]),
            "target_node_id": int(node_ids[int(target)]),
            "distance_um": round(float(distance), 3),
        }
        for source, target, distance in spatial_edges
        if int(source) in frame_index_set and int(target) in frame_index_set
    ]

    geff = load_geff(ENGINE._geff_path(movie_id))
    children: Dict[int, List[int]] = {}

    for source_id, target_id in geff.edges:
        children.setdefault(int(source_id), []).append(int(target_id))

    if focus_node_id and int(focus_node_id) in node_index:
        path_start = int(focus_node_id)
    else:
        path_start = next(
            (
                int(node_ids[index])
                for index in frame_indices
                if int(node_ids[index]) in children
            ),
            int(node_ids[frame_indices[0]]) if frame_indices else int(node_ids[0]),
        )

    # Follow one real GEFF path for at most 12 observations.
    temporal_path_ids = [path_start]
    current = path_start

    for _ in range(11):
        next_ids = [
            child_id
            for child_id in children.get(current, [])
            if child_id in node_index
            and times[node_index[child_id]] > times[node_index[current]]
        ]

        if not next_ids:
            break

        current = min(
            next_ids,
            key=lambda child_id: int(times[node_index[child_id]]),
        )
        temporal_path_ids.append(current)

    temporal_path = [
        {
            "node_id": node_id,
            "t": int(times[node_index[node_id]]),
            "y": round(float(positions[node_index[node_id], 1]), 2),
            "x": round(float(positions[node_index[node_id], 2]), 2),
        }
        for node_id in temporal_path_ids
        if node_id in node_index
    ]

    temporal_links = [
        {
            "source_node_id": temporal_path_ids[index],
            "target_node_id": temporal_path_ids[index + 1],
        }
        for index in range(len(temporal_path_ids) - 1)
    ]

    return {
        "movie_id": movie_id,
        "spatial": {
            "frame": selected_frame,
            "focus_node_id": int(focus_node_id) if focus_node_id else None,
            "nodes": spatial_nodes,
            "edges": visible_spatial_edges,
        },
        "temporal": {
            "path": temporal_path,
            "links": temporal_links,
            "note": (
                "These are supplied GEFF annotation links. "
                "They are visualised as reference relationships, not predictions."
            ),
        },
    }
ENGINE = SpatiotemporalEngine()


def run_full(movie_id: str) -> Dict[str, Any]:
    return ENGINE.run_full(movie_id)


def module_status() -> Dict[str, Any]:
    return ENGINE.state


def safe_call(fn, *args, **kwargs) -> Dict[str, Any]:
    try:
        return {"ok": True, "result": fn(*args, **kwargs)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}