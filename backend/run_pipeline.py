"""
CellTrace — Review-2 pipeline runner (Modules 0 → 4-first-half).

Runs the full chain on REAL data in backend/local_data/train/:
    {movie}.zarr + {movie}.geff  (flat layout, Zarr v3)

Steps:
  0. validate + tensors        (Module 0)
  1. appearance embeddings      (Module 1, prototype ViT encoder)
  2. spatiotemporal fusion      (Module 2)
  3. contrastive GAT            (Module 3 — movie-level when >=2 movies)
  4. group-weight fitting       (Module 4, train movies only)
  5. matching + Hungarian + confidence (Module 4 first half)

Usage:
    cd backend
    python run_pipeline.py --movies <movieA> <movieB> [--val movieC] [--epochs 50]

Movie IDs are the file stems, e.g. `44b6_0049_0438_1330_1273`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import GAT_EPOCHS, ensure_artifact_dirs  # noqa: E402
from modules import module0_data_prep as m0  # noqa: E402
from modules import module1_celldino as m1  # noqa: E402
from modules import module2_spatiotemporal as m2  # noqa: E402
from modules import module3_gat as m3  # noqa: E402
from modules import module4_tracking as m4  # noqa: E402


def _pick(d, key, default="?"):
    """Read `key` from a flat result dict or a safe_call-style wrapper."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        inner = d.get("result", {})
        if isinstance(inner, dict):
            return inner.get(key, default)
        return getattr(inner, key, default)
    return getattr(d, key, default)


def _inner(d):
    if isinstance(d, dict) and "result" in d and isinstance(d["result"], dict):
        return d["result"]
    return d


def run_movie(movies, val_movies, epochs: int, max_cells):
    t0 = time.time()
    manifest: dict = {"movies": list(movies), "val_movies": list(val_movies),
                      "epochs": epochs, "steps": {}}
    all_movies = list(dict.fromkeys(list(movies) + list(val_movies)))

    print("=== [0] Module 0 — validation + tensors ===", flush=True)
    for movie in all_movies:
        v = m0.validate_movie(movie)
        t = m0.tensorize_movie(movie, max_cells=max_cells)
        manifest["steps"].setdefault(movie, {})["module0"] = {
            "validate": v, "tensors": t.get("result", t)}
        print(f"  {movie}: tensors={_pick(t, 'n_tensors')}",
              flush=True)

    print("=== [1] Module 1 — appearance embeddings ===", flush=True)
    for movie in all_movies:
        r = m1.run_celldino(movie)
        manifest["steps"][movie]["module1"] = _inner(r)
        print(f"  {movie}: embeddings={_pick(r, 'n_embeddings')}",
              flush=True)

    print("=== [2] Module 2 — spatiotemporal fusion ===", flush=True)
    for movie in all_movies:
        r = m2.run_full(movie)
        manifest["steps"][movie]["module2"] = r
        print(f"  {movie}: fused={r['n_cell_observations']} "
              f"links={r['n_geff_temporal_links']}", flush=True)

    print("=== [3] Module 3 — contrastive GAT ===", flush=True)
    if val_movies:
        r = m3.train_gat_multimovie(list(movies), list(val_movies), epochs=epochs)
        manifest["module3_multimovie"] = {k: v for k, v in r.items()
                                          if k != "training_history"}
        manifest["module3_history_tail"] = r["training_history"][-5:]
        print(f"  movie-level train={movies} val={val_movies} "
              f"best_val={r['best_val_loss']}", flush=True)
    else:
        for movie in all_movies:
            r = m3.run_gat(movie, epochs=epochs)
            manifest["steps"][movie]["module3"] = {k: v for k, v in r.items()
                                                   if k != "training_history"}
            print(f"  {movie}: mode={r['mode']} "
                  f"final_train={r.get('final_train_loss', '?')}", flush=True)

    print("=== [4a] Module 4 — fit group weights (train movies) ===", flush=True)
    fw = m4.fit_weights(list(movies))
    manifest["module4_weights"] = {k: v for k, v in fw.items() if k != "history"}
    print(f"  weights={fw['weights']} loss={fw['final_loss']}", flush=True)

    print("=== [4b] Module 4 — matching + Hungarian + confidence ===", flush=True)
    for movie in all_movies:
        r = m4.run_tracking(movie)
        manifest["steps"][movie]["module4"] = {k: v for k, v in r.items()
                                               if k not in ("preview",)}
        print(f"  {movie}: associations={r['n_associations']} "
              f"mean_conf={r['mean_confidence']}", flush=True)

    manifest["elapsed_sec"] = round(time.time() - t0, 1)
    out = BACKEND_ROOT / "artifacts" / "review2_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\nDONE in {manifest['elapsed_sec']}s. Manifest: {out}", flush=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--movies", nargs="+", required=True)
    ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--epochs", type=int, default=GAT_EPOCHS)
    ap.add_argument("--max-cells", type=int, default=None)
    args = ap.parse_args()
    ensure_artifact_dirs()
    # Reproducibility: the Module-1 prototype encoder initializes randomly on
    # every run, so seed everything here — repeated runs give identical numbers.
    import random
    import numpy as _np
    import torch as _torch
    random.seed(2026)
    _np.random.seed(2026)
    _torch.manual_seed(2026)
    run_movie(args.movies, args.val, args.epochs, args.max_cells)


if __name__ == "__main__":
    main()
