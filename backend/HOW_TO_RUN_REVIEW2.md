# CellTrace — How to Run (Review 2: Modules 0 → 4-first-half)

All commands run on **real data**. No synthetic data, no invented numbers —
every metric in every summary JSON is computed at runtime from your files.

## 0. Data layout (your 10 local movies — already correct)

```
backend/local_data/train/
    <movie_id>.zarr/      # Zarr v3 image volume, array 0/ with (T,Z,Y,X)
    <movie_id>.geff/      # Zarr v3 annotations (nodes/ids, nodes/props, edges/ids)
```

`<movie_id>` is the file stem, e.g. `44b6_0049_0438_1330_1273`.
This data is git-ignored by design (too big for GitHub) — it stays on your machine.

## 1. Install (Windows)

```bat
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` already covers everything Review 2 needs
(torch CPU, zarr, scipy for Hungarian matching, fastapi, …).
No PyTorch-Geometric needed — the GAT is pure-PyTorch (sparse, CPU-friendly).

## 2. Sanity check (no data needed)

```bat
python tests\test_review2_math.py
```

Verifies the new math (KNN same-frame-only, GT pair labelling, GAT forward,
contrastive loss direction, softmax weights, Hungarian). These are software
tests only — their numbers never enter any report.

## 3. Full pipeline — one command

```bat
python run_pipeline.py --movies <movieA> <movieB> --val <movieC> --epochs 50
```

- `--movies`: 1+ train movies. `--val`: optional held-out movies (movie-level validation).
- `--epochs`: GAT contrastive epochs (use 10–20 for a quick check, 50 for the review).
- `--max-cells N`: optional cap for a fast trial run (e.g. 256).

What it runs: Module 0 (validate+tensors) → Module 1 (appearance) →
Module 2 (fusion) → Module 3 (contrastive GAT, movie-level if `--val` given) →
Module 4a (fit group weights on train movies) → Module 4b
(matching + Hungarian + confidence).

Single-movie quick check:

```bat
python run_pipeline.py --movies <movieA> --epochs 10 --max-cells 256
```

## 4. API + UI (same as Review 1, plus new endpoints)

```bat
uvicorn main:app --reload --port 8000
```

- Docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/api/health
- New: `POST /api/module4/run` (`{"movie_id": ...}`),
  `POST /api/module4/fit-weights` (`{"train_movie_ids": [...]}`),
  `GET /api/module4/status`

Frontend (unchanged from Review 1): `cd frontend && npm install && npm start`.

## 5. Outputs — where the real numbers live

```
backend/artifacts/
    tensors/<movie>/cell_tensors.npz + tensor_manifest.json
    embeddings/<movie>/appearance_embeddings.npz + .json
    features/<movie>/fused_features.npz
    features/<movie>/spatial_neighbour_graph.npz
    features/<movie>/context_enhanced_features.npz
    features/<movie>/neighbour_aware_gat.pt
    features/<movie>/module3_gat_summary.json        # train/val loss history, pair counts
    tracks/group_weight_mlp.pt                       # fitted group weights (train movies)
    tracks/<movie>/associations.npz                  # src/dst/cost/confidence per match
    tracks/<movie>/tracking_half_summary.json        # weights, mean confidence, per-pair stats
    review2_manifest.json                            # whole-run manifest
```

How to prove "real, not made up" in your review:
1. Open `review2_manifest.json` — every step's counts, shapes, losses.
2. `module3_gat_summary.json` → `pairs` (GT positives/negatives actually used),
   `training_history` (loss per epoch, computed, not typed).
3. `tracking_half_summary.json` → `group_weights` + `group_weight_source`
   (`fitted:[movies]` vs `uniform-prior`), per-frame-pair match counts.
4. All `.npz` files: `np.load(...)` and inspect — shapes match the JSON counts.

## 6. Honest-labelling checklist (for your report/PPT)

- Module 1 encoder = **prototype (untrained)** unless you ran
  `module1_training.train_self_supervised` — say which one you ran.
- Module 3 = **trained** (contrastive, this review) — cite loss curves.
- Module 4 weights = fitted (BCE on train movies) or uniform prior —
  the summary file states the source explicitly.
- Tracking = **first half only** (matching + confidence). Thresholding,
  trajectory stitching, division/lineage, MOTA/MOTP → Review 3.

## 7. Fetching one real movie from Kaggle (optional, for a second machine)

Full dataset ≈ 87 GB — don't download all of it. To fetch a single sample
(complete `.geff` + first K frames of `.zarr`):

```bat
pip install kaggle
python scripts\fetch_kaggle_sample.py --frames 5
```

Needs `~/.kaggle/kaggle.json` (from kaggle.com → Settings → Create New API Token)
and accepted competition rules. Delete `kaggle.json` afterwards if shared machine.
