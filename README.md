# CellTrace

The new CellTrace **Dive** experience: descend through the full 3D cell-tracking
pipeline — Specimen → Validate → Extract → Standardize → Embed → Connect →
Train → Track → Export — on real data, with every number computed live.

Previous work (Review-1 modules, lab UI) lives separately at
`SubhikshaVB/CellTracking`. This repo is a self-contained app: the proven
pipeline backend plus the brand-new Dive frontend.

## Structure

```
backend/    FastAPI pipeline (M0–M4 + journey + upload endpoints)
frontend/   React Dive app (dark/light, auto-advance, talk-tracks)
```

## Run

Backend (http://127.0.0.1:8000, docs at `/docs`):

```
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Frontend (http://localhost:3000):

```
cd frontend
npm install
npm start
```

## Data

- **Lab vault:** paste movies into `backend/local_data/train/` (`.zarr` + `.geff`).
- **Your own embryo:** upload `.zarr` / `.geff` *folders* from Stage 00 in the app.

## Push this to GitHub

```
git remote add origin https://github.com/SubhikshaVB/CellTrace.git
git push -u origin main
```

## Principles

- Every number is computed live from real data. Nothing is mocked.
- Display-only visual helpers (motion, contrast boost) are always labeled.
- Honest future work (division/lineage, pretrained encoder) is marked, never claimed.

## Later

- Results database (SQLite run-store) — after the Dive ships.
