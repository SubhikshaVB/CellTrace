"""
CellTrace — fetch ONE real Kaggle movie (partial Zarr) for a real-data run.

The full competition dataset is ~87 GB — far too big to download whole.
This script downloads a single train sample:
  - the complete .geff annotation  (small: sparse graphs)
  - .zarr metadata (zarr.json files) + only the first K timepoint chunks
    (each chunk = 1 timepoint, so K frames of real voxels)

Result runs the REAL pipeline on REAL voxels + REAL annotations, just fewer
frames. Nothing synthetic, nothing invented.

Prerequisites:
  1. You accepted the competition rules (you did — you downloaded before).
  2. Kaggle API token: https://www.kaggle.com/settings -> "Create New API Token"
     saves kaggle.json. Then EITHER:
       - place it at ~/.kaggle/kaggle.json  (chmod 600), OR
       - set env KAGGLE_USERNAME + KAGGLE_KEY.
  3. pip install kaggle

Usage:
    cd backend
    python scripts/fetch_kaggle_sample.py --frames 5
    python scripts/fetch_kaggle_sample.py --sample <name> --frames 8 --out local_data/train
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

COMPETITION = "biohub-cell-tracking-during-development"


def get_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit("pip install kaggle   (needed only for this fetch script)") from exc
    api = KaggleApi()
    api.authenticate()  # reads ~/.kaggle/kaggle.json or env vars
    return api


def list_competition_files(api):
    files = api.competition_list_files(COMPETITION)
    out = []
    for f in files:
        name = getattr(f, "name", str(f))
        size = int(getattr(f, "size", 0) or 0)
        out.append((name.replace("\\", "/"), size))
    return out


def pick_smallest_train_sample(files):
    zarr_roots: dict[str, int] = {}
    for name, size in files:
        if not name.startswith("train/") or ".zarr/" not in name:
            continue
        root = "train/" + name[len("train/"):].split(".zarr/")[0] + ".zarr"
        zarr_roots[root] = zarr_roots.get(root, 0) + size
    if not zarr_roots:
        raise SystemExit("No train/*.zarr files listed — did you accept the competition rules?")
    best = min(zarr_roots, key=zarr_roots.get)
    sample = best[len("train/"):].removesuffix(".zarr")
    print(f"train samples found: {len(zarr_roots)}; smallest = {sample} "
          f"({zarr_roots[best]/1e9:.2f} GB full, we fetch only a few frames)")
    return sample


def download(api, remote: str, local: Path):
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and local.stat().st_size > 0:
        print(f"  skip (exists): {local.name}")
        return
    api.competition_download_file(COMPETITION, remote, path=str(local.parent),
                                  force=False, quiet=True)
    # SDK writes <parent>/<basename>; move into place if needed.
    got = local.parent / Path(remote).name
    if str(got) != str(local) and got.exists():
        got.rename(local)
    print(f"  got: {remote} -> {local.stat().st_size/1e6:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=None)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--out", default="local_data/train")
    args = ap.parse_args()

    api = get_api()
    print("listing competition files...")
    try:
        files = list_competition_files(api)
    except Exception as exc:
        raise SystemExit(
            f"Kaggle API error: {exc}\n"
            "If this is 403/401: accept the rules + join the competition at\n"
            "https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules\n"
            "and check your kaggle.json credentials.") from exc
    print(f"  {len(files)} files listed")

    sample = args.sample or pick_smallest_train_sample(files)
    out = Path(args.out)
    names = {n for n, _ in files}

    # ---- .geff: everything (small) ----
    geff_files = sorted(n for n in names if n.startswith(f"train/{sample}.geff/"))
    if not geff_files:
        raise SystemExit(f"No .geff files listed for train/{sample}")
    print(f"downloading .geff ({len(geff_files)} files)...")
    for remote in geff_files:
        rel = remote[len(f"train/{sample}.geff/"):]
        download(api, remote, out / f"{sample}.geff" / rel)

    # ---- .zarr: metadata + first K frame chunks ----
    print(f"downloading .zarr metadata + first {args.frames} frame chunks...")
    wants = ["zarr.json", "0/zarr.json"]
    for t in range(args.frames):
        wants.append(f"0/c/{t}/0/0/0")
    missing = [w for w in wants if f"train/{sample}.zarr/{w}" not in names]
    if missing:
        print(f"  WARNING: not listed (trying anyway): {missing}")
    for w in wants:
        download(api, f"train/{sample}.zarr/{w}", out / f"{sample}.zarr" / w)

    # ---- verify with the real readers ----
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.io_geff import load_geff
    from utils.io_zarr import inspect_zarr
    g = load_geff(out / f"{sample}.geff")
    v = inspect_zarr(out / f"{sample}.zarr")
    print(f"\nVERIFIED with CellTrace readers: movie_id={sample}")
    print(f"  zarr: key={v.array_key} shape={v.shape} dtype={v.dtype}")
    print(f"  geff: nodes={len(g.nodes)} edges={len(g.edges)} "
          f"t_range={g.time_range()} format={g.meta.get('format')}")
    print(f"\nNext:  python run_pipeline.py --movies {sample} --epochs 20")


if __name__ == "__main__":
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        kb = Path.home() / ".kaggle" / "kaggle.json"
        if not kb.exists():
            print("NOTE: no ~/.kaggle/kaggle.json and no KAGGLE_* env vars. "
                  "Authenticate first (see docstring).")
    main()
