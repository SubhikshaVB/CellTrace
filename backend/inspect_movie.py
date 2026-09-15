#!/usr/bin/env python3
"""CellTrace raw-file explorer (100% READ-ONLY - it never writes anything).

Shows you exactly what is inside your .zarr film reel and your .geff
annotation file for one movie. Run from the backend folder:

    python inspect_movie.py <movie_id>

Example:
    python inspect_movie.py 44b6_0db75fae
"""
import sys
from pathlib import Path

import numpy as np

try:
    import zarr
except ImportError:
    print("The 'zarr' package is not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


def walk_arrays(node, prefix=""):
    """Return [(key, array)] for every array under a zarr node."""
    out = []
    if hasattr(node, "shape") and hasattr(node, "dtype") and not hasattr(node, "keys"):
        return [(prefix or "/", node)]
    if hasattr(node, "keys"):
        for key in node.keys():
            child = node[key]
            child_prefix = f"{prefix}/{key}" if prefix else key
            out.extend(walk_arrays(child, child_prefix))
    return out


def folder_size(path: Path) -> str:
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if total < 1024 or unit == "TB":
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


def main() -> None:
    train_dir = Path(__file__).resolve().parent / "local_data" / "train"
    movies = sorted(p.name[: -len(".zarr")] for p in train_dir.glob("*.zarr")) if train_dir.exists() else []
    if len(sys.argv) < 2:
        print(__doc__)
        print("Movies found in backend/local_data/train/:")
        for m in movies:
            print(f"  - {m}")
        return
    movie = sys.argv[1].replace(".zarr", "").replace(".geff", "")
    zpath = train_dir / f"{movie}.zarr"
    gpath = train_dir / f"{movie}.geff"
    if not zpath.exists():
        print(f"No {zpath.name} in {train_dir}. Available movies:")
        for m in movies:
            print(f"  - {m}")
        return

    print("=" * 70)
    print(f"MOVIE: {movie}")
    print("=" * 70)

    # ---------------- ZARR ----------------
    print("\n### 1. THE .zarr FILM REEL (raw pixels) ###")
    print(f"folder : {zpath}  ({folder_size(zpath)} on disk)")
    zroot = zarr.open(str(zpath), mode="r")
    zarrays = walk_arrays(zroot)
    print(f"arrays inside : {len(zarrays)}")
    for key, arr in zarrays[:8]:
        print(f"  - {key}: shape={tuple(arr.shape)} dtype={arr.dtype} chunks={tuple(arr.chunks) if arr.chunks else None}")
    vol_key, vol = max(zarrays, key=lambda kv: (len(kv[1].shape), int(np.prod(kv[1].shape))))
    print(f"\nmain volume array: '{vol_key}'  shape (T,Z,Y,X) = {tuple(vol.shape)}  dtype = {vol.dtype}")
    print(f"total voxels : {int(np.prod(vol.shape)):,}")
    try:
        attrs = dict(vol.attrs)
        print(f"array metadata keys: {list(attrs)[:10]}")
        for k in ("spacing", "voxel_size", "scale", "axes"):
            if k in attrs:
                print(f"  {k} = {attrs[k]}")
    except Exception:
        pass
    t_mid, z_mid = vol.shape[0] // 2, vol.shape[1] // 2
    frame = np.asarray(vol[t_mid, z_mid])
    print(f"\nmiddle slice (T={t_mid}, Z={z_mid}): min={frame.min()} max={frame.max()} mean={frame.mean():.1f}")
    cy, cx = frame.shape[0] // 2, frame.shape[1] // 2
    print("actual raw voxel numbers (5x5 sample from slice centre):")
    print(frame[cy - 2:cy + 3, cx - 2:cx + 3])
    print("-> These numbers ARE the microscope image. Brighter = bigger number.")

    # ---------------- GEFF ----------------
    print("\n### 2. THE .geff ANNOTATION NOTES (sightings + links) ###")
    if not gpath.exists():
        print(f"MISSING: {gpath.name} not found - this pair is incomplete.")
        print("The cinema, patches, graph and tracking all need this file.")
        return
    print(f"folder : {gpath}  ({folder_size(gpath)} on disk)")
    groot = zarr.open(str(gpath), mode="r")
    garrays = walk_arrays(groot)
    print(f"arrays inside : {len(garrays)}")
    for key, arr in garrays:
        print(f"  - {key}: shape={tuple(arr.shape)} dtype={arr.dtype}")

    def col(*names):
        for name in names:
            try:
                return np.asarray(groot[name]).reshape(-1)
            except Exception:
                continue
        return None

    ids = col("nodes/ids")
    t = col("nodes/props/t/values")
    z = col("nodes/props/z/values")
    y = col("nodes/props/y/values")
    x = col("nodes/props/x/values")
    if any(v is None for v in (t, z, y, x)):
        print("Could not find node coordinate columns (nodes/props/*/values).")
        return
    n = min(len(t), len(z), len(y), len(x))
    if ids is None:
        ids = np.arange(n)
    print(f"\nNODES (sightings): {n}   |   frames T={int(t.min())}..{int(t.max())}")
    print("first 5 rows of the sightings table (node_id, T, Z, Y, X):")
    for i in range(min(5, n)):
        print(f"  cell #{int(ids[i]):>6}  was at  T={int(t[i]):>3}  Z={float(z[i]):>6.1f}  Y={float(y[i]):>7.1f}  X={float(x[i]):>7.1f}")

    edges = []
    try:
        e = np.asarray(groot["edges/ids"])
        if e.ndim == 2 and e.shape[1] >= 2:
            edges = [(int(r[0]), int(r[1])) for r in e]
    except Exception:
        pass
    print(f"\nEDGES (truth links): {len(edges)}")
    id2t = {int(ids[i]): int(t[i]) for i in range(n)}
    for s, d in edges[:5]:
        print(f"  cell #{s} @T{id2t.get(s, '?')}  -->  cell #{d} @T{id2t.get(d, '?')}   (same cell, next sighting)")
    # trace one full chain
    child = {}
    has_parent = set()
    for s, d in edges:
        if s not in child:
            child[s] = d
        has_parent.add(d)
    roots = [i for i in ids.tolist() if i not in has_parent]
    if roots:
        chain, cur, guard = [roots[0]], roots[0], 0
        while cur in child and guard < 500:
            cur = child[cur]
            chain.append(cur)
            guard += 1
        hops = " -> ".join(f"#{c}@T{id2t.get(c, '?')}" for c in chain[:12])
        more = f" ... (+{len(chain) - 12} more)" if len(chain) > 12 else ""
        print(f"\nONE FULL LIFE STORY ({len(chain)} sightings, {len(roots)} separate stories in file):")
        print(f"  {hops}{more}")
        print("-> Follow the arrows: that is one cell, frame by frame. Tracking = rediscovering these chains from pixels.")

    print("\n" + "=" * 70)
    print("WHAT THIS MEANS: .zarr = the film (pixel numbers). .geff = the")
    print("attendance register (who was where, when) + confirmed ID links.")
    print("Your pipeline watches the film and tries to rebuild the register.")
    print("=" * 70)


if __name__ == "__main__":
    main()
