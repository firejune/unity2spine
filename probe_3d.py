"""How much of the rig is genuinely 3D -- the part a 2D skeleton cannot carry.

A Spine bone chain composes flattened 2D locals; Unity composes true 3D and is
flattened only at the end.  Those agree only while every local rotation is
about Z.  This counts the transforms where that does not hold.

Usage: python probe_3d.py <bundle-dir> <out.json> [limit]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import unity_to_spine as u2s  # noqa: E402
from uscene import load_light  # noqa: E402


def main():
    src = Path(sys.argv[1])
    # the census is a generated file: it goes where the caller asks, not next to
    # this script (which lives in the repo)
    out = Path(sys.argv[2])
    bundles = sorted(src.glob("*.bundle"))
    if len(sys.argv) > 3:
        bundles = bundles[: int(sys.argv[3])]
    rows = []
    for i, b in enumerate(bundles):
        try:
            sc = load_light(b)
        except Exception as e:  # noqa: BLE001
            print(f"[{i+1}] {b.stem}: load failed {e}", flush=True)
            continue
        n3d = 0
        worst = 0.0
        for tr, t in sc["TR"].items():
            x, y, z, w = t["rot"]
            # pure-Z quaternion has x == y == 0
            off = float(np.hypot(x, y))
            worst = max(worst, off)
            if off > 1e-3:
                n3d += 1
        rows.append({"bundle": b.stem, "transforms": len(sc["TR"]),
                     "non_planar_rest": n3d, "worst_xy": worst,
                     "clips": len(sc["clips"])})
        print(f"[{i+1}/{len(bundles)}] {b.stem}: {n3d}/{len(sc['TR'])} "
              f"non-planar rest rotations (worst |xy|={worst:.4f})", flush=True)

    anim = [r for r in rows if r["clips"]]
    bad = [r for r in anim if r["non_planar_rest"]]
    print(f"\n=== {len(rows)} rigs read, {len(anim)} with clips ===")
    print(f"  rigs with >=1 non-planar (3D) rest rotation: {len(bad)}")
    tot_tr = sum(r["transforms"] for r in anim)
    tot_3d = sum(r["non_planar_rest"] for r in anim)
    print(f"  transforms: {tot_3d}/{tot_tr} non-planar "
          f"({100*tot_3d/max(1,tot_tr):.2f}%)")
    for r in sorted(bad, key=lambda r: -r["non_planar_rest"])[:15]:
        print(f"    {r['bundle']:44s} {r['non_planar_rest']:4d}/{r['transforms']:4d}")
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
