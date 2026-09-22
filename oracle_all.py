"""Run the pose oracle over every bundle, before-fix vs after-fix, side by side.

Usage: python oracle_all.py <bundle-dir> <before-dir> <after-dir> <out.json>
                            [--samples 8] [--aligned] [--shard I --nshard N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from oracle import compare_scene  # noqa: E402
from uscene import bone_names, load_light  # noqa: E402


def rotate_stats(skel_path: Path) -> dict:
    skel = json.loads(skel_path.read_text())
    anims = skel.get("animations", {})
    tracks = {"rotate": 0, "translate": 0, "scale": 0, "shear": 0}
    keys = dict(tracks)
    anims_with_rotate = 0
    for a in anims.values():
        bones = a.get("bones", {})
        if any("rotate" in t for t in bones.values()):
            anims_with_rotate += 1
        for t in bones.values():
            for k in tracks:
                if k in t:
                    tracks[k] += 1
                    keys[k] += len(t[k])
    return {
        "bones": len(skel.get("bones", [])),
        "slots": len(skel.get("slots", [])),
        "meshes": sum(len(s) for sk in skel.get("skins", [])
                      for s in sk.get("attachments", {}).values()),
        "weighted_meshes": sum(
            1 for sk in skel.get("skins", [])
            for s in sk.get("attachments", {}).values()
            for att in s.values()
            if att.get("type") == "mesh" and len(att.get("vertices", [])) >
            len(att.get("uvs", []))
        ),
        "anims": len(anims),
        "anims_with_rotate": anims_with_rotate,
        "tracks": tracks,
        "keys": keys,
        "slot_tracks": sum(len(a.get("slots", {})) for a in anims.values()),
        "width": skel.get("skeleton", {}).get("width"),
        "height": skel.get("skeleton", {}).get("height"),
    }


def rig_dir(root: Path, stem: str) -> Path:
    """Where this bundle's converted rig lives.

    The squad's scratch named the folder after the bundle file (`2dmodel_x`);
    the repo's `spine/` names it after the rig (`x`, the bundle name without the
    `2dmodel_` prefix).  Accept both so this runs against either tree.
    """
    direct = root / stem / "skeleton.json"
    if direct.exists():
        return direct
    if stem.startswith("2dmodel_"):
        return root / stem[len("2dmodel_"):] / "skeleton.json"
    return direct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundles", type=Path)
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--aligned", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--only-after", action="store_true",
                    help="measure the 'after' tree alone (fidelity run)")
    args = ap.parse_args()

    bundles = sorted(args.bundles.glob("*.bundle"))
    bundles = [b for i, b in enumerate(bundles) if i % args.nshard == args.shard]

    rows = []
    for i, b in enumerate(bundles):
        bj = rig_dir(args.before, b.stem)
        aj = rig_dir(args.after, b.stem)
        row = {"bundle": b.name, "before_exists": bj.exists(),
               "after_exists": aj.exists()}
        if args.only_after and aj.exists():
            try:
                sc = load_light(b)
                tn = bone_names(sc)
                row["after"] = compare_scene(sc, tn, b.name, aj, args.samples, False)
                row["after_aligned"] = compare_scene(sc, tn, b.name, aj,
                                                     args.samples, True)
                row["stats_after"] = rotate_stats(aj)
                print(f"[{i+1}/{len(bundles)}] {b.stem}: "
                      f"rot med {row['after'].get('rot_med', float('nan')):.5f} "
                      f"max {row['after'].get('rot_max', float('nan')):.3f}", flush=True)
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {e}"
                print(f"[{i+1}/{len(bundles)}] {b.stem}: ERROR {e}", flush=True)
            rows.append(row)
            continue
        if not (bj.exists() and aj.exists()):
            rows.append(row)
            print(f"[{i+1}/{len(bundles)}] {b.stem}: skipped "
                  f"(before={bj.exists()} after={aj.exists()})", flush=True)
            continue
        try:
            sc = load_light(b)
            tn = bone_names(sc)
            row["before"] = compare_scene(sc, tn, b.name, bj, args.samples, False)
            row["after"] = compare_scene(sc, tn, b.name, aj, args.samples, False)
            row["before_aligned"] = compare_scene(sc, tn, b.name, bj, args.samples, True)
            row["after_aligned"] = compare_scene(sc, tn, b.name, aj, args.samples, True)
            row["stats_before"] = rotate_stats(bj)
            row["stats_after"] = rotate_stats(aj)
            print(f"[{i+1}/{len(bundles)}] {b.stem}: "
                  f"rot med {row['before'].get('rot_med', float('nan')):.3f} -> "
                  f"{row['after'].get('rot_med', float('nan')):.3f} deg | "
                  f"pos med {row['before'].get('pos_med', float('nan')):.4f} -> "
                  f"{row['after'].get('pos_med', float('nan')):.4f} %", flush=True)
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"[{i+1}/{len(bundles)}] {b.stem}: ERROR {e}", flush=True)
        rows.append(row)

    args.out.write_text(json.dumps(rows, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
