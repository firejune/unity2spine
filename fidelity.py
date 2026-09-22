"""Write fidelity.json: the oracle's verdict per rig, in a consumable form.

Deterministic: reads the oracle rows and the 3D census already on disk and
writes the same file for the same inputs.  No sampling, no randomness.

Usage: python fidelity.py <oracle-glob> <probe_3d.json> <out.json>
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

# Thresholds live here, next to the distribution that justifies them.
# Rotation is the binding axis: position error tracks it (a bone's position is
# its parent chain's rotation applied to fixed offsets), so a rig that passes
# on rotation passes on position in this corpus -- fidelity.json carries both
# so a consumer can re-decide.
STRICT_ROT_DEG = 1.0
STRICT_POS_PCT = 1.0
LOOSE_ROT_DEG = 15.0
LOOSE_POS_PCT = 5.0


def load_rows(pattern: str):
    """Oracle shards named by `pattern`.  A pattern with a path separator (or an
    absolute one) is taken as given, so this runs from the repo root; a bare
    pattern still resolves next to this file, as it did in the squad's scratch."""
    rows = []
    where = pattern if (os.sep in pattern or os.path.isabs(pattern)) else str(HERE / pattern)
    for f in sorted(glob.glob(where)):
        rows += json.loads(Path(f).read_text())
    return rows


def describe(name, arr):
    a = np.asarray([x for x in arr if x is not None], dtype=float)
    if not a.size:
        return {"n": 0}
    return {
        "n": int(a.size),
        "min": float(a.min()), "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)), "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)), "max": float(a.max()),
        "histogram": histogram(a),
    }


def histogram(a):
    """Log-spaced buckets -- the errors span six orders of magnitude."""
    edges = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 5.0, 15.0, 45.0, 90.0, 1e9]
    labels = ["<0.001", "0.001-0.01", "0.01-0.1", "0.1-1", "1-5", "5-15",
              "15-45", "45-90", ">90"]
    counts = [int(((a >= edges[i]) & (a < edges[i + 1])).sum())
              for i in range(len(labels))]
    return dict(zip(labels, counts))


def gap_report(a, min_side=5):
    """Is the distribution cleanly bimodal?

    Reports the widest empty band overall and the widest band that actually
    splits the rigs (at least ``min_side`` on each side) -- an outlier at the
    very bottom makes the global maximum useless as a cut.
    """
    s = np.sort(np.asarray(a, dtype=float))
    if s.size < 2 * min_side:
        return {"bimodal": False, "reason": "too few rigs"}
    ls = np.log10(np.maximum(s, 1e-6))
    gaps = np.diff(ls)
    i = int(np.argmax(gaps))
    interior = range(min_side - 1, s.size - min_side)
    j = max(interior, key=lambda k: gaps[k])
    return {
        "largest_log10_gap": float(gaps[i]),
        "gap_between": [float(s[i]), float(s[i + 1])],
        "rigs_below_gap": int(i + 1), "rigs_above_gap": int(s.size - i - 1),
        "widest_splitting_band": {
            "log10_width": float(gaps[j]),
            "between": [float(s[j]), float(s[j + 1])],
            "rigs_below": int(j + 1), "rigs_above": int(s.size - j - 1),
        },
        # a clean split needs a decade-wide empty band with a real share of the
        # rigs on each side; the global maximum here is a lone low outlier
        "bimodal": bool(gaps[i] >= 1.0 and min_side <= i + 1 <= s.size - min_side),
        "splitting_band_is_clean": bool(gaps[j] >= 0.5),
    }


def main():
    pattern, probe3d, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    rows = load_rows(pattern)
    nonplanar = {r["bundle"]: r for r in json.loads(probe3d.read_text())}

    rigs = []
    for r in rows:
        a = r.get("after")
        al = r.get("after_aligned") or {}
        if not a or not a.get("n"):
            continue
        stem = r["bundle"].replace(".bundle", "")
        p3 = nonplanar.get(stem, {})
        ok_clips = [c for c in a["clips"] if c.get("status") == "ok"]
        rigs.append({
            "bundle": stem,
            "clips": len(ok_clips),
            "samples": a["n"],
            "samples_set_aside": a["n"] - a.get("n_sane", a["n"]),
            "rotation_deg": {
                "median": al.get("rot_med_sane", a.get("rot_med_sane")),
                "p99": al.get("rot_p99_sane", a.get("rot_p99_sane")),
                "worst": al.get("rot_max_sane", a.get("rot_max_sane")),
                "worst_free_times": a.get("rot_max_sane"),
            },
            "position_pct": {
                "median": al.get("pos_med_sane", a.get("pos_med_sane")),
                "p99": al.get("pos_p99_sane", a.get("pos_p99_sane")),
                "worst": al.get("pos_max_sane", a.get("pos_max_sane")),
                "worst_free_times": a.get("pos_max_sane"),
            },
            "non_planar_rest_transforms": p3.get("non_planar_rest"),
            "transforms": p3.get("transforms"),
            "worst_bone": (al.get("worst_sane") or a.get("worst_sane") or [{}])[0],
        })

    rot = [x["rotation_deg"]["worst"] for x in rigs
           if x["rotation_deg"]["worst"] is not None]
    pos = [x["position_pct"]["worst"] for x in rigs
           if x["position_pct"]["worst"] is not None]
    gap = gap_report(rot)

    for x in rigs:
        rw = x["rotation_deg"]["worst"]
        pw = x["position_pct"]["worst"]
        x["pose_faithful"] = bool(
            rw is not None and pw is not None
            and rw <= STRICT_ROT_DEG and pw <= STRICT_POS_PCT)
        x["pose_faithful_loose"] = bool(
            rw is not None and pw is not None
            and rw <= LOOSE_ROT_DEG and pw <= LOOSE_POS_PCT)

    doc = {
        "what": "per-rig agreement between the converted Spine 3.8 skeleton and "
                "the Unity bundle it came from, measured by oracle.py",
        "units": {"rotation": "degrees of world rotation error",
                  "position": "percent of the rig's own bounding size"},
        "sampling": "8 times per clip at the exporter's own frame times "
                    "(--aligned); every bone present in both. Samples on bones "
                    "whose world basis has collapsed, or that the clip sampler "
                    "throws far outside the rig, are counted and set aside.",
        "thresholds": {
            "strict": {"rotation_deg": STRICT_ROT_DEG, "position_pct": STRICT_POS_PCT,
                       "field": "pose_faithful"},
            "loose": {"rotation_deg": LOOSE_ROT_DEG, "position_pct": LOOSE_POS_PCT,
                      "field": "pose_faithful_loose"},
            "why_two": "the worst-bone error is NOT cleanly bimodal (see "
                       "distribution.bimodality), so no single cut is honest: "
                       "strict = 'every bone lands where Unity puts it', loose = "
                       "'the silhouette is right, some out-of-plane bones drift'.",
        },
        "distribution": {
            "worst_bone_rotation_deg": describe("rot", rot),
            "worst_bone_position_pct": describe("pos", pos),
            "bimodality": gap,
        },
        "summary": {
            "rigs": len(rigs),
            "pose_faithful_strict": sum(1 for x in rigs if x["pose_faithful"]),
            "pose_faithful_loose": sum(1 for x in rigs if x["pose_faithful_loose"]),
        },
        "rigs": sorted(rigs, key=lambda x: x["bundle"]),
    }
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: doc[k] for k in
                      ("summary", "distribution", "thresholds")}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
