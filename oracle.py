"""Pose oracle: Unity bundle (ground truth) vs converted Spine 3.8 JSON.

Original side
    Sample the AnimationClip curves straight out of the bundle, override the
    Transform hierarchy with them, and walk the hierarchy to a world 4x4 per
    Transform.  Purely the converter's own maths (``make_world`` /
    ``clip_overrides``) on values UnityPy read -- no Spine involved.

Converted side
    A minimal Spine 3.8 evaluator: bone setup pose + rotate / translate /
    scale / shear timelines (linear, stepped and bezier), parent inheritance
    ``normal``.  Mirrors ``Bone.updateWorldTransform`` from spine-runtimes 3.8.

Both world matrices go through the same ``decompose2d``, so rotation / scale /
shear are compared on one convention.  The exporter bakes a uniform world
scale onto the ``root`` bone; the oracle divides it out, so positions are
compared in Unity units and reported as a percentage of the rig's own width.

Usage::

    python oracle.py <bundle> <skeleton.json> [--samples 8] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import unity_to_spine as u2s  # noqa: E402
from uscene import bone_names, load_light  # noqa: E402


# ---------------------------------------------------------------------------
# Spine 3.8 evaluator
# ---------------------------------------------------------------------------
SETUP_KEYS = ("x", "y", "rotation", "scaleX", "scaleY", "shearX", "shearY")
SETUP_DEFAULT = {"x": 0.0, "y": 0.0, "rotation": 0.0, "scaleX": 1.0,
                 "scaleY": 1.0, "shearX": 0.0, "shearY": 0.0}


def _bezier_at(t: float, cx1, cy1, cx2, cy2) -> float:
    """Spine curve: cubic bezier through (0,0),(cx1,cy1),(cx2,cy2),(1,1)."""
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) * 0.5
        u = 1.0 - mid
        x = 3 * u * u * mid * cx1 + 3 * u * mid * mid * cx2 + mid ** 3
        if x < t:
            lo = mid
        else:
            hi = mid
    p = (lo + hi) * 0.5
    u = 1.0 - p
    return 3 * u * u * p * cy1 + 3 * u * p * p * cy2 + p ** 3


class Timeline:
    """One Spine 3.8 bone timeline (list of keys with optional curve)."""

    def __init__(self, keys, fields):
        self.times = [float(k.get("time", 0.0)) for k in keys]
        self.fields = fields
        self.vals = [[float(k.get(f, d)) for f, d in fields] for k in keys]
        self.curves = [k.get("curve") for k in keys]

    def sample(self, t: float):
        times = self.times
        if not times:
            return None
        if t <= times[0]:
            return list(self.vals[0]) if t >= times[0] else None
        if t >= times[-1]:
            return list(self.vals[-1])
        i = 0
        while i + 1 < len(times) and times[i + 1] <= t:
            i += 1
        t0, t1 = times[i], times[i + 1]
        v0, v1 = self.vals[i], self.vals[i + 1]
        span = t1 - t0
        p = 0.0 if span <= 0 else (t - t0) / span
        c = self.curves[i]
        if c == "stepped":
            p = 0.0
        elif isinstance(c, (list, tuple)) and len(c) == 4:
            p = _bezier_at(p, *[float(x) for x in c])
        return [a + (b - a) * p for a, b in zip(v0, v1)]


class SpineSkeleton:
    def __init__(self, skel: dict):
        self.raw = skel
        self.bones = []
        self.index = {}
        for b in skel.get("bones", []):
            entry = {k: float(b.get(k, SETUP_DEFAULT[k])) for k in SETUP_KEYS}
            entry["name"] = b["name"]
            entry["parent"] = b.get("parent")
            self.index[b["name"]] = len(self.bones)
            self.bones.append(entry)
        self.parent_idx = [
            self.index.get(b["parent"]) if b["parent"] else None for b in self.bones
        ]
        self.anims = {}
        for name, anim in skel.get("animations", {}).items():
            tls = {}
            for bone, tracks in anim.get("bones", {}).items():
                bt = {}
                if "rotate" in tracks:
                    bt["rotate"] = Timeline(tracks["rotate"], [("angle", 0.0)])
                if "translate" in tracks:
                    bt["translate"] = Timeline(tracks["translate"], [("x", 0.0), ("y", 0.0)])
                if "scale" in tracks:
                    bt["scale"] = Timeline(tracks["scale"], [("x", 1.0), ("y", 1.0)])
                if "shear" in tracks:
                    bt["shear"] = Timeline(tracks["shear"], [("x", 0.0), ("y", 0.0)])
                tls[bone] = bt
            self.anims[name] = tls

    def root_scale(self) -> float:
        b = self.bones[0]
        return b["scaleX"] if abs(b["scaleX"]) > 1e-9 else 1.0

    def pose(self, anim_name: str | None, t: float):
        """Setup pose with the animation applied; returns per-bone local TRS."""
        local = [dict(b) for b in self.bones]
        if anim_name is None:
            return local
        tls = self.anims.get(anim_name, {})
        for bone, bt in tls.items():
            i = self.index.get(bone)
            if i is None:
                continue
            L = local[i]
            s = self.bones[i]
            v = bt["rotate"].sample(t) if "rotate" in bt else None
            if v is not None:
                L["rotation"] = s["rotation"] + v[0]
            v = bt["translate"].sample(t) if "translate" in bt else None
            if v is not None:
                L["x"] = s["x"] + v[0]
                L["y"] = s["y"] + v[1]
            v = bt["scale"].sample(t) if "scale" in bt else None
            if v is not None:
                L["scaleX"] = s["scaleX"] * v[0]
                L["scaleY"] = s["scaleY"] * v[1]
            v = bt["shear"].sample(t) if "shear" in bt else None
            if v is not None:
                L["shearX"] = s["shearX"] + v[0]
                L["shearY"] = s["shearY"] + v[1]
        return local

    def world(self, local):
        """spine-runtimes 3.8 Bone.updateWorldTransform, transformMode normal."""
        out = [None] * len(local)
        for i, L in enumerate(local):
            rot, sx, sy = L["rotation"], L["scaleX"], L["scaleY"]
            shx, shy = L["shearX"], L["shearY"]
            rot_y = rot + 90.0 + shy
            la = math.cos(math.radians(rot + shx)) * sx
            lb = math.cos(math.radians(rot_y)) * sy
            lc = math.sin(math.radians(rot + shx)) * sx
            ld = math.sin(math.radians(rot_y)) * sy
            p = self.parent_idx[i]
            if p is None or out[p] is None:
                out[i] = (la, lb, lc, ld, L["x"], L["y"])
                continue
            pa, pb, pc, pd, pwx, pwy = out[p]
            out[i] = (
                pa * la + pb * lc,
                pa * lb + pb * ld,
                pc * la + pd * lc,
                pc * lb + pd * ld,
                pa * L["x"] + pb * L["y"] + pwx,
                pc * L["x"] + pd * L["y"] + pwy,
            )
        return out


def m4_from_world(w):
    a, b, c, d, wx, wy = w
    M = np.eye(4)
    M[0, 0], M[0, 1], M[1, 0], M[1, 1] = a, b, c, d
    M[0, 3], M[1, 3] = wx, wy
    return M


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
def clip_times(clip, n_samples: int, aligned: bool, fps: int = u2s.FPS):
    """Sample times inside the window the exporter actually keyed."""
    stop = float(clip.m_MuscleClip.m_StopTime)
    n_frames = max(1, int(round(stop * fps)))
    last = n_frames / fps
    if aligned:
        if n_frames <= n_samples:
            return [i / fps for i in range(n_frames + 1)]
        step = n_frames / n_samples
        return sorted({round(i * step) / fps for i in range(n_samples + 1)})
    return [last * i / n_samples for i in range(n_samples + 1)]


def clip_rot_flags(clip, hash2tr) -> tuple[bool, bool]:
    """(has quaternion curve, has euler curve) for resolvable transforms."""
    quat = euler = False
    for b in clip.m_ClipBindingConstant.genericBindings:
        if b.typeID == 4 and hash2tr.get(b.path) is not None:
            if b.attribute == 2:
                quat = True
            elif b.attribute == u2s.TRANSFORM_EULER_ATTR:
                euler = True
    return quat, euler


def compare(bundle: Path, skel_path: Path, n_samples: int = 8,
            aligned: bool = False, verbose: bool = False) -> dict:
    sc = load_light(bundle)
    return compare_scene(sc, bone_names(sc), bundle.name, skel_path,
                         n_samples, aligned, verbose)


def compare_scene(sc, tr_name, bundle_name: str, skel_path: Path,
                  n_samples: int = 8, aligned: bool = False,
                  verbose: bool = False) -> dict:
    # The ground-truth side ALWAYS uses the fixed reader (``u2s``): a pre-fix
    # reader would drop the same euler curves on both sides and hide the bug.
    module = u2s
    skel = json.loads(skel_path.read_text())
    sk = SpineSkeleton(skel)
    s = sk.root_scale()

    rig_w = float(skel.get("skeleton", {}).get("width", 0.0)) / s
    rig_h = float(skel.get("skeleton", {}).get("height", 0.0)) / s
    rig_size = max(rig_w, rig_h) or 1.0

    # setup-pose world, for bones the animation never touches
    per_clip = []
    pos_all, rot_all, scale_all = [], [], []
    pos_sane, rot_sane = [], []
    worst = []
    for clip in sc["clips"]:
        name = clip.m_Name
        has_quat, has_euler = clip_rot_flags(clip, sc["hash2tr"])
        if name not in sk.anims:
            per_clip.append({"clip": name, "status": "missing-in-json",
                             "quat": has_quat, "euler": has_euler})
            continue
        try:
            sampler, stop, _ = module.decode_clip(clip)
        except Exception as e:  # noqa: BLE001
            per_clip.append({"clip": name, "status": f"decode-failed: {e}",
                             "quat": has_quat, "euler": has_euler})
            continue
        times = clip_times(clip, n_samples, aligned)
        pos_e, rot_e, sc_e = [], [], []
        pos_s, rot_s = [], []          # "sane" subset (see below)
        n_degen = n_runaway = 0
        for t in times:
            ov = module.clip_overrides(clip, sc["hash2tr"], sampler, t)
            uworld = module.make_world(sc["TR"], ov)
            sworld = sk.world(sk.pose(name, t))
            for tr in sc["TR"]:
                bname = tr_name.get(tr)
                i = sk.index.get(bname)
                if i is None:
                    continue
                MU = uworld(tr)
                du = module.decompose2d(MU)
                ds = module.decompose2d(m4_from_world(sworld[i]))
                dp = math.hypot(ds["x"] / s - du["x"], ds["y"] / s - du["y"])
                dp_pct = dp / rig_size * 100.0
                dr = abs(u2s.norm180(ds["rotation"] - du["rotation"]))
                dsx = abs(ds["scaleX"] / s - du["scaleX"])
                dsy = abs(ds["scaleY"] / s - du["scaleY"])
                pos_e.append(dp_pct)
                rot_e.append(dr)
                sc_e.append(max(dsx, dsy))
                # A bone whose world basis has collapsed has no defined
                # orientation, and a bone thrown far outside the rig is the
                # clip sampler extrapolating before its first key -- both are
                # reproduced identically on each side, so they say nothing
                # about the rotation conversion.  Counted, then set aside.
                lx = math.hypot(MU[0, 0], MU[1, 0])
                ly = math.hypot(MU[0, 1], MU[1, 1])
                lz = math.hypot(MU[0, 2], MU[1, 2])
                degen = lx < 1e-3 or (ly < 1e-3 and lz < 1e-3)
                runaway = math.hypot(du["x"], du["y"]) > 10.0 * rig_size
                if degen:
                    n_degen += 1
                if runaway:
                    n_runaway += 1
                if not (degen or runaway):
                    pos_s.append(dp_pct)
                    rot_s.append(dr)
                worst.append((dr, dp_pct, name, bname, round(t, 3),
                              bool(degen), bool(runaway)))
        if not pos_e:
            per_clip.append({"clip": name, "status": "no-bones",
                             "quat": has_quat, "euler": has_euler})
            continue
        pos_all += pos_e
        rot_all += rot_e
        scale_all += sc_e
        pos_sane += pos_s
        rot_sane += rot_s
        entry = {
            "clip": name, "status": "ok", "samples": len(pos_e),
            "quat": has_quat, "euler": has_euler,
            "degenerate": n_degen, "runaway": n_runaway,
            "pos_med": float(np.median(pos_e)), "pos_max": float(np.max(pos_e)),
            "pos_p99": float(np.percentile(pos_e, 99)),
            "rot_med": float(np.median(rot_e)), "rot_max": float(np.max(rot_e)),
            "rot_p99": float(np.percentile(rot_e, 99)),
            "scale_max": float(np.max(sc_e)),
        }
        if pos_s:
            entry.update({
                "pos_med_sane": float(np.median(pos_s)),
                "pos_max_sane": float(np.max(pos_s)),
                "rot_med_sane": float(np.median(rot_s)),
                "rot_max_sane": float(np.max(rot_s)),
            })
        per_clip.append(entry)
        if verbose:
            print(f"    {name}: pos med {np.median(pos_e):.4f}% max {np.max(pos_e):.4f}%  "
                  f"rot med {np.median(rot_e):.4f} max {np.max(rot_e):.4f} deg")

    worst.sort(reverse=True)
    res = {
        "bundle": bundle_name,
        "skeleton": str(skel_path),
        "rig_size": rig_size,
        "root_scale": s,
        "clips": per_clip,
        "n": len(pos_all),
    }
    if pos_all:
        res.update({
            "pos_med": float(np.median(pos_all)),
            "pos_p99": float(np.percentile(pos_all, 99)),
            "pos_max": float(np.max(pos_all)),
            "rot_med": float(np.median(rot_all)),
            "rot_p99": float(np.percentile(rot_all, 99)),
            "rot_max": float(np.max(rot_all)),
            "scale_max": float(np.max(scale_all)),
            "n_sane": len(pos_sane),
            "worst": [
                {"rot_err": w[0], "pos_err_pct": w[1], "clip": w[2],
                 "bone": w[3], "t": w[4], "degenerate": w[5], "runaway": w[6]}
                for w in worst[:8]
            ],
        })
    if pos_sane:
        res.update({
            "pos_med_sane": float(np.median(pos_sane)),
            "pos_p99_sane": float(np.percentile(pos_sane, 99)),
            "pos_max_sane": float(np.max(pos_sane)),
            "rot_med_sane": float(np.median(rot_sane)),
            "rot_p99_sane": float(np.percentile(rot_sane, 99)),
            "rot_max_sane": float(np.max(rot_sane)),
        })
        sane_worst = [w for w in worst if not (w[5] or w[6])]
        sane_worst.sort(reverse=True)
        res["worst_sane"] = [
            {"rot_err": w[0], "pos_err_pct": w[1], "clip": w[2],
             "bone": w[3], "t": w[4]} for w in sane_worst[:8]
        ]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("skeleton", type=Path)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--aligned", action="store_true",
                    help="sample only on exporter frame times (isolates resampling error)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    r = compare(args.bundle, args.skeleton, args.samples, args.aligned, verbose=True)
    print(json.dumps({k: v for k, v in r.items() if k != "clips"}, indent=1))
    if args.json:
        args.json.write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
