#!/usr/bin/env python3
"""Export a Unity 2D rigged character AssetBundle to Spine 3.8.

Usage::

    python unity_to_spine.py path/to/__data
    python unity_to_spine.py path/to/__data --output path/to/spine_editor
    python unity_to_spine.py path/to/__data --runtime
    python unity_to_spine.py path/to/__data --gif
    python unity_to_spine.py path/to/__data --gif-only --gif-width 720

Pass the bundle file or directory that UnityPy accepts (typically ``__data``).
Outputs ``skeleton.json`` + ``skeleton.atlas`` (+ texture pages) under
``<bundle-parent>/spine_editor/`` by default (Spine Editor import layout),
or ``--output`` / ``--runtime`` for the runtime layout.
With ``--gif``, each ``AnimationClip`` is also rasterised to
``<bundle-parent>/gifs/anim_<name>.gif``.

This script is self-contained: it reads the Unity bundle, decodes skinned
meshes / sprites / animation clips, and writes a Spine skeleton.  It does not
import :mod:`assemble` or :mod:`to_spine`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
import zlib
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import UnityPy
from PIL import Image
from UnityPy.helpers.MeshHelper import MeshHandler

SPINE_VERSION = "3.8.99"
FPS = int(os.environ.get("SPINE_FPS", "30"))
TARGET_W = int(os.environ.get("SPINE_TARGET_W", "1600"))
PAD = 0.04
GIF_W = int(os.environ.get("GIF_W", "720"))
GIF_FPS = int(os.environ.get("GIF_FPS", "24"))
GIF_BG = os.environ.get("GIF_BG", "ffffff")
GIF_WORKERS = int(os.environ.get("GIF_WORKERS", "0"))  # 0 = auto (cpu count, cap 8)
EPS = 1e-4
ATLAS_SUFFIX_RE = re.compile(r"(parts?_?\d*)", re.I)

# Unity AnimationClip generic binding IDs (Transform=4, GameObject=1, SMR=137, SpriteRenderer=212).
# Transform attributes: 1=m_LocalPosition, 2=m_LocalRotation, 3=m_LocalScale,
# 4=m_LocalEulerAngles (kBindTransformEuler, degrees, ZXY order).
TRANSFORM_EULER_ATTR = 4
GO_TYPE = 1
GO_ACTIVE_ATTR = 2086281974
SMR_TYPE = 137
SMR_COLOR_A_ATTR = 2108656497
SPRITE_RENDERER_TYPE = 212
SPRITE_COLOR_A_ATTR = 304273561  # zlib.crc32(b"m_Color.a") & 0xFFFFFFFF
# Renderer tint curves; Spine's slot colour multiplies the attachment the same
# way Unity's m_Color does, and the alpha channel already takes this path.
COLOR_RGB_ATTRS = {
    2526845255: 0,  # m_Color.r
    4215373228: 1,  # m_Color.g
    2334886179: 2,  # m_Color.b
}
COMPONENT_ENABLED_ATTR = 3305885265  # zlib.crc32(b"m_Enabled") & 0xFFFFFFFF
# Material float curve for additional opacity/alpha (customType 22 RendererMaterial, e.g. _AdditionalAlpha).
# Evaluates to (zlib.crc32(b"_AdditionalAlpha") & 0x0FFFFFFF) | 0x80000000.
MATERIAL_ADDITIONAL_ALPHA_ATTR = 2274245065

GIZMO_SPRITE_NAMES = frozenset({
    "BoneJoint", "BoneNoJoint", "BoneScaled", "IKControl",
    "splineControl", "splineMiddleControl", "bone_joint", "bone_no_joint",
    "128_glow_texture",
})
VIS_EPS = 0.01


def material_blend_mode(mat_name: str | None) -> str | None:
    """Map Unity material/shader name to Spine 3.8 blend mode (multiply/additive/screen)."""
    if not mat_name:
        return None
    low = mat_name.lower()
    if "multiply" in low:
        return "multiply"
    if "additive" in low or "add" in low:
        return "additive"
    if "screen" in low:
        return "screen"
    return None


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------
def mat4(m) -> np.ndarray:
    return np.array(
        [[getattr(m, f"e{r}{c}") for c in range(4)] for r in range(4)],
        dtype=np.float64,
    )


def quat_to_m3(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def make_world(TR, overrides=None):
    """Local->world matrices; optional per-bone overrides without copying TR."""
    overrides = overrides or {}
    wcache = {}

    def trs(pid):
        t = TR[pid]
        o = overrides.get(pid)
        if not o:
            return t["pos"], t["rot"], t["scale"]
        return (
            o.get("pos", t["pos"]),
            o.get("rot", t["rot"]),
            o.get("scale", t["scale"]),
        )

    def world(pid):
        if pid in wcache:
            return wcache[pid]
        chain = []
        q = pid
        while q and q in TR:
            chain.append(q)
            q = TR[q]["father"]
        M = np.eye(4)
        for q in reversed(chain):
            pos, rot, scale = trs(q)
            T = np.eye(4)
            T[:3, 3] = pos
            R = np.eye(4)
            R[:3, :3] = quat_to_m3(rot)
            S = np.diag([*scale, 1.0])
            M = M @ T @ R @ S
        wcache[pid] = M
        return M

    return world


def local_matrix(pos, rot, scale):
    T = np.eye(4)
    T[:3, 3] = pos
    R = np.eye(4)
    R[:3, :3] = quat_to_m3(rot)
    S = np.diag([scale[0], scale[1], scale[2], 1.0])
    return T @ R @ S


def norm180(a):
    return (a + 180.0) % 360.0 - 180.0


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    )


def normalize_tilted_pitch(TR):
    """Detect and normalize dummy 3D pitch rotation pairs (e.g. parent Rx(±90) and children planar tilt)
    so 2D Spine bones are upright with positive vertical scales and correct 2D positions."""
    tilted_pos_children_neg = set()
    tilted_pos_children_pos = set()
    tilted_rot_children_neg = set()
    tilted_rot_children_pos = set()

    q_rx_neg90 = (-0.7071067811865475, 0.0, 0.0, 0.7071067811865475)
    q_rx_pos90 = (0.7071067811865475, 0.0, 0.0, 0.7071067811865475)

    for p_tr, p_t in list(TR.items()):
        rx, ry, rz, rw = p_t["rot"]
        is_neg90 = abs(rx - (-0.7071068)) < 0.05 and abs(rw - 0.7071068) < 0.05 and abs(ry) < 0.05 and abs(rz) < 0.05
        is_pos90 = abs(rx - 0.7071068) < 0.05 and abs(rw - 0.7071068) < 0.05 and abs(ry) < 0.05 and abs(rz) < 0.05
        if not (is_neg90 or is_pos90):
            continue

        q_comp = q_rx_pos90 if is_pos90 else q_rx_neg90

        children = [c_tr for c_tr, c_t in TR.items() if c_t.get("father") == p_tr]
        if not children:
            continue
        matched = []
        for c_tr in children:
            crx, cry, crz, crw = TR[c_tr]["rot"]
            q_comb = quat_mul(q_comp, TR[c_tr]["rot"])
            R_comb = quat_to_m3(q_comb)
            is_planar_direct = abs(R_comb[0, 2]) < 0.05 and abs(R_comb[1, 2]) < 0.05 and abs(abs(R_comb[2, 2]) - 1.0) < 0.05
            is_rx90 = abs(crx * crw) > 0.45 and abs(cry) < 0.05 and abs(crz) < 0.05
            is_ryrz = abs(cry * crz) > 0.45 and abs(crx) < 0.05 and abs(crw) < 0.05

            if is_rx90 or is_ryrz or is_planar_direct:
                matched.append((c_tr, "direct", []))
            else:
                is_identity = abs(abs(crw) - 1.0) < 0.05 and abs(crx) < 0.05 and abs(cry) < 0.05 and abs(crz) < 0.05
                if is_identity:
                    grandchildren = [gc_tr for gc_tr, gc_t in TR.items() if gc_t.get("father") == c_tr]
                    gc_rx90 = []
                    for gc_tr in grandchildren:
                        gq_comb = quat_mul(q_comp, TR[gc_tr]["rot"])
                        gR_comb = quat_to_m3(gq_comb)
                        g_is_planar = abs(gR_comb[0, 2]) < 0.05 and abs(gR_comb[1, 2]) < 0.05 and abs(abs(gR_comb[2, 2]) - 1.0) < 0.05
                        grx, gry, grz, grw = TR[gc_tr]["rot"]
                        g_is_rx90 = (abs(grx * grw) > 0.45 and abs(gry) < 0.05 and abs(grz) < 0.05) or (abs(gry * grz) > 0.45 and abs(grx) < 0.05 and abs(grw) < 0.05)
                        if g_is_rx90 or g_is_planar:
                            gc_rx90.append(gc_tr)
                    if gc_rx90:
                        matched.append((c_tr, "intermediate", gc_rx90))
        if len(matched) >= len(children) * 0.5:
            p_t["rot"] = (0.0, 0.0, 0.0, 1.0)
            t_pos_set = tilted_pos_children_pos if is_pos90 else tilted_pos_children_neg
            t_rot_set = tilted_rot_children_pos if is_pos90 else tilted_rot_children_neg
            for item in matched:
                c_tr, mtype, gcs = item
                ct = TR[c_tr]
                px, py, pz = ct["pos"]
                ct["pos"] = (px, -pz, py) if is_pos90 else (px, pz, -py)
                t_pos_set.add(c_tr)
                if mtype == "direct":
                    ct["rot"] = quat_mul(q_comp, ct["rot"])
                    t_rot_set.add(c_tr)
                else:
                    for gc_tr in gcs:
                        gct = TR[gc_tr]
                        gpx, gpy, gpz = gct["pos"]
                        if abs(gpz) > 1e-4:
                            gct["pos"] = (gpx, -gpz, gpy) if is_pos90 else (gpx, gpz, -gpy)
                            t_pos_set.add(gc_tr)
                        gct["rot"] = quat_mul(q_comp, gct["rot"])
                        t_rot_set.add(gc_tr)
            for c_tr in children:
                t_pos_set.add(c_tr)

    tilted_pos_children = tilted_pos_children_neg | tilted_pos_children_pos
    tilted_rot_children = tilted_rot_children_neg | tilted_rot_children_pos
    return (
        tilted_pos_children, tilted_rot_children,
        tilted_pos_children_neg, tilted_pos_children_pos,
        tilted_rot_children_neg, tilted_rot_children_pos,
    )


def _normalize_face_parts(TR, parts, go_name, tr2go, repairs):
    """Normalize inverted local scale.x on facial expression parts when sibling/base parts are non-inverted."""
    for p in parts:
        name = p.get("name", "").lower()
        tr = p.get("tr_pid") or p.get("go_tr_pid")
        if not tr or tr not in TR:
            continue
        g_name = go_name.get(tr2go.get(tr, 0), "").lower()
        if any(k in name or k in g_name for k in ("face", "smile", "eye", "mouth", "embarrass", "surprised")):
            t = TR[tr]
            if t["scale"][0] < 0:
                t["scale"] = (abs(t["scale"][0]), t["scale"][1], t["scale"][2])
                repairs.append({
                    "part": p.get("name", ""),
                    "fix": "normalize-face-scale-x",
                    "scale": t["scale"],
                })


def decompose2d(M):
    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    tx, ty = M[0, 3], M[1, 3]

    det = a * d - b * c
    rotation = math.degrees(math.atan2(c, a))
    scaleX = math.hypot(a, c)

    # If Y basis vector is collapsed into 3D Z depth (e.g. quad rotated in 3D),
    # project Z basis vector (M[0, 2], M[1, 2]) into 2D Y axis when it better represents the 2D vertical axis.
    len_y = math.hypot(b, d)
    ez_x, ez_y = M[0, 2], M[1, 2]
    len_z = math.hypot(ez_x, ez_y)
    if len_z > len_y and len_z > 1e-3:
        b, d = ez_x, ez_y
        len_y = len_z
        det = a * d - b * c

    if det < 0:
        # Proper 2D reflection/flip in Spine: negative scaleY instead of shearY=-180
        scaleY = -len_y
        rotationY = math.degrees(math.atan2(-d, -b))
    else:
        scaleY = len_y
        rotationY = math.degrees(math.atan2(d, b))

    if len_y < 1e-3:
        shearY = 0.0
    else:
        shearY = norm180(rotationY - 90.0 - rotation)
        if abs(shearY) > 45.0:
            shearY = 0.0

    return dict(
        x=float(tx), y=float(ty), rotation=float(rotation),
        scaleX=float(scaleX), scaleY=float(scaleY), shearY=float(shearY),
    )


def trs_to_spine(t):
    return decompose2d(local_matrix(t["pos"], t["rot"], t["scale"]))


def r2(v):
    return round(float(v), 4)


def atlas_key_from_name(name: str | None) -> str | None:
    if not name:
        return None
    m = ATLAS_SUFFIX_RE.search(name)
    return m.group(1).lower() if m else None


def skin_part(part, world):
    if part["kind"] == "sprite":
        M = world(part["tr_pid"])
        return (M @ part["sv2"].T).T[:, :2]

    bones, bind, vh = part["bones"], part["bind"], part["vh"]
    if part["single"]:
        if len(bones) == 0:
            target_tr = part.get("go_tr_pid") or part.get("tr_pid")
            if target_tr:
                M = world(target_tr)
                dy = vh[:, 1].max() - vh[:, 1].min()
                dz = vh[:, 2].max() - vh[:, 2].min()
                y_idx = 2 if (dy < 1e-3 and dz > 1e-3) else 1
                pts = np.c_[vh[:, 0], vh[:, y_idx], np.zeros(len(vh)), np.ones(len(vh))]
                return (M @ pts.T).T[:, :2]
            return vh[:, :2]
        M = world(bones[0]) @ bind[0]
        return (M @ vh.T).T[:, :2]

    bi, bw = part["bi"], part["bw"]
    vc, k = bi.shape
    BW = [world(b) @ bind[i] for i, b in enumerate(bones)]
    pos = np.zeros((vc, 3))
    for ki in range(k):
        w = bw[:, ki]
        if not np.any(w):
            continue
        for bidx in np.unique(bi[:, ki]):
            sel = bi[:, ki] == bidx
            pts = (BW[bidx] @ vh[sel].T).T[:, :3]
            pos[sel] += pts * w[sel, None]
    return pos[:, :2]


def skin_all(parts, world):
    return [skin_part(p, world) for p in parts]


def bounds_of(positions, pad=0.0):
    allxy = np.vstack(positions)
    minx, miny = allxy.min(0)
    maxx, maxy = allxy.max(0)
    w, h = maxx - minx, maxy - miny
    return (minx - w * pad, miny - h * pad, maxx + w * pad, maxy + h * pad)


def make_canvas_tf(bounds, target_w):
    minx, miny, maxx, maxy = bounds
    scale = target_w / (maxx - minx)
    W = int(round((maxx - minx) * scale))
    H = int(round((maxy - miny) * scale))

    def to_canvas(xy):
        cx = (xy[:, 0] - minx) * scale
        cy = (maxy - xy[:, 1]) * scale
        return np.c_[cx, cy]

    return to_canvas, W, H


def atlases_bgra(atlases: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    """PIL atlas pages -> OpenCV BGRA arrays for rasterisation."""
    out = {}
    for key, page in atlases.items():
        rgba = np.array(page["image"].convert("RGBA"))
        out[key] = np.ascontiguousarray(rgba[:, :, [2, 1, 0, 3]])
    return out


def prepare_raster_cache(parts, atlas_bgra):
    """Pre-crop atlas pages per part; mark rigid parts for single-warp path."""
    cache = []
    for p in parts:
        atlas = atlas_bgra[p["atlas"]]
        ah, aw = atlas.shape[:2]
        src = p["src_px"]
        pad = 2
        x0 = max(0, int(np.floor(src[:, 0].min())) - pad)
        y0 = max(0, int(np.floor(src[:, 1].min())) - pad)
        x1 = min(aw, int(np.ceil(src[:, 0].max())) + pad)
        y1 = min(ah, int(np.ceil(src[:, 1].max())) + pad)
        src_local = src - np.array([x0, y0], dtype=np.float64)
        rigid = p["kind"] == "sprite" or (p["kind"] == "mesh" and p["single"])
        n = len(src_local)
        aff_idx = (0, max(1, n // 2), max(2, n - 1)) if n >= 3 else (0, 0, 0)
        cache.append(dict(
            part=p,
            atlas_crop=np.ascontiguousarray(atlas[y0:y1, x0:x1]),
            src_local=src_local,
            faces=p["faces"],
            rigid=rigid,
            aff_idx=aff_idx,
        ))
    return cache


def is_part_visible(part, atlases_bgra) -> bool:
    """Return True if part samples at least one non-transparent pixel from its atlas."""
    atlas_name = part.get("atlas")
    if not atlas_name or atlas_name not in atlases_bgra:
        return False
    atlas_arr = atlases_bgra[atlas_name]
    src_px = part.get("src_px")
    if src_px is None or len(src_px) == 0:
        return False
    ah, aw = atlas_arr.shape[:2]
    x0, y0 = int(np.floor(src_px[:, 0].min())), int(np.floor(src_px[:, 1].min()))
    x1, y1 = int(np.ceil(src_px[:, 0].max())), int(np.ceil(src_px[:, 1].max()))
    x0, x1 = max(0, x0), min(aw, x1)
    y0, y1 = max(0, y0), min(ah, y1)
    if x1 <= x0 or y1 <= y0:
        return False
    crop_a = atlas_arr[y0:y1, x0:x1, 3]
    if np.count_nonzero(crop_a > 10) == 0:
        return False
    xs = np.clip(np.round(src_px[:, 0]).astype(int), 0, aw - 1)
    ys = np.clip(np.round(src_px[:, 1]).astype(int), 0, ah - 1)
    if np.count_nonzero(atlas_arr[ys, xs, 3] > 10) > 0:
        return True
    faces = part.get("faces")
    if faces is not None and len(faces) > 0:
        tri_centers = src_px[faces].mean(axis=1)
        tc_x = np.clip(np.round(tri_centers[:, 0]).astype(int), 0, aw - 1)
        tc_y = np.clip(np.round(tri_centers[:, 1]).astype(int), 0, ah - 1)
        if np.count_nonzero(atlas_arr[tc_y, tc_x, 3] > 10) > 0:
            return True
    return False


def _composite_layer(canvas, layer, px0, py0):
    la = layer[:, :, 3:4].astype(np.float32) / 255.0
    lc = layer[:, :, :3].astype(np.float32)
    cv_roi = canvas[py0:py0 + layer.shape[0], px0:px0 + layer.shape[1]]
    ca = cv_roi[:, :, 3:4]
    out_a = la + ca * (1 - la)
    out_rgb = lc * la + cv_roi[:, :, :3] * ca * (1 - la)
    with np.errstate(invalid="ignore", divide="ignore"):
        out_rgb = np.where(out_a > 0, out_rgb / np.maximum(out_a, 1e-6), 0)
    cv_roi[:, :, :3] = out_rgb
    cv_roi[:, :, 3:4] = out_a


def _rasterize_rigid(entry, cpos, px0, py0, pw, ph):
    crop = entry["atlas_crop"]
    src_local = entry["src_local"]
    layer = np.zeros((ph, pw, 4), dtype=np.uint8)
    i0, i1, i2 = entry["aff_idx"]
    s = src_local[[i0, i1, i2]].astype(np.float32)
    d = (cpos[[i0, i1, i2]] - np.array([px0, py0], np.float32)).astype(np.float32)
    Maff = cv2.getAffineTransform(s, d)
    warped = cv2.warpAffine(
        crop, Maff, (pw, ph),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    mask = np.zeros((ph, pw), np.uint8)
    hull = cv2.convexHull(np.round(cpos - np.array([px0, py0])).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    layer[mask > 0] = warped[mask > 0]
    return layer


def _rasterize_skinned(entry, cpos, px0, py0, pw, ph):
    crop = entry["atlas_crop"]
    src_local = entry["src_local"]
    faces = entry["faces"]
    origin = np.array([px0, py0], np.float32)
    layer = np.zeros((ph, pw, 4), dtype=np.uint8)
    for f in faces:
        d = cpos[f].astype(np.float32)
        s = src_local[f].astype(np.float32)
        x0 = int(np.floor(d[:, 0].min()))
        x1 = int(np.ceil(d[:, 0].max()))
        y0 = int(np.floor(d[:, 1].min()))
        y1 = int(np.ceil(d[:, 1].max()))
        x0c, y0c = max(x0, px0), max(y0, py0)
        x1c, y1c = min(x1, px0 + pw), min(y1, py0 + ph)
        if x1c <= x0c or y1c <= y0c:
            continue
        dw, dh = x1c - x0c, y1c - y0c
        d_local = (d - origin).astype(np.float32)
        d_patch = d_local - np.array([x0c - px0, y0c - py0], np.float32)

        sx0 = max(0, int(np.floor(s[:, 0].min())))
        sy0 = max(0, int(np.floor(s[:, 1].min())))
        sx1 = min(crop.shape[1], int(np.ceil(s[:, 0].max())) + 1)
        sy1 = min(crop.shape[0], int(np.ceil(s[:, 1].max())) + 1)
        tri_crop = crop[sy0:sy1, sx0:sx1]
        if tri_crop.size == 0:
            continue
        s_rel = s - np.array([sx0, sy0], np.float32)

        Maff = cv2.getAffineTransform(
            np.ascontiguousarray(s_rel[:3], np.float32),
            np.ascontiguousarray(d_patch[:3], np.float32),
        )
        patch = cv2.warpAffine(
            tri_crop, Maff, (dw, dh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        mask = np.zeros((dh, dw), np.uint8)
        cv2.fillConvexPoly(
            mask, np.round(d_local - np.array([x0c - px0, y0c - py0])).astype(np.int32),
            255,
        )
        sel = mask > 0
        roi = layer[y0c - py0:y1c - py0, x0c - px0:x1c - px0]
        roi[sel] = patch[sel]
    return layer


def _render_part(entry, pos, to_canvas, W, H):
    cpos = to_canvas(pos)
    faces = entry["faces"]
    if len(faces) == 0:
        return None
    fv = cpos[faces.ravel()]
    px0 = max(int(np.floor(fv[:, 0].min())), 0)
    py0 = max(int(np.floor(fv[:, 1].min())), 0)
    px1 = min(int(np.ceil(fv[:, 0].max())), W)
    py1 = min(int(np.ceil(fv[:, 1].max())), H)
    if px1 <= px0 or py1 <= py0:
        return None
    pw, ph = px1 - px0, py1 - py0
    if entry["rigid"]:
        layer = _rasterize_rigid(entry, cpos, px0, py0, pw, ph)
    else:
        layer = _rasterize_skinned(entry, cpos, px0, py0, pw, ph)
    return px0, py0, layer


def rasterize(cache, positions, to_canvas, W, H, workers=0, opacities=None):
    canvas = np.zeros((H, W, 4), dtype=np.float32)
    jobs = [
        (entry, pos, (1.0 if opacities is None else opacities[i]))
        for i, (entry, pos) in enumerate(zip(cache, positions))
    ]

    def run(entry, pos, opacity):
        if opacity < VIS_EPS:
            return None
        item = _render_part(entry, pos, to_canvas, W, H)
        if item is None:
            return None
        px0, py0, layer = item
        if opacity < 1.0 - VIS_EPS:
            layer = layer.copy()
            layer[:, :, 3] = np.clip(
                layer[:, :, 3].astype(np.float32) * opacity, 0, 255,
            ).astype(np.uint8)
        return px0, py0, layer

    n_workers = workers
    if n_workers <= 0:
        n_workers = min(os.cpu_count() or 1, 8)

    if n_workers > 1 and len(jobs) > 8:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            layers = pool.map(lambda jp: run(*jp), jobs)
    else:
        layers = (run(entry, pos, opacity) for entry, pos, opacity in jobs)

    for item in layers:
        if item is None:
            continue
        px0, py0, layer = item
        _composite_layer(canvas, layer, px0, py0)

    out = np.zeros((H, W, 4), np.uint8)
    out[:, :, :3] = np.clip(canvas[:, :, :3], 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(canvas[:, :, 3] * 255, 0, 255).astype(np.uint8)
    return out


def export_gif(scene, clip, out_path, raster_cache, target_w=GIF_W, fps=GIF_FPS,
               bg=GIF_BG, workers=GIF_WORKERS):
    parts = scene["parts"]
    rest_TR = scene["TR"]
    hash2tr = scene["hash2tr"]

    sampler, stop, _ = decode_clip(clip)
    n_frames = max(1, int(round(stop * fps)))
    times = [i / fps for i in range(n_frames)]

    gmin = np.array([np.inf, np.inf])
    gmax = np.array([-np.inf, -np.inf])
    for t in times:
        ov = clip_overrides(clip, hash2tr, sampler, t)
        world = make_world(rest_TR, ov)
        positions = skin_all(parts, world)
        allxy = np.vstack(positions)
        gmin = np.minimum(gmin, allxy.min(0))
        gmax = np.maximum(gmax, allxy.max(0))

    w = gmax[0] - gmin[0]
    h = gmax[1] - gmin[1]
    bounds = (
        gmin[0] - w * PAD, gmin[1] - h * PAD,
        gmax[0] + w * PAD, gmax[1] + h * PAD,
    )
    to_canvas, W, H = make_canvas_tf(bounds, target_w)
    print(f"  {out_path.name}: {W}x{H}, {n_frames} frames @ {fps}fps")

    bg_rgb = np.array(
        [int(bg[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = int(round(1000 / fps))
    pal_img = None
    pil_frames = []

    for k, t in enumerate(times):
        ov = clip_overrides(clip, hash2tr, sampler, t)
        go_active, smr_alpha, smr_props = sample_clip_properties(clip, sampler, t)
        opacities = part_opacities(scene, go_active, smr_alpha, smr_props)
        world = make_world(rest_TR, ov)
        positions = skin_all(parts, world)
        out = rasterize(
            raster_cache, positions, to_canvas, W, H,
            workers=workers, opacities=opacities,
        )
        rgb = out[:, :, [2, 1, 0]].astype(np.float32)
        a = out[:, :, 3:4].astype(np.float32) / 255.0
        comp = np.clip(rgb * a + bg_rgb * (1 - a), 0, 255).astype(np.uint8)
        img = Image.fromarray(comp, "RGB")
        if pal_img is None:
            pal_img = img.convert("P", palette=Image.ADAPTIVE, colors=256)
            pframe = pal_img
        else:
            pframe = img.quantize(palette=pal_img, dither=Image.Dither.NONE)
        pil_frames.append(pframe)
        print(f"    frame {k + 1}/{n_frames}", end="\r", flush=True)
    print()

    pil_frames[0].save(
        out_path, save_all=True, append_images=pil_frames[1:],
        duration=duration, loop=0, optimize=True, disposal=1,
    )
    print(f"  wrote {out_path}")


def export_gifs(scene, gif_dir: Path, clip_filter: str | None = None,
                target_w=GIF_W, fps=GIF_FPS, bg=GIF_BG, workers=GIF_WORKERS):
    atlas_bgra = atlases_bgra(scene["atlases"])
    raster_cache = prepare_raster_cache(scene["parts"], atlas_bgra)
    rigid = sum(1 for e in raster_cache if e["rigid"])
    print(f"raster cache: {len(raster_cache)} parts ({rigid} rigid, "
          f"{len(raster_cache) - rigid} skinned)")
    clips = [o.read() for o in scene["objs"] if o.type.name == "AnimationClip"]
    if clip_filter:
        wanted = [s.strip().lower() for s in clip_filter.split(",") if s.strip()]
        clips = [c for c in clips if any(w in c.m_Name.lower() for w in wanted)]
    print(f"exporting {len(clips)} gif(s) -> {gif_dir}")
    for clip in clips:
        out_path = gif_dir / f"anim_{clip.m_Name}.gif"
        try:
            export_gif(scene, clip, out_path, raster_cache, target_w, fps, bg, workers)
        except Exception as e:
            print(f"  !! failed {clip.m_Name}: {e}")


# ---------------------------------------------------------------------------
# AnimationClip decoding
# ---------------------------------------------------------------------------
def curve_size(binding):
    # PPtr curves (sprite / material swaps) live in the clip's separate PPtr
    # curve list, not in the float stream, so they consume no float slots.
    # Counting them as 1 shifted every later binding's stream index by one.
    # [관찰] across the 173-bundle sample: 3 clips carry a PPtr binding, all 3
    # had sum(curve_size) == curveCount + 1; with 0 here every one of the 299
    # clips balances exactly.
    if getattr(binding, "isPPtrCurve", 0):
        return 0
    if binding.typeID == 4:
        if binding.attribute == 2:
            return 4
        return 3
    return 1


def decode_clip(clip):
    mc = clip.m_MuscleClip
    c = mc.m_Clip.data
    sc, dc, cc = c.m_StreamedClip, c.m_DenseClip, c.m_ConstantClip

    n_stream = int(sc.curveCount)
    n_dense = int(dc.m_CurveCount)
    const = np.array(cc.data, dtype=np.float64) if cc.data else np.zeros(0)
    n_total = n_stream + n_dense + len(const)

    per = {}
    if sc.data:
        buf = struct.pack("<%dI" % len(sc.data), *sc.data)
        f = np.frombuffer(buf, dtype="<f4")
        u = np.frombuffer(buf, dtype="<u4")
        ii = np.frombuffer(buf, dtype="<i4")
        pos, n = 0, len(u)
        while pos < n:
            t = float(f[pos])
            nk = int(u[pos + 1])
            pos += 2
            finite = -1e30 < t < 1e30
            for _ in range(nk):
                idx = int(ii[pos])
                coeff = f[pos + 1:pos + 5].astype(np.float64)
                pos += 5
                if finite:
                    per.setdefault(idx, []).append((t, coeff))
    stream_curves = {}
    for idx, keys in per.items():
        keys.sort(key=lambda x: x[0])
        ts = np.array([k[0] for k in keys])
        cf = np.array([k[1] for k in keys])
        stream_curves[idx] = (ts, cf)

    dense = np.array(dc.m_SampleArray, dtype=np.float64) if dc.m_SampleArray else np.zeros(0)
    dense_begin = float(dc.m_BeginTime)
    dense_rate = float(dc.m_SampleRate) or 1.0
    dense_frames = int(dc.m_FrameCount)

    def sampler(t):
        vals = np.zeros(n_total)
        for idx, (ts, cf) in stream_curves.items():
            # Hold the first/last key outside the keyed range instead of
            # evaluating the segment cubic with a negative (or overrun) dt.
            # ⚠️ [추정] that this matches Unity: this sandbox has no network,
            # so no Unity documentation was read.  What *was* observed
            # [관찰]: evaluating outside the range is cubic in dt and diverges
            # without bound -- br_echidna_ns1 / BR_Echidna_NS1_Tep_Normal has
            # a curve whose first key is at t=3.267s worth -40.0, and the old
            # sampler returned -120,575 at t=0, three orders of magnitude
            # outside that curve's own key range.  Holding the end key is the
            # only reading under which the emitted animation stays inside the
            # authored values.
            tc = t
            if tc < ts[0]:
                tc = ts[0]
            elif tc > ts[-1]:
                tc = ts[-1]
            j = bisect_right(ts, tc) - 1
            if j < 0:
                j = 0
            co = cf[j]
            dt = tc - ts[j]
            vals[idx] = ((co[0] * dt + co[1]) * dt + co[2]) * dt + co[3]
        if n_dense > 0 and dense.size:
            fr = (t - dense_begin) * dense_rate
            f0 = int(np.clip(np.floor(fr), 0, dense_frames - 1))
            f1 = min(f0 + 1, dense_frames - 1)
            a = min(1.0, max(0.0, fr - f0))     # same clamp for the dense clip
            row0 = dense[f0 * n_dense:(f0 + 1) * n_dense]
            row1 = dense[f1 * n_dense:(f1 + 1) * n_dense]
            vals[n_stream:n_stream + n_dense] = row0 * (1 - a) + row1 * a
        if len(const):
            vals[n_stream + n_dense:] = const
        return vals

    return sampler, float(mc.m_StopTime), n_total


def euler_to_quat(e) -> np.ndarray:
    """Unity ``m_LocalEulerAngles`` (degrees) -> quaternion ``(x, y, z, w)``.

    Unity applies euler rotations in ZXY order, i.e. ``q = qy * qx * qz``,
    which is what :func:`Quaternion.Euler` produces.  The result feeds the same
    :func:`quat_to_m3` / :func:`decompose2d` path as the quaternion curves
    (``attribute == 2``), so both rotation sources share one convention.
    """
    hx = math.radians(float(e[0])) * 0.5
    hy = math.radians(float(e[1])) * 0.5
    hz = math.radians(float(e[2])) * 0.5
    sx, cx = math.sin(hx), math.cos(hx)
    sy, cy = math.sin(hy), math.cos(hy)
    sz, cz = math.sin(hz), math.cos(hz)
    return np.array([
        sx * cy * cz + cx * sy * sz,
        cx * sy * cz - sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
        cx * cy * cz + sx * sy * sz,
    ], dtype=np.float64)


def clip_overrides(clip, hash2tr, sampler, t):
    vals = sampler(t)
    gb = clip.m_ClipBindingConstant.genericBindings
    out = {}
    idx = 0
    for b in gb:
        size = curve_size(b)
        if b.typeID == 4:
            tr = hash2tr.get(b.path)
            if tr is not None:
                o = out.setdefault(tr, {})
                if b.attribute == 1:
                    o["pos"] = vals[idx:idx + 3]
                elif b.attribute == 2:
                    q = vals[idx:idx + 4]
                    nrm = np.linalg.norm(q)
                    o["rot"] = (q / nrm) if nrm > 1e-9 else q
                elif b.attribute == 3:
                    o["scale"] = vals[idx:idx + 3]
                elif b.attribute == TRANSFORM_EULER_ATTR:
                    o["euler"] = vals[idx:idx + 3]
        idx += size
    # Euler curves only apply where the clip has no quaternion curve for the
    # same transform; a quaternion binding always wins, whatever the order the
    # bindings appear in.
    for o in out.values():
        e = o.pop("euler", None)
        if e is not None and "rot" not in o:
            o["rot"] = euler_to_quat(e)
    return out


def path_hash(path: str) -> int:
    return zlib.crc32(path.encode("utf-8")) & 0xFFFFFFFF


def sample_clip_properties(clip, sampler, t) -> tuple[dict[int, float], dict[int, float], dict[int, dict[int, float]]]:
    """Sample GameObject active, SMR/SpriteRenderer alpha/enabled, and SMR blend weights at time *t*."""
    vals = sampler(t)
    go_active: dict[int, float] = {}
    smr_alpha: dict[int, float] = {}
    smr_props: dict[int, dict[int, float]] = {}
    idx = 0
    for b in clip.m_ClipBindingConstant.genericBindings:
        size = curve_size(b)
        val = float(vals[idx])
        if b.typeID == GO_TYPE and b.attribute == GO_ACTIVE_ATTR:
            go_active[b.path] = val
        elif b.typeID == SMR_TYPE:
            if b.attribute == SMR_COLOR_A_ATTR:
                smr_alpha[b.path] = val
            elif b.attribute == COMPONENT_ENABLED_ATTR:
                if val < 0.5:
                    smr_alpha[b.path] = 0.0
            else:
                smr_props.setdefault(b.path, {})[b.attribute] = val
        elif b.typeID == SPRITE_RENDERER_TYPE:
            if b.attribute == SPRITE_COLOR_A_ATTR:
                smr_alpha[b.path] = val
            elif b.attribute == COMPONENT_ENABLED_ATTR:
                if val < 0.5:
                    smr_alpha[b.path] = 0.0
            else:
                smr_props.setdefault(b.path, {})[b.attribute] = val
        idx += size
    return go_active, smr_alpha, smr_props


def transform_active(tr, go_active, tr_to_path, TR, tr_initial_active=None) -> bool:
    """False when this transform or an animated ancestor is explicitly disabled."""
    while tr:
        path = tr_to_path.get(tr)
        if path is not None:
            active = go_active.get(path_hash(path))
            if active is not None:
                if active < 0.5:
                    return False
            elif tr_initial_active is not None:
                if not tr_initial_active.get(tr, True):
                    return False
        elif tr_initial_active is not None:
            if not tr_initial_active.get(tr, True):
                return False
        tr = TR.get(tr, {}).get("father", 0)
    return True


def _material_additional_alpha(smr_props: dict[int, dict[int, float]], path_h: int) -> float | None:
    """Optional material _AdditionalAlpha curve keyed by renderer path hash."""
    val = smr_props.get(path_h, {}).get(MATERIAL_ADDITIONAL_ALPHA_ATTR)
    if val is None:
        return None
    return max(0.0, min(1.0, val))


def part_rgb(scene, smr_props) -> list[tuple[float, float, float]]:
    """Per-part tint from renderer ``m_Color.r/g/b`` curves (white when absent).

    Channels arrive in ``smr_props`` already, keyed by renderer path hash --
    ``sample_clip_properties`` files every non-alpha renderer attribute there.
    """
    tr_to_path = scene["tr_to_path"]
    out: list[tuple[float, float, float]] = []
    for part in scene["parts"]:
        props = None
        go_tr = part.get("go_tr_pid") or part.get("tr_pid")
        if go_tr:
            path = tr_to_path.get(go_tr)
            if path is not None:
                props = smr_props.get(path_hash(path))
        rgb = [1.0, 1.0, 1.0]
        if props:
            for attr, ch in COLOR_RGB_ATTRS.items():
                v = props.get(attr)
                if v is not None:
                    rgb[ch] = max(0.0, min(1.0, float(v)))
        out.append((rgb[0], rgb[1], rgb[2]))
    return out


def rgba_hex(rgb, alpha) -> str:
    r, g, b = rgb
    return "".join(
        f"{int(round(max(0.0, min(1.0, v)) * 255)):02x}" for v in (r, g, b, alpha)
    )


def part_opacities(scene, go_active, smr_alpha, smr_props, default_opacities=None) -> list[float]:
    """Per-part visibility/alpha from GameObject active + SMR/SpriteRenderer curves."""
    TR = scene["TR"]
    tr_to_path = scene["tr_to_path"]
    tr_initial_active = scene.get("tr_initial_active")
    part_names = part_slot_names(scene["parts"])
    out: list[float] = []
    for part, slot_name in zip(scene["parts"], part_names):
        go_tr = part.get("go_tr_pid") or part.get("tr_pid")
        path_h = None
        if go_tr:
            path = tr_to_path.get(go_tr)
            if path is not None:
                path_h = path_hash(path)

        if go_tr:
            has_explicit_active = False
            explicit_active = True
            curr = go_tr
            while curr:
                path = tr_to_path.get(curr)
                if path is not None:
                    h = path_hash(path)
                    if h in go_active:
                        has_explicit_active = True
                        if go_active[h] < 0.5:
                            explicit_active = False
                            break
                curr = TR.get(curr, {}).get("father", 0)

            if has_explicit_active:
                if not explicit_active:
                    out.append(0.0)
                    continue
                opacity = 1.0
            else:
                if not transform_active(go_tr, go_active, tr_to_path, TR, tr_initial_active):
                    out.append(0.0)
                    continue
                opacity = 1.0 if default_opacities is None else default_opacities.get(slot_name, default_opacities.get(part["name"], 1.0))
        else:
            opacity = 1.0 if default_opacities is None else default_opacities.get(slot_name, default_opacities.get(part["name"], 1.0))

        if path_h is not None:
            alpha = smr_alpha.get(path_h)
            if alpha is not None:
                opacity *= max(0.0, min(1.0, alpha))
            add_alpha = _material_additional_alpha(smr_props, path_h)
            if add_alpha is not None:
                opacity *= add_alpha
        alpha = smr_alpha.get(part.get("name_hash", 0))
        if alpha is not None:
            opacity *= max(0.0, min(1.0, alpha))
        out.append(opacity)
    return out


# ---------------------------------------------------------------------------
# Unity bundle -> scene dict
# ---------------------------------------------------------------------------
class ConversionError(RuntimeError):
    """A failure the caller can act on; ``detail`` is machine-readable."""

    def __init__(self, kind: str, message: str, **detail):
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.message = message
        self.detail = detail

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "message": self.message, **self.detail}


def bone_name_hash_maps(TR, go2tr, go_name) -> dict[int, dict[int, int]]:
    """``{root transform: {crc32(path relative to it): transform}}``.

    ``Mesh.m_BoneNameHashes`` is the CRC32 of each bone's transform path, but
    the rig root those paths are relative to is not stored in the mesh.  Build
    the map for every candidate root; :func:`resolve_bone_slots` picks the one
    that reproduces the bone slots the renderer does declare, and only then
    trusts it for the slots it does not.
    """
    tr2go = {t: g for g, t in go2tr.items()}

    def nm(tr):
        return go_name.get(tr2go.get(tr, 0), "")

    chains = {}
    for tr in TR:
        c, q = [], tr
        while q and q in TR:
            c.append(q)
            q = TR[q]["father"]
        chains[tr] = c                      # [self, parent, ..., scene root]

    roots = {0}
    for c in chains.values():
        roots.update(c)

    maps: dict[int, dict[int, int]] = {}
    for R in roots:
        m: dict[int, int] = {}
        for tr, c in chains.items():
            if R:
                if R == tr or R not in c:
                    continue
                c = c[:c.index(R)]
            m[path_hash("/".join(nm(x) for x in reversed(c)))] = tr
        maps[R] = m
    return maps


def resolve_bone_slots(mesh_name, declared, bind_count, max_index, slot_weight,
                       name_hashes, maps, TR) -> tuple[list[int], list[dict]]:
    """Fill in bone slots the SkinnedMeshRenderer does not resolve.

    Returns ``(bones, notes)``.  Raises :class:`ConversionError` when a slot
    that vertices are actually weighted to cannot be identified from the data
    -- guessing a bone there would silently move geometry.
    """
    need = max(len(declared), int(bind_count), int(max_index) + 1)
    bones = list(declared) + [0] * max(0, need - len(declared))
    missing = [i for i, b in enumerate(bones) if not b or b not in TR]
    if not missing:
        return bones, []

    # Pick the root whose hash map agrees with every slot the renderer *did*
    # declare, and resolves the most of the rest.
    best_map, best_hits = None, -1
    if name_hashes:
        for R, m in maps.items():
            ok = True
            for i, b in enumerate(bones):
                if b and b in TR and i < len(name_hashes):
                    if m.get(int(name_hashes[i])) not in (None, b):
                        ok = False
                        break
            if not ok:
                continue
            hits = sum(1 for i in missing
                       if i < len(name_hashes) and m.get(int(name_hashes[i])))
            if hits > best_hits:
                best_map, best_hits = m, hits

    notes = []
    for i in missing:
        tr = None
        if best_map is not None and i < len(name_hashes):
            tr = best_map.get(int(name_hashes[i]))
        if tr is not None:
            bones[i] = tr
            notes.append({"mesh": mesh_name, "slot": i, "fix": "bone-name-hash"})
            continue
        w = slot_weight(i)
        if w <= 0.0:
            # No vertex is weighted to this slot, so which bone it points at
            # cannot move a single vertex.  Park it on the first valid bone.
            fallback = next((b for b in bones if b and b in TR), None)
            if fallback is None:
                raise ConversionError(
                    "unresolved-bones",
                    f"mesh {mesh_name!r} has no resolvable bone at all",
                    mesh=mesh_name, slots=missing)
            bones[i] = fallback
            notes.append({"mesh": mesh_name, "slot": i, "fix": "unweighted-slot"})
            continue
        raise ConversionError(
            "unresolved-weighted-bone",
            f"mesh {mesh_name!r} slot {i} carries weight {w:.4f} but the "
            f"renderer's bone reference is null/short and its bone-name hash "
            f"matches no transform in this bundle",
            mesh=mesh_name, slot=i, weight=float(w),
            declared_bones=len(declared), bind_poses=int(bind_count),
            max_bone_index=int(max_index),
            name_hash=int(name_hashes[i]) if i < len(name_hashes) else None)
    return bones, notes


def load_atlases(objs) -> tuple[dict[str, dict[str, Any]], dict[int, str]]:
    """Return atlases keyed by partN/name and Texture2D path_id -> atlas key."""
    atlases: dict[str, dict[str, Any]] = {}
    tex2atlas: dict[int, str] = {}
    tex_objs = [o for o in objs if o.type.name == "Texture2D"]

    # Filter out editor rigging gizmos / control handles
    valid_tex = []
    for o in tex_objs:
        d = o.read()
        if d.m_Name in GIZMO_SPRITE_NAMES:
            continue
        valid_tex.append((o, d))

    for idx, (o, data) in enumerate(valid_tex):
        cand_key = atlas_key_from_name(data.m_Name)
        if cand_key is None:
            if len(valid_tex) == 1:
                cand_key = "parts"
            else:
                cand_key = data.m_Name or f"tex_{idx+1}"

        key = cand_key
        if key in atlases:
            key = f"{cand_key}_{idx+1}"

        img = data.image
        w, h = img.size
        atlases[key] = {
            "key": key,
            "name": data.m_Name or key,
            "image": img,
            "width": w,
            "height": h,
        }
        tex2atlas[o.path_id] = key
    return atlases, tex2atlas


def load_scene(src: Path) -> dict[str, Any]:
    env = UnityPy.load(str(src))
    objs = list(env.objects)
    byid = {o.path_id: o for o in objs}

    TR = {}
    go2tr = {}
    go_name = {}
    go_initial_active = {}
    tr_children = {}
    for o in objs:
        if o.type.name == "Transform":
            d = o.read()
            p, r, s = d.m_LocalPosition, d.m_LocalRotation, d.m_LocalScale
            TR[o.path_id] = dict(
                pos=(p.x, p.y, p.z),
                rot=(r.x, r.y, r.z, r.w),
                scale=(s.x, s.y, s.z),
                father=getattr(d.m_Father, "path_id", 0),
            )
            go2tr[getattr(d.m_GameObject, "path_id", 0)] = o.path_id
            tr_children[o.path_id] = [getattr(c, "path_id", 0) for c in d.m_Children]
        elif o.type.name == "GameObject":
            d = o.read()
            go_name[o.path_id] = d.m_Name
            go_initial_active[o.path_id] = getattr(d, "m_IsActive", True)

    atlases, tex2atlas = load_atlases(objs)
    if not atlases:
        raise RuntimeError("no atlas textures found in bundle")

    def atlas_for_material(mat_pid):
        mo = byid.get(mat_pid)
        if not mo:
            return next(iter(atlases.keys())) if atlases else None
        mat = mo.read()

        # 1. Direct _MainTex inspection from Material properties
        if hasattr(mat, "m_SavedProperties"):
            tex_envs = getattr(mat.m_SavedProperties, "m_TexEnvs", [])
            for item in tex_envs:
                if isinstance(item, (tuple, list)) and item[0] == "_MainTex":
                    tex_ptr = getattr(item[1], "m_Texture", None)
                    tex_pid = getattr(tex_ptr, "path_id", 0)
                    if tex_pid in tex2atlas:
                        return tex2atlas[tex_pid]

        # 2. Material name search
        k = atlas_key_from_name(mat.m_Name)
        if k and k in atlases:
            return k
        if mat.m_Name in atlases:
            return mat.m_Name

        # 3. Fallback
        return next(iter(atlases.keys())) if atlases else None

    parts = []
    repairs: list[dict] = []
    name_hash_maps = None
    for o in objs:
        if o.type.name != "SkinnedMeshRenderer":
            continue
        smr = o.read()
        mp = getattr(smr.m_Mesh, "path_id", 0)
        if mp not in byid:
            continue
        mesh = byid[mp].read()
        mats = smr.m_Materials or []
        atlas = atlas_for_material(getattr(mats[0], "path_id", 0)) if mats else None
        if atlas is None:
            continue

        bones = [getattr(b, "path_id", 0) for b in smr.m_Bones]
        bind = [mat4(m) for m in mesh.m_BindPose]

        h = MeshHandler(mesh)
        h.process()
        vc = h.m_VertexCount
        v = np.array(h.m_Vertices, dtype=np.float64).reshape(vc, 3)
        uv = np.array(h.m_UV0, dtype=np.float64).reshape(vc, 2)
        vh = np.c_[v, np.ones(vc)]

        single = len(bones) <= 1
        bi = bw = None
        if not single:
            bi_raw = np.array(h.m_BoneIndices)
            # Guard: UnityPy may return a single object instead of an empty array
            # for meshes with no bone data (e.g. parts_Head with bones=0).
            if bi_raw.dtype == object or bi_raw.size < vc:
                single = True
            else:
                k = max(1, bi_raw.size // vc)
                bi = bi_raw.reshape(vc, k)
                if h.m_BoneWeights is not None and np.array(h.m_BoneWeights).size == vc * k:
                    bw = np.array(h.m_BoneWeights, dtype=np.float64).reshape(vc, k)
                else:
                    bw = np.zeros((vc, k))
                    bw[:, 0] = 1.0

        # Some renderers ship a bone list that is null in places, or shorter
        # than the mesh's bind poses / the bone indices its vertices use.
        broken = any((not b) or (b not in TR) for b in bones)
        overrun = bi is not None and int(bi.max()) >= len(bones)
        if bones and (broken or overrun):
            def slot_weight(i, bi=bi, bw=bw):
                if bi is None:
                    return 0.0
                sel = bi == i
                return float(bw[sel].sum()) if bw is not None else float(sel.sum())

            if name_hash_maps is None:
                name_hash_maps = bone_name_hash_maps(TR, go2tr, go_name)
            bones, notes = resolve_bone_slots(
                mesh.m_Name, bones, len(bind),
                int(bi.max()) if bi is not None else len(bones) - 1,
                slot_weight, list(getattr(mesh, "m_BoneNameHashes", []) or []),
                name_hash_maps, TR,
            )
            for i in range(len(bind), len(bones)):
                if slot_weight(i) > 0.0:
                    raise ConversionError(
                        "missing-bind-pose",
                        f"mesh {mesh.m_Name!r} slot {i} is weighted but the mesh "
                        f"has only {len(bind)} bind poses",
                        mesh=mesh.m_Name, slot=i, bind_poses=len(bind))
                bind.append(np.eye(4))       # unweighted slot: never applied
            repairs.extend(notes)
            single = len(bones) <= 1

        # Renderers also ship bone lists LONGER than the mesh's bind poses.
        # Those surplus slots have no bind matrix, so they can only be dropped
        # -- which is safe exactly while no vertex is weighted to them.
        if len(bones) > len(bind):
            for i in range(len(bind), len(bones)):
                w = 0.0
                if bi is not None:
                    sel = bi == i
                    w = float(bw[sel].sum()) if bw is not None else float(sel.sum())
                if w > 0.0:
                    raise ConversionError(
                        "missing-bind-pose",
                        f"mesh {mesh.m_Name!r} slot {i} is weighted ({w:.4f}) but "
                        f"the mesh has only {len(bind)} bind poses",
                        mesh=mesh.m_Name, slot=i, weight=w,
                        declared_bones=len(bones), bind_poses=len(bind))
            repairs.append({"mesh": mesh.m_Name, "fix": "drop-unweighted-slots",
                            "dropped": len(bones) - len(bind)})
            bones = bones[:len(bind)]
            single = len(bones) <= 1

        submesh_triangles = h.get_triangles()
        go_tr_pid = go2tr.get(getattr(smr.m_GameObject, "path_id", 0))
        order = getattr(smr, "m_SortingOrder", 0)

        if len(submesh_triangles) <= 1:
            faces = []
            if submesh_triangles:
                faces = np.array(submesh_triangles[0]).reshape(-1, 3)
            else:
                faces = np.zeros((0, 3), int)

            aw = atlases[atlas]["width"]
            ah = atlases[atlas]["height"]
            src_px = np.c_[uv[:, 0] * aw, (1 - uv[:, 1]) * ah]
            parts.append(dict(
                name=mesh.m_Name, atlas=atlas, kind="mesh",
                bones=bones, bind=bind, vh=vh, single=single,
                bi=bi, bw=bw, src_px=src_px, faces=faces,
                order=order,
                go_tr_pid=go_tr_pid,
                name_hash=path_hash(mesh.m_Name),
            ))
        else:
            for sm_idx, sm_tris in enumerate(submesh_triangles):
                sm_mat = mats[sm_idx] if sm_idx < len(mats) else mats[0]
                sm_atlas = atlas_for_material(getattr(sm_mat, "path_id", 0)) if sm_mat else atlas
                if sm_atlas is None:
                    continue
                sm_tris_arr = np.array(sm_tris, dtype=int).reshape(-1, 3)
                if len(sm_tris_arr) == 0:
                    continue
                v_set = sorted(list({idx for tri in sm_tris_arr for idx in tri}))
                if not v_set:
                    continue
                remap = {old: new for new, old in enumerate(v_set)}
                sub_faces = np.array([[remap[idx] for idx in tri] for tri in sm_tris_arr], dtype=int)
                sub_vh = vh[v_set]
                sub_uv = uv[v_set]
                sub_bi = bi[v_set] if bi is not None else None
                sub_bw = bw[v_set] if bw is not None else None
                aw = atlases[sm_atlas]["width"]
                ah = atlases[sm_atlas]["height"]
                sub_src_px = np.c_[sub_uv[:, 0] * aw, (1 - sub_uv[:, 1]) * ah]
                sub_name = mesh.m_Name if sm_idx == 0 else f"{mesh.m_Name}#sub{sm_idx}"
                parts.append(dict(
                    name=sub_name, atlas=sm_atlas, kind="mesh",
                    bones=bones, bind=bind, vh=sub_vh, single=single,
                    bi=sub_bi, bw=sub_bw, src_px=sub_src_px, faces=sub_faces,
                    order=order,
                    go_tr_pid=go_tr_pid,
                    name_hash=path_hash(sub_name),
                ))

    for o in objs:
        if o.type.name != "MeshRenderer":
            continue
        mr = o.read()
        go_ptr = getattr(mr, "m_GameObject", None)
        if not go_ptr or getattr(go_ptr, "path_id", 0) not in byid:
            continue
        go = byid[go_ptr.path_id].read()

        mf_ptr = None
        for c in getattr(go, "m_Components", []):
            if getattr(c, "type", None) and c.type.name == "MeshFilter":
                mf_ptr = c
                break
        if not mf_ptr or getattr(mf_ptr, "path_id", 0) not in byid:
            continue
        mf = byid[mf_ptr.path_id].read()
        mp = getattr(mf.m_Mesh, "path_id", 0)
        if mp not in byid:
            continue
        mesh = byid[mp].read()
        mats = mr.m_Materials or []
        atlas = atlas_for_material(getattr(mats[0], "path_id", 0)) if mats else None
        if atlas is None:
            continue

        go_tr_pid = go2tr.get(go_ptr.path_id)
        if not go_tr_pid:
            continue

        # For facial MeshRenderer parts under eye/mouth/eyebrow controllers,
        # bind directly to Bone_Head so controller non-uniform scale does not distort facial features.
        target_tr = go_tr_pid
        curr = go_tr_pid
        tr2go = {t: g for g, t in go2tr.items()}
        while curr:
            curr_name = go_name.get(tr2go.get(curr, 0), "")
            if "head" in curr_name.lower():
                target_tr = curr
                break
            curr = TR.get(curr, {}).get("father")

        bones = [target_tr]
        bind = [np.eye(4)]

        h = MeshHandler(mesh)
        h.process()
        vc = h.m_VertexCount
        v = np.array(h.m_Vertices, dtype=np.float64).reshape(vc, 3)
        uv = np.array(h.m_UV0, dtype=np.float64).reshape(vc, 2)
        vh = np.c_[v, np.ones(vc)]

        submesh_triangles = h.get_triangles()
        order = getattr(mr, "m_SortingOrder", 0)

        if len(submesh_triangles) <= 1:
            faces = []
            if submesh_triangles:
                faces = np.array(submesh_triangles[0]).reshape(-1, 3)
            else:
                faces = np.zeros((0, 3), int)

            aw = atlases[atlas]["width"]
            ah = atlases[atlas]["height"]
            src_px = np.c_[uv[:, 0] * aw, (1 - uv[:, 1]) * ah]

            parts.append(dict(
                name=mesh.m_Name, atlas=atlas, kind="mesh", is_mesh_renderer=True,
                bones=bones, bind=bind, vh=vh, single=True,
                bi=None, bw=None, src_px=src_px, faces=faces,
                order=order,
                go_tr_pid=go_tr_pid,
                name_hash=path_hash(mesh.m_Name),
            ))
        else:
            mats = mr.m_Materials or []
            for sm_idx, sm_tris in enumerate(submesh_triangles):
                sm_mat = mats[sm_idx] if sm_idx < len(mats) else mats[0]
                sm_atlas = atlas_for_material(getattr(sm_mat, "path_id", 0)) if sm_mat else atlas
                if sm_atlas is None:
                    continue
                sm_tris_arr = np.array(sm_tris, dtype=int).reshape(-1, 3)
                if len(sm_tris_arr) == 0:
                    continue
                v_set = sorted(list({idx for tri in sm_tris_arr for idx in tri}))
                if not v_set:
                    continue
                remap = {old: new for new, old in enumerate(v_set)}
                sub_faces = np.array([[remap[idx] for idx in tri] for tri in sm_tris_arr], dtype=int)
                sub_vh = vh[v_set]
                sub_uv = uv[v_set]
                aw = atlases[sm_atlas]["width"]
                ah = atlases[sm_atlas]["height"]
                sub_src_px = np.c_[sub_uv[:, 0] * aw, (1 - sub_uv[:, 1]) * ah]
                sub_name = mesh.m_Name if sm_idx == 0 else f"{mesh.m_Name}#sub{sm_idx}"
                parts.append(dict(
                    name=sub_name, atlas=sm_atlas, kind="mesh", is_mesh_renderer=True,
                    bones=bones, bind=bind, vh=sub_vh, single=True,
                    bi=None, bw=None, src_px=sub_src_px, faces=sub_faces,
                    order=order,
                    go_tr_pid=go_tr_pid,
                    name_hash=path_hash(sub_name),
                ))

    for o in objs:
        if o.type.name != "SpriteRenderer":
            continue
        sr = o.read()
        sp_ptr = getattr(sr, "m_Sprite", None)
        if not sp_ptr or getattr(sp_ptr, "path_id", 0) not in byid:
            continue
        sp_o = byid[sp_ptr.path_id]
        sp = sp_o.read()
        rd = sp.m_RD
        tex_pid = getattr(getattr(rd, "texture", None), "path_id", 0)
        atlas = tex2atlas.get(tex_pid)
        if atlas is None:
            continue
        tr_pid = go2tr.get(getattr(sr.m_GameObject, "path_id", 0))
        if tr_pid is None:
            continue

        # Filter out editor rigging gizmos / control handles / transparent touch blanks
        sp_name = sp.m_Name
        go_name_str = go_name.get(getattr(sr.m_GameObject, "path_id", 0), "")
        is_sr_enabled = getattr(sr, "m_Enabled", True)
        clean_go_name = go_name_str.lower().replace("_", "").replace(" ", "")
        c = getattr(sr, "m_Color", None)
        alpha = float(c.a) if c is not None and hasattr(c, "a") else 1.0
        if (
            sp_name in GIZMO_SPRITE_NAMES
            or "hiddenBone" in go_name_str
            or "CTRL" in go_name_str
            or "cameraboundary" in clean_go_name
            or "touchblank" in clean_go_name
            or alpha <= 0.01
            or (not is_sr_enabled and ("bone_" in go_name_str or "spline_" in go_name_str))
        ):
            continue

        rect = sp.m_Rect
        ptu = sp.m_PixelsToUnits or 100.0
        piv = sp.m_Pivot

        h = MeshHandler(rd, sp_o.version)
        h.process()
        vc = h.m_VertexCount
        sv = np.array(h.m_Vertices, dtype=np.float64).reshape(vc, 3)

        ah = atlases[atlas]["height"]
        ax = rect.x + (sv[:, 0] * ptu + piv.x * rect.width)
        ay = rect.y + (sv[:, 1] * ptu + piv.y * rect.height)
        src = np.c_[ax, ah - ay]

        flip_x = getattr(sr, "m_FlipX", False)
        flip_y = getattr(sr, "m_FlipY", False)
        if flip_x:
            sv[:, 0] = -sv[:, 0]
        if flip_y:
            sv[:, 1] = -sv[:, 1]
        sv2 = np.c_[sv[:, :2], np.zeros(vc), np.ones(vc)]

        faces = []
        for sm in h.get_triangles():
            faces.append(np.array(sm).reshape(-1, 3))
        faces = np.vstack(faces) if faces else np.zeros((0, 3), int)
        if (flip_x ^ flip_y) and len(faces) > 0:
            faces = faces[:, [0, 2, 1]]

        blend = None
        for m_ptr in getattr(sr, "m_Materials", []) or []:
            if m_ptr and getattr(m_ptr, "path_id", 0) in byid:
                m_data = byid[m_ptr.path_id].read()
                b = material_blend_mode(getattr(m_data, "m_Name", ""))
                if b:
                    blend = b
                    break

        initial_color = (1.0, 1.0, 1.0, 1.0)
        c = getattr(sr, "m_Color", None)
        if c is not None:
            initial_color = (float(c.r), float(c.g), float(c.b), float(c.a))

        parts.append(dict(
            name=sp.m_Name, atlas=atlas, kind="sprite",
            tr_pid=tr_pid, sv2=sv2, src_px=src, faces=faces,
            order=getattr(sr, "m_SortingOrder", 0),
            go_tr_pid=tr_pid,
            name_hash=path_hash(sp.m_Name),
            blend=blend,
            initial_color=initial_color,
        ))

    # If a CloseUp hierarchy exists, prefer its full composite sprite set over
    # duplicate scaled copies outside CloseUp (e.g. 2dmodel_br_echidna_ns2).
    closeup_tr = None
    tr2go = {t: g for g, t in go2tr.items()}
    for tr, g in tr2go.items():
        if "closeup" in go_name.get(g, "").lower():
            closeup_tr = tr
            break
    if closeup_tr:
        closeup_sprites = set()
        for p in parts:
            cur = p.get("tr_pid") or p.get("go_tr_pid")
            while cur:
                if cur == closeup_tr:
                    closeup_sprites.add(p["name"])
                    break
                cur = TR.get(cur, {}).get("father")
        if closeup_sprites:
            retained_parts = []
            for p in parts:
                cur = p.get("tr_pid") or p.get("go_tr_pid")
                is_under_closeup = False
                while cur:
                    if cur == closeup_tr:
                        is_under_closeup = True
                        break
                    cur = TR.get(cur, {}).get("father")
                if not is_under_closeup and p["name"] in closeup_sprites:
                    continue
                retained_parts.append(p)
            parts = retained_parts

    world_fn_sort = make_world(TR)
    for p in parts:
        tr = p.get("tr_pid") or p.get("go_tr_pid")
        p["world_z"] = float(world_fn_sort(tr)[2, 3]) if tr and tr in TR else 0.0
    parts.sort(key=lambda p: (p.get("order", 0), -round(p.get("world_z", 0.0), 3)))

    animator_candidates = []
    for o in objs:
        if o.type.name == "Animator":
            an = o.read()
            go_ptr = getattr(an, "m_GameObject", None)
            g_pid = getattr(go_ptr, "path_id", 0) if go_ptr else 0
            if g_pid in go2tr:
                animator_candidates.append(g_pid)

    animator_go = None
    if len(animator_candidates) == 1:
        animator_go = animator_candidates[0]
    elif len(animator_candidates) > 1:
        tr2go = {t: g for g, t in go2tr.items()}
        clip_hashes = set()
        for o in objs:
            if o.type.name == "AnimationClip":
                c = o.read()
                if hasattr(c, "m_ClipBindingConstant"):
                    for b in c.m_ClipBindingConstant.genericBindings:
                        if b.typeID == 4:
                            clip_hashes.add(b.path)

        best_cand = None
        best_score = (-1, -1, -1, -999)
        for g_pid in animator_candidates:
            cand_root_tr = go2tr[g_pid]
            sub_trs = set()
            sub_hashes = set()

            def _walk_eval(tr, prefix):
                sub_trs.add(tr)
                for c in tr_children.get(tr, []):
                    nm = go_name.get(tr2go.get(c, 0), "")
                    p = nm if prefix == "" else prefix + "/" + nm
                    sub_hashes.add(path_hash(p))
                    _walk_eval(c, p)

            _walk_eval(cand_root_tr, "")
            sub_hashes.add(path_hash(""))

            n_mesh = sum(1 for p in parts if p.get("go_tr_pid") in sub_trs and p["kind"] == "mesh")
            n_active_mesh = sum(1 for p in parts if p.get("go_tr_pid") in sub_trs and p["kind"] == "mesh" and go_initial_active.get(tr2go.get(p.get("go_tr_pid"), 0), True))
            n_clip_match = len(clip_hashes.intersection(sub_hashes)) if clip_hashes else 0
            n_total = sum(1 for p in parts if p.get("go_tr_pid") in sub_trs)

            cand_name = go_name.get(g_pid, "").lower()
            is_only_tag = 1 if any(t in cand_name for t in ("_only", "only")) else 0

            depth = 0
            curr = cand_root_tr
            while curr and curr in TR and TR[curr].get("father"):
                curr = TR[curr]["father"]
                depth += 1

            # Prioritize: 1. not a dummy '_only' tag, 2. most active meshes, 3. total meshes, 4. clip match, 5. total parts, 6. shallower depth
            score = (-is_only_tag, n_active_mesh, n_mesh, n_clip_match, n_total, -depth)
            if score > best_score:
                best_score = score
                best_cand = g_pid

        animator_go = best_cand
    hash2tr = {}
    tr_to_path = {}
    if animator_go is not None and animator_go in go2tr:
        root_tr = go2tr[animator_go]
        tr2go = {t: g for g, t in go2tr.items()}

        def name_of(tr):
            return go_name.get(tr2go.get(tr, 0), "")

        def walk(tr, prefix):
            for c in tr_children.get(tr, []):
                nm = name_of(c)
                p = nm if prefix == "" else prefix + "/" + nm
                tr_to_path[c] = p
                hash2tr[path_hash(p)] = c
                walk(c, p)

        walk(root_tr, "")
        tr_to_path[root_tr] = ""
        hash2tr[path_hash("")] = root_tr

    tr_initial_active = {
        go2tr[g]: act for g, act in go_initial_active.items() if g in go2tr
    }

    # Filter and merge parts when Animator is present:
    # 1. Main parts in the animator hierarchy that have visible texture.
    # 2. Valid parts from active sibling hierarchies (e.g. Chair_Only, Table_Only, BG_Only)
    #    that share the same bone structure and have visible textures.
    if tr_to_path:
        atlases_bgra_temp = atlases_bgra(atlases)
        main_tr_by_path = {p: tr for tr, p in tr_to_path.items()}
        main_root_tr = hash2tr.get(path_hash(""))

        merged_parts = []
        merged_names = set()
        for p in parts:
            if p.get("go_tr_pid") in tr_to_path:
                if is_part_visible(p, atlases_bgra_temp):
                    merged_parts.append(p)
                    merged_names.add(p["name"])

        main_parent_tr = TR.get(main_root_tr, {}).get("father") if main_root_tr else None
        sibling_roots = [c for c in tr_children.get(main_parent_tr, []) if c != main_root_tr] if main_parent_tr else []

        for sib_tr in sibling_roots:
            sib_go_name = go_name.get(tr2go.get(sib_tr, 0), "")
            sib_tr_to_path = {}
            def walk_sib(tr, prefix):
                for c in tr_children.get(tr, []):
                    nm = go_name.get(tr2go.get(c, 0), "")
                    p = nm if prefix == "" else prefix + "/" + nm
                    sib_tr_to_path[c] = p
                    walk_sib(c, p)
            walk_sib(sib_tr, "")
            sib_tr_to_path[sib_tr] = ""

            for p in parts:
                if p.get("go_tr_pid") in sib_tr_to_path:
                    if not is_part_visible(p, atlases_bgra_temp):
                        continue
                    new_bones = []
                    valid_remap = True
                    for b in p.get("bones", []):
                        if b in sib_tr_to_path:
                            b_path = sib_tr_to_path[b]
                            if b_path in main_tr_by_path:
                                new_bones.append(main_tr_by_path[b_path])
                            else:
                                valid_remap = False
                                break
                        elif b in tr_to_path:
                            new_bones.append(b)
                        else:
                            valid_remap = False
                            break
                    if not valid_remap:
                        continue
                    p["bones"] = new_bones
                    part_path = sib_tr_to_path.get(p.get("go_tr_pid"), "")
                    if part_path in main_tr_by_path:
                        p["go_tr_pid"] = main_tr_by_path[part_path]

                    if p["name"] in merged_names:
                        m = re.search(r"_(chair|table|bg|weapon|sub|only|prop)", sib_go_name, re.I)
                        suffix = ("_" + m.group(1).lower()) if m else "_sub"
                        if suffix == "_only":
                            suffix = "_prop"
                        p["name"] = f"{p['name']}{suffix}"
                        p["name_hash"] = path_hash(p["name"])

                    merged_parts.append(p)
                    merged_names.add(p["name"])

        if merged_parts:
            merged_parts.sort(key=lambda x: (x.get("order", 0), -round(x.get("world_z", 0.0), 3), x["name"]))
            parts = merged_parts
        else:
            in_animator = [p for p in parts if p.get("go_tr_pid") in tr_to_path]
            if in_animator:
                parts = in_animator

    if not parts:
        kind, why = _why_no_parts(objs)
        raise ConversionError(kind, why, **_bundle_shape(objs))

    (
        tilted_pos_children, tilted_rot_children,
        tilted_pos_children_neg, tilted_pos_children_pos,
        tilted_rot_children_neg, tilted_rot_children_pos,
    ) = normalize_tilted_pitch(TR)
    tr2go = {t: g for g, t in go2tr.items()}
    _normalize_face_parts(TR, parts, go_name, tr2go, repairs)

    return dict(
        objs=objs, TR=TR, atlases=atlases, parts=parts, hash2tr=hash2tr,
        tr_to_path=tr_to_path, go2tr=go2tr, go_name=go_name,
        tr_initial_active=tr_initial_active, repairs=repairs,
        tilted_pos_children=tilted_pos_children,
        tilted_rot_children=tilted_rot_children,
        tilted_pos_children_neg=tilted_pos_children_neg,
        tilted_pos_children_pos=tilted_pos_children_pos,
        tilted_rot_children_neg=tilted_rot_children_neg,
        tilted_rot_children_pos=tilted_rot_children_pos,
        tilted_children=tilted_pos_children | tilted_rot_children,
    )


def _bundle_shape(objs) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for o in objs:
        counts[o.type.name] = counts.get(o.type.name, 0) + 1
    scripts = sorted({
        getattr(o.read(), "m_ClassName", "") for o in objs
        if o.type.name == "MonoScript"
    } - {""})
    texts = [o.read().m_Name for o in objs if o.type.name == "TextAsset"]
    return {"object_counts": counts, "mono_scripts": scripts,
            "text_assets": texts}


SPINE_RUNTIME_SCRIPTS = frozenset({
    "SkeletonDataAsset", "SpineAtlasAsset", "SkeletonMecanim",
    "SkeletonAnimation", "SkeletonGraphic",
})


def _why_no_parts(objs) -> tuple[str, str]:
    shape = _bundle_shape(objs)
    if SPINE_RUNTIME_SCRIPTS & set(shape["mono_scripts"]):
        return ("native-spine-bundle",
                "native spine bundle - not a conversion target")
    if not shape["object_counts"].get("SkinnedMeshRenderer") and \
            not shape["object_counts"].get("SpriteRenderer"):
        return ("no-renderable-parts",
                "no SkinnedMeshRenderer and no SpriteRenderer in the bundle")
    return ("no-renderable-parts",
            "every renderer was dropped: its mesh/sprite is missing from the "
            "bundle, or no material resolved to an atlas page")


# ---------------------------------------------------------------------------
# Spine skeleton
# ---------------------------------------------------------------------------
def build_bones(scene):
    TR = scene["TR"]
    go2tr = scene["go2tr"]
    go_name = scene["go_name"]
    tr2go = {t: g for g, t in go2tr.items()}

    def raw_name(tr):
        return go_name.get(tr2go.get(tr, 0)) or f"bone_{tr}"

    used = {}
    tr_name = {}
    for tr in TR:
        nm = raw_name(tr).replace("/", "_")
        if nm in used:
            used[nm] += 1
            nm = f"{nm}#{used[nm]}"
        else:
            used[nm] = 0
        tr_name[tr] = nm

    children = {tr: [] for tr in TR}
    roots = []
    for tr, t in TR.items():
        f = t["father"]
        if f and f in TR:
            children[f].append(tr)
        else:
            roots.append(tr)

    ordered = []
    parent_name = {}
    stack = list(reversed(roots))
    while stack:
        tr = stack.pop()
        ordered.append(tr)
        for c in reversed(children[tr]):
            parent_name[c] = tr
            stack.append(c)

    bones = [{"name": "root"}]
    setup = {}
    bone_index = {"root": 0}
    for tr in ordered:
        s = trs_to_spine(TR[tr])
        setup[tr] = s
        parent = tr_name[parent_name[tr]] if tr in parent_name else "root"
        b = {"name": tr_name[tr], "parent": parent}
        if abs(s["x"]) > EPS:
            b["x"] = r2(s["x"])
        if abs(s["y"]) > EPS:
            b["y"] = r2(s["y"])
        if abs(s["rotation"]) > EPS:
            b["rotation"] = r2(s["rotation"])
        if abs(s["scaleX"] - 1) > EPS:
            b["scaleX"] = r2(s["scaleX"])
        if abs(s["scaleY"] - 1) > EPS:
            b["scaleY"] = r2(s["scaleY"])
        if abs(s["shearY"]) > EPS:
            b["shearY"] = r2(s["shearY"])
        bones.append(b)
        bone_index[tr_name[tr]] = len(bones) - 1

    return bones, tr_name, bone_index, setup


def part_weighted_vertices(part, tr_name, bone_index):
    out = []
    if part["kind"] == "sprite":
        bidx = bone_index[tr_name[part["tr_pid"]]]
        for v in part["sv2"]:
            out += [1, bidx, r2(v[0]), r2(v[1]), 1.0]
        return out

    bones = part["bones"]
    bind = part["bind"]
    vh = part["vh"]
    bone_idx = [bone_index[tr_name[b]] for b in bones]

    if part["single"]:
        if len(bones) == 0:
            # No bones at all – bind to part parent transform or root (0)
            target_bidx = 0
            target_tr = part.get("go_tr_pid") or part.get("tr_pid")
            if target_tr and target_tr in tr_name:
                target_bidx = bone_index.get(tr_name[target_tr], 0)
            dy = vh[:, 1].max() - vh[:, 1].min()
            dz = vh[:, 2].max() - vh[:, 2].min()
            y_idx = 2 if (dy < 1e-3 and dz > 1e-3) else 1
            for p in vh:
                out += [1, target_bidx, r2(p[0]), r2(p[y_idx]), 1.0]
            return out
        local = (bind[0] @ vh.T).T
        dy = local[:, 1].max() - local[:, 1].min()
        dz = local[:, 2].max() - local[:, 2].min()
        y_idx = 2 if (dy < 1e-3 and dz > 1e-3) else 1
        for p in local:
            out += [1, bone_idx[0], r2(p[0]), r2(p[y_idx]), 1.0]
        return out

    bi, bw = part["bi"], part["bw"]
    vc, k = bi.shape
    locals_by_bone = {j: (bind[j] @ vh.T).T for j in range(len(bones))}
    y_idx_by_bone = {}
    for j in range(len(bones)):
        loc = locals_by_bone[j]
        dy = loc[:, 1].max() - loc[:, 1].min()
        dz = loc[:, 2].max() - loc[:, 2].min()
        y_idx_by_bone[j] = 2 if (dy < 1e-3 and dz > 1e-3) else 1

    for i in range(vc):
        acc = {}
        for ki in range(k):
            w = float(bw[i, ki])
            if w <= 0:
                continue
            j = int(bi[i, ki])
            acc[j] = acc.get(j, 0.0) + w
        if not acc:
            acc[int(bi[i, 0])] = 1.0
        out.append(len(acc))
        for j, w in acc.items():
            p = locals_by_bone[j][i]
            y_idx = y_idx_by_bone[j]
            out += [bone_idx[j], r2(p[0]), r2(p[y_idx]), r2(w)]
    return out


def mesh_image_size(part, atlases):
    """Mesh attachment image size must match the exported atlas page PNG."""
    page = atlases[part["atlas"]]
    return int(page["width"]), int(page["height"])


def _mesh_edge_pairs(faces: np.ndarray) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for a, b, c in np.asarray(faces, dtype=np.int64).reshape(-1, 3):
        for i, j in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def _mesh_hull_order(faces: np.ndarray, vcount: int) -> tuple[int, list[int]]:
    """Return hull length and vertex order with boundary vertices first."""
    edge_use: dict[tuple[int, int], int] = {}
    for a, b, c in np.asarray(faces, dtype=np.int64).reshape(-1, 3):
        for i, j in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (min(i, j), max(i, j))
            edge_use[key] = edge_use.get(key, 0) + 1

    boundary = [edge for edge, count in edge_use.items() if count == 1]
    if not boundary:
        return vcount, list(range(vcount))

    adj: dict[int, list[int]] = {}
    for i, j in boundary:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)

    start = boundary[0][0]
    hull = [start]
    prev = -1
    cur = start
    used: set[tuple[int, int]] = set()
    for _ in range(len(boundary) + 1):
        nxt = None
        for cand in adj.get(cur, []):
            if cand == prev:
                continue
            key = (min(cur, cand), max(cur, cand))
            if key in used:
                continue
            nxt = cand
            used.add(key)
            break
        if nxt is None or nxt == start:
            break
        hull.append(nxt)
        prev, cur = cur, nxt

    hull_set = {i for edge in boundary for i in edge}
    for v in sorted(hull_set - set(hull)):
        hull.append(v)

    internal = sorted(set(range(vcount)) - set(hull))
    return len(hull), hull + internal


def _reorder_part_vertices(part, order: list[int]):
    order_arr = np.asarray(order, dtype=np.int64)
    rp = dict(part)
    rp["src_px"] = part["src_px"][order_arr]
    if part["kind"] == "sprite":
        rp["sv2"] = part["sv2"][order_arr]
    else:
        rp["vh"] = part["vh"][order_arr]
        if not part["single"]:
            rp["bi"] = part["bi"][order_arr]
            rp["bw"] = part["bw"][order_arr]
    old_to_new = {old: new for new, old in enumerate(order)}
    rp["faces"] = np.vectorize(old_to_new.get)(part["faces"])
    return rp


def build_spine_mesh_attachment(part, tr_name, bone_index, aw, ah):
    """Build mesh attachment arrays with Spine hull order and edge pairs."""
    faces = np.asarray(part["faces"], dtype=np.int64)
    edge_pairs = _mesh_edge_pairs(faces)
    hull_len, order = _mesh_hull_order(faces, len(part["src_px"]))
    old_to_new = {old: new for new, old in enumerate(order)}
    rp = _reorder_part_vertices(part, order)

    uvs = []
    for sx, sy in rp["src_px"]:
        uvs += [r2(sx / aw), r2(sy / ah)]
    tris = [int(i) for i in rp["faces"].ravel()]
    verts = part_weighted_vertices(rp, tr_name, bone_index)
    edges = []
    for i, j in edge_pairs:
        edges.extend([old_to_new[i], old_to_new[j]])
    return hull_len, uvs, tris, verts, edges


def part_slot_names(parts) -> list[str]:
    """Spine slot names for scene parts (matches build_slots_and_skin naming)."""
    used_names: dict[str, int] = {}
    names: list[str] = []
    for p in parts:
        base = p["name"] or f"part_{p['order']}"
        name = base
        if name in used_names:
            used_names[name] += 1
            name = f"{name}#{used_names[name]}"
        else:
            used_names[name] = 0
        names.append(name)
    return names


def _opacity_key_stepped(prev: float | None, cur: float) -> bool:
    """Use stepped interpolation for hard on/off visibility switches."""
    if prev is None or abs(cur - prev) <= EPS:
        return False
    if abs(cur - prev) >= 0.5:
        return True
    near_edge = lambda x: x < VIS_EPS or x > 1.0 - VIS_EPS
    return near_edge(prev) and near_edge(cur)


def build_slot_opacity_tracks(scene, clip, sampler, times, default_opacities=None,
                              default_rgb=None) -> dict:
    """Slot colour keyframes from GameObject active / renderer colour curves."""
    default_opacities = default_opacities or {}
    default_rgb = default_rgb or {}
    part_names = part_slot_names(scene["parts"])
    has_onestore = any("onestore" in p["name"].lower() for p in scene["parts"])
    series = {name: [] for name in part_names}
    rgb_series = {name: [] for name in part_names}
    for t in times:
        go_active, smr_alpha, smr_props = sample_clip_properties(clip, sampler, t)
        opacities = part_opacities(scene, go_active, smr_alpha, smr_props, default_opacities=default_opacities)
        rgbs = part_rgb(scene, smr_props)
        for name, op, rgb in zip(part_names, opacities, rgbs):
            if has_onestore and "google" in name.lower():
                series[name].append(0.0)
            else:
                series[name].append(max(0.0, min(1.0, op)))
            rgb_series[name].append(rgb)

    slots_out = {}
    for name, vals in series.items():
        setup_op = default_opacities.get(name, 1.0)
        setup_rgb = default_rgb.get(name, (1.0, 1.0, 1.0))
        rgbs = rgb_series[name]
        # If every frame matches the setup pose within EPS, omit the track.
        if (all(abs(v - setup_op) <= EPS for v in vals)
                and all(max(abs(c - s) for c, s in zip(rgb, setup_rgb)) <= EPS
                        for rgb in rgbs)):
            continue

        n = len(vals)
        keep = [True] * n
        for i in range(1, n - 1):
            if (vals[i] == vals[i - 1] and vals[i] == vals[i + 1]
                    and rgbs[i] == rgbs[i - 1] and rgbs[i] == rgbs[i + 1]):
                keep[i] = False

        prev_v = None
        keys = []
        for i in range(n):
            if not keep[i]:
                continue
            v = vals[i]
            key = {
                "time": r2(times[i]),
                "color": rgba_hex(rgbs[i], v),
            }
            if _opacity_key_stepped(prev_v, v):
                key["curve"] = "stepped"
            keys.append(key)
            prev_v = v

        if keys:
            slots_out[name] = {"color": keys}
    return slots_out


def build_slots_and_skin(scene, tr_name, bone_index, editor=False,
                         default_opacities=None, default_rgb=None):
    parts = scene["parts"]
    atlases = scene["atlases"]
    default_opacities = default_opacities or {}
    default_rgb = default_rgb or {}

    slots = []
    attachments = {}
    for p, name in zip(parts, part_slot_names(parts)):
        op = default_opacities.get(name, 1.0)
        rgb = default_rgb.get(name, (1.0, 1.0, 1.0))
        slot_entry = {"name": name, "bone": "root", "attachment": name}
        if p.get("blend"):
            slot_entry["blend"] = p["blend"]
        is_partial_alpha = 0.0 < op < 1.0 and abs(op - round(op)) > 1e-3
        if is_partial_alpha or any(abs(c - 1.0) > EPS for c in rgb):
            slot_entry["color"] = rgba_hex(rgb, op)
        elif op < 0.5:
            slot_entry["color"] = rgba_hex(rgb, 0.0)
        slots.append(slot_entry)

        page = atlases[p["atlas"]]
        aw, ah = page["width"], page["height"]
        hull_len, uvs, tris, verts, edges = build_spine_mesh_attachment(
            p, tr_name, bone_index, aw, ah,
        )
        width, height = mesh_image_size(p, atlases)

        attachments[name] = {
            name: {
                "type": "mesh",
                "uvs": uvs,
                "triangles": tris,
                "vertices": verts,
                "hull": hull_len,
                "edges": edges,
                "width": width,
                "height": height,
                "path": page["name"],
            }
        }

    skins = [{"name": "default", "attachments": attachments}]
    return slots, skins


def despike_series(series: list[float], threshold: float = 0.25) -> list[float]:
    """Smooth isolated 1-frame scale spikes caused by 3D out-of-plane projection cliff."""
    n = len(series)
    if n < 3:
        return series
    out = list(series)
    for i in range(1, n - 1):
        prev_val = out[i - 1]
        cur_val = out[i]
        next_val = series[i + 1]
        avg = (prev_val + next_val) / 2.0
        diff_prev = abs(cur_val - prev_val)
        diff_next = abs(cur_val - next_val)
        neighbor_diff = abs(prev_val - next_val)
        if diff_prev > threshold and diff_next > threshold and neighbor_diff < max(diff_prev, diff_next) * 0.6:
            out[i] = round(avg, 4)
    return out


def reduce_keys(times, values, key_builder):
    n = len(values)
    keep = [True] * n
    for i in range(1, n - 1):
        if values[i] == values[i - 1] and values[i] == values[i + 1]:
            keep[i] = False
    return [key_builder(times[i], values[i]) for i in range(n) if keep[i]]


def build_animation(scene, clip, setup, tr_name, bone_index, fps=FPS,
                    default_opacities=None, default_rgb=None):
    sampler, stop, _ = decode_clip(clip)
    n_frames = max(1, int(round(stop * fps)))
    times = [i / fps for i in range(n_frames + 1)]

    keys0 = clip_overrides(clip, scene["hash2tr"], sampler, 0.0)
    animated = [tr for tr in keys0 if tr in setup]

    series = {tr: [] for tr in animated}
    tilted_pos_neg = scene.get("tilted_pos_children_neg", scene.get("tilted_pos_children", set()))
    tilted_pos_pos = scene.get("tilted_pos_children_pos", set())
    tilted_rot_neg = scene.get("tilted_rot_children_neg", scene.get("tilted_rot_children", set()))
    tilted_rot_pos = scene.get("tilted_rot_children_pos", set())
    q_rx_neg90 = (-0.7071067811865475, 0.0, 0.0, 0.7071067811865475)
    q_rx_pos90 = (0.7071067811865475, 0.0, 0.0, 0.7071067811865475)
    for t in times:
        ov = clip_overrides(clip, scene["hash2tr"], sampler, t)
        for tr in animated:
            base = scene["TR"][tr]
            o = ov.get(tr, {})
            pos = o.get("pos", base["pos"])
            rot = o.get("rot", base["rot"])
            scale = o.get("scale", base["scale"])
            if tr in tilted_pos_neg and "pos" in o:
                pos = (pos[0], pos[2], -pos[1])
            elif tr in tilted_pos_pos and "pos" in o:
                pos = (pos[0], -pos[2], pos[1])

            if tr in tilted_rot_neg and "rot" in o:
                rot = quat_mul(q_rx_neg90, rot)
            elif tr in tilted_rot_pos and "rot" in o:
                rot = quat_mul(q_rx_pos90, rot)
            series[tr].append(decompose2d(local_matrix(pos, rot, scale)))

    bones_out = {}
    for tr in animated:
        s0 = setup[tr]
        frames = series[tr]
        track = {}

        rot_vals = []
        prev = 0.0
        for f in frames:
            off = norm180(f["rotation"] - s0["rotation"])
            while off - prev > 180.0:
                off -= 360.0
            while off - prev < -180.0:
                off += 360.0
            rot_vals.append(r2(off))
            prev = off
        if any(abs(v) > EPS for v in rot_vals):
            track["rotate"] = reduce_keys(
                times, rot_vals, lambda t, v: {"time": r2(t), "angle": v},
            )

        tx = [r2(f["x"] - s0["x"]) for f in frames]
        ty = [r2(f["y"] - s0["y"]) for f in frames]
        if any(abs(v) > EPS for v in tx) or any(abs(v) > EPS for v in ty):
            xy = list(zip(tx, ty))
            track["translate"] = reduce_keys(
                times, xy, lambda t, v: {"time": r2(t), "x": v[0], "y": v[1]},
            )

        def factor(cur, base):
            return cur / base if abs(base) > 1e-6 else 1.0

        sx = despike_series([r2(factor(f["scaleX"], s0["scaleX"])) for f in frames])
        sy = despike_series([r2(factor(f["scaleY"], s0["scaleY"])) for f in frames])
        if any(abs(v - 1) > EPS for v in sx) or any(abs(v - 1) > EPS for v in sy):
            xy = list(zip(sx, sy))
            track["scale"] = reduce_keys(
                times, xy, lambda t, v: {"time": r2(t), "x": v[0], "y": v[1]},
            )

        shy = [r2(norm180(f["shearY"] - s0["shearY"])) for f in frames]
        if any(abs(v) > EPS for v in shy):
            track["shear"] = reduce_keys(
                times, shy, lambda t, v: {"time": r2(t), "x": 0.0, "y": v},
            )

        if track:
            bones_out[tr_name[tr]] = track

    anim_out: dict[str, Any] = {"bones": bones_out}
    slots_out = build_slot_opacity_tracks(
        scene, clip, sampler, times,
        default_opacities=default_opacities, default_rgb=default_rgb,
    )
    if slots_out:
        anim_out["slots"] = slots_out
    return anim_out


def write_atlas(scene, out_dir, editor=False, skip_images=False):
    atlases = scene["atlases"]
    used_atlases = {p["atlas"] for p in scene["parts"] if "atlas" in p}
    lines = []
    img_dir = out_dir / "images" if editor else out_dir
    img_dir.mkdir(parents=True, exist_ok=True)

    for key in sorted(atlases.keys(), key=lambda k: (len(k), k)):
        if key not in used_atlases:
            continue
        page = atlases[key]
        w, h = page["width"], page["height"]
        src_name = page["name"]
        page_name = f"{src_name}.png"
        atlas_page = f"images/{page_name}" if editor else page_name
        target_png = img_dir / page_name
        if not skip_images or not target_png.exists():
            page["image"].save(target_png)
        lines += [
            "",
            atlas_page,
            f"size: {w},{h}",
            "format: RGBA8888",
            "filter: Linear,Linear",
            "repeat: none",
            src_name,
            "  rotate: false",
            "  xy: 0, 0",
            f"  size: {w}, {h}",
            f"  orig: {w}, {h}",
            "  offset: 0, 0",
            "  index: -1",
        ]
    (out_dir / "skeleton.atlas").write_text("\n".join(lines).lstrip("\n") + "\n", encoding="utf-8")


def world_scale(minx, maxx):
    if "SPINE_SCALE" in os.environ:
        return float(os.environ["SPINE_SCALE"])
    if TARGET_W <= 0:
        return 1.0
    w = maxx - minx
    return TARGET_W / w if w > EPS else 1.0


def _primary_idle_clip(clips):
    for c in clips:
        c_name = c.m_Name.lower()
        if c_name == "idle" or c_name.endswith("_idle"):
            return c
    for c in clips:
        if "idle" in c.m_Name.lower():
            return c
    return clips[0] if clips else None


def get_default_opacities(scene) -> dict[str, float]:
    """Determine setup pose opacity for each part from the primary idle clip at t=0.0."""
    part_names = part_slot_names(scene["parts"])
    has_onestore = any("onestore" in p["name"].lower() for p in scene["parts"])
    clips = [o.read() for o in scene["objs"] if o.type.name == "AnimationClip"]
    if not clips:
        res = {}
        for p, name in zip(scene["parts"], part_names):
            tr = p.get("go_tr_pid") or p.get("tr_pid")
            act = transform_active(tr, {}, scene["tr_to_path"], scene["TR"], scene.get("tr_initial_active"))
            base_alpha = p.get("initial_color", (1.0, 1.0, 1.0, 1.0))[3]
            res[name] = 0.0 if not act or (has_onestore and "google" in name.lower()) else base_alpha
        return res

    idle_clip = _primary_idle_clip(clips)

    sampler, _, _ = decode_clip(idle_clip)
    go_active, smr_alpha, smr_props = sample_clip_properties(idle_clip, sampler, 0.0)
    opacities = part_opacities(scene, go_active, smr_alpha, smr_props)
    res = {}
    for name, op in zip(part_names, opacities):
        if has_onestore and "google" in name.lower():
            res[name] = 0.0
        else:
            res[name] = op

    # Suppress co-existing expression variant sprites in default pose when a base head part is active
    has_base_head = any(
        ("head" in k.lower() or "parts_0" in k.lower()) and v >= 0.5
        for k, v in res.items()
    )
    if has_base_head:
        for k in list(res.keys()):
            if any(x in k.lower() for x in ("smile", "embarrass", "surprised", "face_ani")):
                res[k] = 0.0

    return res


def get_default_rgb(scene) -> dict[str, tuple[float, float, float]]:
    """Setup pose tint per part, read from the same clip/time as the opacities."""
    part_names = part_slot_names(scene["parts"])
    clips = [o.read() for o in scene["objs"] if o.type.name == "AnimationClip"]
    idle_clip = _primary_idle_clip(clips)
    if idle_clip is None:
        return {name: p.get("initial_color", (1.0, 1.0, 1.0, 1.0))[:3] for p, name in zip(scene["parts"], part_names)}
    sampler, _, _ = decode_clip(idle_clip)
    _, _, smr_props = sample_clip_properties(idle_clip, sampler, 0.0)
    return dict(zip(part_names, part_rgb(scene, smr_props)))


def export_spine(scene, out_dir: Path, editor=False, skip_images=False):
    # The output directory is created only once there is something to write:
    # a failed conversion must not leave an empty folder behind.
    default_opacities = get_default_opacities(scene)
    default_rgb = get_default_rgb(scene)
    bones, tr_name, bone_index, setup = build_bones(scene)

    # For MeshRenderer parts without bind poses, compute exact 2D projection bind matrix
    # so they align perfectly in Spine space regardless of 3D bone orientation.
    mesh_renderer_parts = [p for p in scene["parts"] if p.get("is_mesh_renderer")]
    if mesh_renderer_parts:
        TR = scene["TR"]
        children = {tr: [] for tr in TR}
        for tr, t in TR.items():
            f = t["father"]
            if f and f in TR:
                children[f].append(tr)

        spine_world = {}
        def calc_sw(tr, parent_M):
            s = setup[tr]
            rad = math.radians(s["rotation"])
            c, sn = math.cos(rad), math.sin(rad)
            sx, sy = s["scaleX"], s["scaleY"]
            M_loc = np.array([
                [c * sx, -sn * sy, s["x"]],
                [sn * sx,  c * sy, s["y"]],
                [0.0, 0.0, 1.0]
            ])
            M_w = parent_M @ M_loc
            spine_world[tr] = M_w
            for ch in children[tr]:
                calc_sw(ch, M_w)

        roots = [tr for tr, t in TR.items() if not t["father"] or t["father"] not in TR]
        for r in roots:
            calc_sw(r, np.eye(3))

        world_fn = make_world(scene["TR"])
        for p in mesh_renderer_parts:
            b = p["bones"][0]
            sw3 = spine_world.get(b, np.eye(3))
            sw4 = np.eye(4)
            sw4[0, 0] = sw3[0, 0]
            sw4[0, 1] = sw3[0, 1]
            sw4[0, 3] = sw3[0, 2]
            sw4[1, 0] = sw3[1, 0]
            sw4[1, 1] = sw3[1, 1]
            sw4[1, 3] = sw3[1, 2]
            try:
                sw4_inv = np.linalg.inv(sw4)
            except np.linalg.LinAlgError:
                sw4_inv = np.eye(4)
            M_unity = world_fn(p.get("go_tr_pid") or b)
            p["bind"] = [sw4_inv @ M_unity]

    slots, skins = build_slots_and_skin(
        scene, tr_name, bone_index, editor=editor,
        default_opacities=default_opacities, default_rgb=default_rgb,
    )

    world = make_world(scene["TR"])
    part_names = part_slot_names(scene["parts"])
    visible_parts = [
        p for p, name in zip(scene["parts"], part_names)
        if default_opacities.get(name, 1.0) >= 0.5
    ]
    if not visible_parts:
        visible_parts = scene["parts"]

    positions = skin_all(visible_parts, world)
    minx, miny, maxx, maxy = bounds_of(positions, pad=0.0)
    scale = world_scale(minx, maxx)
    if abs(scale - 1.0) > EPS:
        bones[0]["scaleX"] = r2(scale)
        bones[0]["scaleY"] = r2(scale)

    animations = {}
    clips = [o.read() for o in scene["objs"] if o.type.name == "AnimationClip"]
    for clip in clips:
        try:
            animations[clip.m_Name] = build_animation(
                scene, clip, setup, tr_name, bone_index,
                default_opacities=default_opacities, default_rgb=default_rgb,
            )
        except Exception as e:
            print(f"  !! animation {clip.m_Name} failed: {e}")

    skel = {
        "skeleton": {
            "hash": "unity2spine",
            "spine": SPINE_VERSION,
            "x": r2(minx * scale),
            "y": r2(miny * scale),
            "width": r2((maxx - minx) * scale),
            "height": r2((maxy - miny) * scale),
            "images": "./images/" if editor else "./",
            "audio": "",
        },
        "bones": bones,
        "slots": slots,
        "skins": skins,
        "animations": animations,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "skeleton.json"
    out_json.write_text(json.dumps(skel, separators=(",", ":")), encoding="utf-8")
    write_atlas(scene, out_dir, editor=editor, skip_images=skip_images)

    mode = "editor" if editor else "runtime"
    print(
        f"[{mode}] bones={len(bones)} slots={len(slots)} animations={len(animations)}"
        f"  scale={r2(scale)} (target_w={TARGET_W})"
    )
    for name, anim in animations.items():
        n_slots = len(anim.get("slots", {}))
        extra = f", {n_slots} animated slots" if n_slots else ""
        print(f"  anim {name}: {len(anim['bones'])} animated bones{extra}")
    print("wrote", out_json)
    print("wrote", out_dir / "skeleton.atlas")


def resolve_output(src: Path, output: Path | None, editor: bool) -> Path:
    if output is not None:
        return output
    parent = src.parent if src.is_file() else src.parent
    return parent / ("spine_editor" if editor else "spine")


def resolve_gif_dir(src: Path, output: Path | None) -> Path:
    if output is not None:
        return output / "gifs"
    parent = src.parent if src.is_file() else src.parent
    return parent / "gifs"


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "src",
        type=Path,
        help="Unity AssetBundle path (e.g. …/__data)",
    )
    p.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="output directory (default: <bundle-parent>/spine_editor, or spine with --runtime)",
    )
    p.add_argument(
        "--editor",
        action="store_true",
        help="Spine Editor import layout (default)",
    )
    p.add_argument(
        "--runtime",
        action="store_true",
        help="runtime export layout (images beside atlas, default dir: spine/)",
    )
    p.add_argument(
        "--both",
        action="store_true",
        help="write runtime and editor exports",
    )
    p.add_argument(
        "--skip-images",
        action="store_true",
        help="skip saving atlas images if they already exist",
    )
    p.add_argument(
        "--gif",
        action="store_true",
        help="also export AnimationClip previews as GIF",
    )
    p.add_argument(
        "--gif-only",
        action="store_true",
        help="export GIFs only (skip skeleton.json / atlas)",
    )
    p.add_argument(
        "--gif-width",
        type=int,
        default=GIF_W,
        metavar="PX",
        help=f"GIF canvas width (default: {GIF_W})",
    )
    p.add_argument(
        "--gif-fps",
        type=int,
        default=GIF_FPS,
        help=f"GIF frame rate (default: {GIF_FPS})",
    )
    p.add_argument(
        "--gif-bg",
        default=GIF_BG,
        metavar="RRGGBB",
        help=f"GIF background colour (default: {GIF_BG})",
    )
    p.add_argument(
        "--gif-clips",
        default=os.environ.get("GIF_CLIPS"),
        metavar="NAMES",
        help="comma-separated clip name filter (substring match)",
    )
    p.add_argument(
        "--gif-workers",
        type=int,
        default=GIF_WORKERS,
        metavar="N",
        help="parallel part render threads (0=auto, default: auto up to 8)",
    )
    args = p.parse_args(argv)

    src = args.src.resolve()
    if not src.exists():
        p.error(f"source not found: {src}")

    print(f"loading {src}")
    try:
        scene = load_scene(src)
    except ConversionError as e:
        # Loud: machine-readable on stderr, nothing left behind, non-zero exit.
        # 3 = the bundle is not a conversion target at all (it already carries
        #     an authored Spine skeleton); 2 = a real conversion failure.
        print(json.dumps({"source": str(src), "ok": False, **e.as_dict()}),
              file=sys.stderr)
        print(f"!! cannot convert {src.name}: {e}")
        raise SystemExit(3 if e.kind == "native-spine-bundle" else 2)
    for note in scene.get("repairs", []):
        print(f"  repaired bone slot: {note}")

    if args.gif or args.gif_only:
        gif_dir = resolve_gif_dir(src, args.output)
        export_gifs(
            scene, gif_dir,
            clip_filter=args.gif_clips,
            target_w=args.gif_width,
            fps=args.gif_fps,
            bg=args.gif_bg,
            workers=args.gif_workers,
        )

    if args.gif_only:
        return

    spine_export = os.environ.get("SPINE_EXPORT", "").lower()
    if args.both:
        modes = (False, True)
    elif args.runtime or spine_export == "runtime":
        modes = (False,)
    else:
        modes = (True,)  # default: editor

    for editor in modes:
        out = resolve_output(src, args.output, editor)
        if args.both and editor:
            out = out.parent / "spine_editor" if args.output is None else out
        elif args.both and not editor and args.output is None:
            out = out.parent / "spine"
        export_spine(scene, out, editor=editor, skip_images=args.skip_images)


if __name__ == "__main__":
    main()
