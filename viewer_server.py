"""Flask server: reads Unity AssetBundle and serves Spine data + textures for the viewer."""
from __future__ import annotations

import io
import json
import math
import os
import struct
import sys
import zlib
from bisect import bisect_right
from pathlib import Path

import cv2
import numpy as np
import UnityPy
from flask import Flask, jsonify, request, send_file
from PIL import Image
from UnityPy.helpers.MeshHelper import MeshHandler

SPINE_VERSION = "3.8.99"
FPS = 30
TARGET_W = int(os.environ.get("SPINE_TARGET_W", "1600"))
EPS = 1e-4
ATLAS_SUFFIX_RE = __import__("re").compile(r"(part\d+)$", __import__("re").I)

BUNDLE_PATH = Path(__file__).parent / "__data"

app = Flask(__name__, static_folder="static")
_scene_cache = None
_export_cache = None


# ====== math (same as unity_to_spine.py) ======

def mat4(m):
    return np.array([[getattr(m, f"e{r}{c}") for c in range(4)] for r in range(4)], dtype=np.float64)

def quat_to_m3(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)

def local_matrix(pos, rot, scale):
    T = np.eye(4); T[:3, 3] = pos
    R = np.eye(4); R[:3, :3] = quat_to_m3(rot)
    S = np.diag([scale[0], scale[1], scale[2], 1.0])
    return T @ R @ S

def norm180(a):
    return (a + 180.0) % 360.0 - 180.0

def decompose2d(M):
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    tx, ty = M[0, 3], M[1, 3]
    rotation = math.degrees(math.atan2(c, a))
    scaleX = math.hypot(a, c)
    scaleY = math.hypot(b, d)
    shearY = norm180(math.degrees(math.atan2(d, b)) - 90.0 - rotation)
    return dict(x=float(tx), y=float(ty), rotation=float(rotation),
                scaleX=float(scaleX), scaleY=float(scaleY), shearY=float(shearY))

def trs_to_spine(t):
    return decompose2d(local_matrix(t["pos"], t["rot"], t["scale"]))

def r2(v):
    return round(float(v), 4)

def atlas_key_from_name(name):
    if not name: return None
    m = ATLAS_SUFFIX_RE.search(name)
    return m.group(1).lower() if m else None

def path_hash(path):
    return zlib.crc32(path.encode("utf-8")) & 0xFFFFFFFF

def curve_size(binding):
    if binding.typeID == 4:
        return 4 if binding.attribute == 2 else 3
    return 1


# ====== AnimationClip decoding ======

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
            j = bisect_right(ts, t) - 1
            if j < 0: j = 0
            kt = ts[j]; co = cf[j]; dt = t - kt
            vals[idx] = ((co[0] * dt + co[1]) * dt + co[2]) * dt + co[3]
        if n_dense > 0 and dense.size:
            fr = (t - dense_begin) * dense_rate
            f0 = int(np.clip(np.floor(fr), 0, dense_frames - 1))
            f1 = min(f0 + 1, dense_frames - 1)
            a = fr - f0
            row0 = dense[f0 * n_dense:(f0 + 1) * n_dense]
            row1 = dense[f1 * n_dense:(f1 + 1) * n_dense]
            vals[n_stream:n_stream + n_dense] = row0 * (1 - a) + row1 * a
        if len(const):
            vals[n_stream + n_dense:] = const
        return vals
    return sampler, float(mc.m_StopTime), n_total


def clip_overrides(clip, hash2tr, sampler, t):
    vals = sampler(t)
    out = {}
    idx = 0
    for b in clip.m_ClipBindingConstant.genericBindings:
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
        idx += size
    return out


# ====== Scene loading ======

def load_scene(src: Path):
    env = UnityPy.load(str(src))
    objs = list(env.objects)
    byid = {o.path_id: o for o in objs}

    TR = {}
    go2tr = {}
    go_name = {}
    tr_children = {}
    for o in objs:
        if o.type.name == "Transform":
            d = o.read()
            p, r, s = d.m_LocalPosition, d.m_LocalRotation, d.m_LocalScale
            TR[o.path_id] = dict(
                pos=(p.x, p.y, p.z), rot=(r.x, r.y, r.z, r.w),
                scale=(s.x, s.y, s.z), father=getattr(d.m_Father, "path_id", 0),
            )
            go2tr[getattr(d.m_GameObject, "path_id", 0)] = o.path_id
            tr_children[o.path_id] = [getattr(c, "path_id", 0) for c in d.m_Children]
        elif o.type.name == "GameObject":
            go_name[o.path_id] = o.read().m_Name

    # Load atlases
    atlases = {}
    tex2atlas = {}
    for o in objs:
        if o.type.name != "Texture2D":
            continue
        data = o.read()
        key = atlas_key_from_name(data.m_Name)
        if key is None:
            continue
        img = data.image
        w, h = img.size
        # Convert to base64 for web
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        atlases[key] = {
            "key": key, "name": data.m_Name or key,
            "image": img, "width": w, "height": h,
            "base64": "data:image/png;base64," + __import__("base64").b64encode(buf.getvalue()).decode(),
        }
        tex2atlas[o.path_id] = key

    parts = []
    for o in objs:
        if o.type.name != "SkinnedMeshRenderer":
            continue
        smr = o.read()
        mp = getattr(smr.m_Mesh, "path_id", 0)
        if mp not in byid: continue
        mesh = byid[mp].read()
        mats = smr.m_Materials or []
        atlas = atlas_key_from_name(byid[getattr(mats[0], "path_id", 0)].read().m_Name) if mats else None
        if atlas is None: continue

        bones = [getattr(b, "path_id", 0) for b in smr.m_Bones]
        bind = [mat4(m) for m in mesh.m_BindPose]
        h = MeshHandler(mesh); h.process()
        vc = h.m_VertexCount
        v = np.array(h.m_Vertices, dtype=np.float64).reshape(vc, 3)
        uv = np.array(h.m_UV0, dtype=np.float64).reshape(vc, 2)
        vh = np.c_[v, np.ones(vc)]
        single = len(bones) == 1
        bi = bw = None
        if not single:
            bi_raw = np.array(h.m_BoneIndices)
            k = max(1, bi_raw.size // vc)
            bi = bi_raw.reshape(vc, k)
            if h.m_BoneWeights is not None and np.array(h.m_BoneWeights).size == vc * k:
                bw = np.array(h.m_BoneWeights, dtype=np.float64).reshape(vc, k)
            else:
                bw = np.zeros((vc, k)); bw[:, 0] = 1.0
        faces = []
        for sm in h.get_triangles():
            faces.append(np.array(sm).reshape(-1, 3))
        faces = np.vstack(faces) if faces else np.zeros((0, 3), int)
        aw, ah = atlases[atlas]["width"], atlases[atlas]["height"]
        src_px = np.c_[uv[:, 0] * aw, (1 - uv[:, 1]) * ah]
        go_tr_pid = go2tr.get(getattr(smr.m_GameObject, "path_id", 0))
        parts.append(dict(
            name=mesh.m_Name, atlas=atlas, kind="mesh",
            bones=bones, bind=bind, vh=vh, single=single,
            bi=bi, bw=bw, src_px=src_px, faces=faces,
            order=getattr(smr, "m_SortingOrder", 0), go_tr_pid=go_tr_pid,
        ))

    for o in objs:
        if o.type.name != "SpriteRenderer": continue
        sr = o.read()
        sp_ptr = getattr(sr, "m_Sprite", None)
        if not sp_ptr or getattr(sp_ptr, "path_id", 0) not in byid: continue
        sp_o = byid[sp_ptr.path_id]
        sp = sp_o.read()
        rd = sp.m_RD
        tex_pid = getattr(getattr(rd, "texture", None), "path_id", 0)
        atlas = tex2atlas.get(tex_pid)
        if atlas is None: continue
        tr_pid = go2tr.get(getattr(sr.m_GameObject, "path_id", 0))
        if tr_pid is None: continue
        rect = sp.m_Rect
        ptu = sp.m_PixelsToUnits or 100.0
        piv = sp.m_Pivot
        h = MeshHandler(rd, sp_o.version); h.process()
        vc = h.m_VertexCount
        sv = np.array(h.m_Vertices, dtype=np.float64).reshape(vc, 3)
        sv2 = np.c_[sv[:, :2], np.zeros(vc), np.ones(vc)]
        ah = atlases[atlas]["height"]
        ax = rect.x + (sv[:, 0] * ptu + piv.x * rect.width)
        ay = rect.y + (sv[:, 1] * ptu + piv.y * rect.height)
        src = np.c_[ax, ah - ay]
        faces = []
        for sm in h.get_triangles():
            faces.append(np.array(sm).reshape(-1, 3))
        faces = np.vstack(faces) if faces else np.zeros((0, 3), int)
        parts.append(dict(
            name=sp.m_Name, atlas=atlas, kind="sprite",
            tr_pid=tr_pid, sv2=sv2, src_px=src, faces=faces,
            order=getattr(sr, "m_SortingOrder", 0), go_tr_pid=tr_pid,
        ))

    parts.sort(key=lambda p: p["order"])

    animator_go = None
    for o in objs:
        if o.type.name == "Animator":
            animator_go = getattr(o.read().m_GameObject, "path_id", 0)
            break
    hash2tr = {}; tr_to_path = {}
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

    return dict(objs=objs, TR=TR, atlases=atlases, parts=parts, hash2tr=hash2tr,
                tr_to_path=tr_to_path, go2tr=go2tr, go_name=go_name)


# ====== Build Spine JSON ======

def build_export(scene):
    TR = scene["TR"]
    go2tr = scene["go2tr"]
    go_name = scene["go_name"]
    tr2go = {t: g for g, t in go2tr.items()}

    def raw_name(tr):
        return go_name.get(tr2go.get(tr, 0)) or f"bone_{tr}"
    used = {}; tr_name = {}
    for tr in TR:
        nm = raw_name(tr).replace("/", "_")
        if nm in used:
            used[nm] += 1; nm = f"{nm}#{used[nm]}"
        else:
            used[nm] = 0
        tr_name[tr] = nm

    children = {tr: [] for tr in TR}
    roots = []
    for tr, t in TR.items():
        f = t["father"]
        if f and f in TR: children[f].append(tr)
        else: roots.append(tr)

    ordered = []; parent_name = {}
    stack = list(reversed(roots))
    while stack:
        tr = stack.pop(); ordered.append(tr)
        for c in reversed(children[tr]):
            parent_name[c] = tr; stack.append(c)

    bones = [{"name": "root"}]
    setup = {}; bone_index = {"root": 0}
    for tr in ordered:
        s = trs_to_spine(TR[tr])
        setup[tr] = s
        parent = tr_name[parent_name[tr]] if tr in parent_name else "root"
        b = {"name": tr_name[tr], "parent": parent}
        if abs(s["x"]) > EPS: b["x"] = r2(s["x"])
        if abs(s["y"]) > EPS: b["y"] = r2(s["y"])
        if abs(s["rotation"]) > EPS: b["rotation"] = r2(s["rotation"])
        if abs(s["scaleX"] - 1) > EPS: b["scaleX"] = r2(s["scaleX"])
        if abs(s["scaleY"] - 1) > EPS: b["scaleY"] = r2(s["scaleY"])
        if abs(s["shearY"]) > EPS: b["shearY"] = r2(s["shearY"])
        bones.append(b)
        bone_index[tr_name[tr]] = len(bones) - 1

    # Slots & skin
    atlases = scene["atlases"]
    parts = scene["parts"]
    slots = []
    attachments = {}
    used_names = {}
    for p in parts:
        base = p["name"] or f"part_{p['order']}"
        name = base
        if name in used_names:
            used_names[name] += 1; name = f"{name}#{used_names[name]}"
        else:
            used_names[name] = 0

        # Determine slot bone
        if p["kind"] == "sprite":
            slot_bone = tr_name[p["tr_pid"]]
        elif p["single"]:
            slot_bone = tr_name[p["bones"][0]]
        else:
            # For multi-bone mesh, use root
            slot_bone = "root"

        slots.append({"name": name, "bone": slot_bone, "attachment": name})
        page = atlases[p["atlas"]]

        # Build vertices
        verts = []
        if p["kind"] == "sprite":
            bidx = bone_index[tr_name[p["tr_pid"]]]
            for v in p["sv2"]:
                verts += [1, bidx, r2(v[0]), r2(v[1]), 1.0]
        else:
            bones_list = p["bones"]
            bind = p["bind"]
            vh = p["vh"]
            bone_idx_list = [bone_index[tr_name[b]] for b in bones_list]
            if p["single"]:
                local = (bind[0] @ vh.T).T
                for pt in local:
                    verts += [1, bone_idx_list[0], r2(pt[0]), r2(pt[1]), 1.0]
            else:
                bi, bw = p["bi"], p["bw"]
                vc, k = bi.shape
                locals_by_bone = {j: (bind[j] @ vh.T).T for j in range(len(bones_list))}
                for i in range(vc):
                    acc = {}
                    for ki in range(k):
                        w = float(bw[i, ki])
                        if w <= 0: continue
                        j = int(bi[i, ki])
                        acc[j] = acc.get(j, 0.0) + w
                    if not acc:
                        acc[int(bi[i, 0])] = 1.0
                    verts.append(len(acc))
                    for j, w in acc.items():
                        pt = locals_by_bone[j][i]
                        verts += [bone_idx_list[j], r2(pt[0]), r2(pt[1]), r2(w)]

        uvs = []
        for sx, sy in p["src_px"]:
            uvs += [r2(sx / page["width"]), r2(sy / page["height"])]

        tris = [int(i) for i in p["faces"].ravel().tolist()]

        # Edge pairs
        edges_set = set()
        for a, b, c in np.asarray(p["faces"], dtype=np.int64).reshape(-1, 3):
            for i, j in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
                edges_set.add((min(i, j), max(i, j)))
        edges = []
        for i, j in sorted(edges_set):
            edges.extend([i, j])

        attachments[name] = {
            name: {
                "type": "mesh",
                "uvs": uvs,
                "triangles": tris,
                "vertices": verts,
                "hull": len(p["src_px"]),
                "edges": edges,
                "width": int(page["width"]),
                "height": int(page["height"]),
                "path": page["name"],
            }
        }

    skins = [{"name": "default", "attachments": attachments}]

    # Compute bounds
    def make_world_fn(TR_dict, overrides=None):
        overrides = overrides or {}
        wcache = {}
        def trs(pid):
            t = TR_dict[pid]; o = overrides.get(pid)
            if not o: return t["pos"], t["rot"], t["scale"]
            return (o.get("pos", t["pos"]), o.get("rot", t["rot"]), o.get("scale", t["scale"]))
        def world(pid):
            if pid in wcache: return wcache[pid]
            chain = []; q = pid
            while q and q in TR_dict:
                chain.append(q); q = TR_dict[q]["father"]
            M = np.eye(4)
            for q in reversed(chain):
                pos, rot, scale = trs(q)
                T = np.eye(4); T[:3, 3] = pos
                R = np.eye(4); R[:3, :3] = quat_to_m3(rot)
                S = np.diag([*scale, 1.0])
                M = M @ T @ R @ S
            wcache[pid] = M; return M
        return world

    world_fn = make_world_fn(TR)

    def skin_part(part, world):
        if part["kind"] == "sprite":
            M = world(part["tr_pid"])
            return (M @ part["sv2"].T).T[:, :2]
        bones_list, bind, vh = part["bones"], part["bind"], part["vh"]
        if part["single"]:
            M = world(bones_list[0]) @ bind[0]
            return (M @ vh.T).T[:, :2]
        bi, bw = part["bi"], part["bw"]
        vc, k = bi.shape
        BW = [world(b) @ bind[i] for i, b in enumerate(bones_list)]
        pos = np.zeros((vc, 3))
        for ki in range(k):
            w = bw[:, ki]
            if not np.any(w): continue
            for bidx in np.unique(bi[:, ki]):
                sel = bi[:, ki] == bidx
                pts = (BW[bidx] @ vh[sel].T).T[:, :3]
                pos[sel] += pts * w[sel, None]
        return pos[:, :2]

    positions = [skin_part(p, world_fn) for p in parts]
    allxy = np.vstack(positions)
    minx, miny = allxy.min(0); maxx, maxy = allxy.max(0)
    scale = TARGET_W / (maxx - minx) if (maxx - minx) > EPS else 1.0
    if abs(scale - 1.0) > EPS:
        bones[0]["scaleX"] = r2(scale)
        bones[0]["scaleY"] = r2(scale)

    # Animations
    animations = {}
    clips = [o.read() for o in scene["objs"] if o.type.name == "AnimationClip"]
    for clip in clips:
        try:
            sampler, stop, _ = decode_clip(clip)
            n_frames = max(1, int(round(stop * FPS)))
            times = [i / FPS for i in range(n_frames + 1)]
            keys0 = clip_overrides(clip, scene["hash2tr"], sampler, 0.0)
            animated = [tr for tr in keys0 if tr in setup]
            series = {tr: [] for tr in animated}
            for t in times:
                ov = clip_overrides(clip, scene["hash2tr"], sampler, t)
                for tr in animated:
                    base = TR[tr]
                    o = ov.get(tr, {})
                    pos = o.get("pos", base["pos"])
                    rot = o.get("rot", base["rot"])
                    scale_s = o.get("scale", base["scale"])
                    series[tr].append(decompose2d(local_matrix(pos, rot, scale_s)))
            bones_out = {}
            for tr in animated:
                s0 = setup[tr]
                frames = series[tr]
                track = {}
                rot_vals = []; prev = 0.0
                for f in frames:
                    off = norm180(f["rotation"] - s0["rotation"])
                    while off - prev > 180.0: off -= 360.0
                    while off - prev < -180.0: off += 360.0
                    rot_vals.append(r2(off)); prev = off
                if any(abs(v) > EPS for v in rot_vals):
                    track["rotate"] = [{"time": r2(times[i]), "angle": v} for i, v in enumerate(rot_vals)]
                tx = [r2(f["x"] - s0["x"]) for f in frames]
                ty = [r2(f["y"] - s0["y"]) for f in frames]
                if any(abs(v) > EPS for v in tx) or any(abs(v) > EPS for v in ty):
                    track["translate"] = [{"time": r2(times[i]), "x": tx[i], "y": ty[i]} for i in range(len(tx))]
                def factor(cur, base):
                    return cur / base if abs(base) > 1e-6 else 1.0
                sx = [r2(factor(f["scaleX"], s0["scaleX"])) for f in frames]
                sy = [r2(factor(f["scaleY"], s0["scaleY"])) for f in frames]
                if any(abs(v - 1) > EPS for v in sx) or any(abs(v - 1) > EPS for v in sy):
                    track["scale"] = [{"time": r2(times[i]), "x": sx[i], "y": sy[i]} for i in range(len(sx))]
                shy = [r2(norm180(f["shearY"] - s0["shearY"])) for f in frames]
                if any(abs(v) > EPS for v in shy):
                    track["shear"] = [{"time": r2(times[i]), "x": 0.0, "y": shy[i]} for i in range(len(shy))]
                if track:
                    bones_out[tr_name[tr]] = track
            animations[clip.m_Name] = {"bones": bones_out, "duration": round(stop, 2)}
        except Exception as e:
            print(f"  !! animation {clip.m_Name} failed: {e}")

    skel = {
        "skeleton": {
            "hash": "unity2spine", "spine": SPINE_VERSION,
            "x": r2(minx * scale), "y": r2(miny * scale),
            "width": r2((maxx - minx) * scale), "height": r2((maxy - miny) * scale),
            "images": "./",
        },
        "bones": bones, "slots": slots, "skins": skins, "animations": animations,
    }

    # Atlas data for frontend
    atlas_data = {}
    for key, page in atlases.items():
        atlas_data[key] = {
            "name": page["name"],
            "width": page["width"],
            "height": page["height"],
            "base64": page["base64"],
        }

    return skel, atlas_data


# ====== Flask routes ======

@app.route("/")
def index():
    return send_file("static/index.html")

@app.route("/api/scene")
def api_scene():
    global _scene_cache, _export_cache
    if _scene_cache is None:
        print("Loading bundle...")
        _scene_cache = load_scene(BUNDLE_PATH)
    if _export_cache is None:
        print("Building Spine export...")
        _export_cache = build_export(_scene_cache)
    skel, atlas_data = _export_cache
    return jsonify({"skeleton": skel, "atlases": atlas_data})


def main():
    global BUNDLE_PATH, _scene_cache, _export_cache
    import argparse
    parser = argparse.ArgumentParser(
        description="Run local Flask web viewer server for Unity AssetBundle -> Spine 3.8 preview."
    )
    parser.add_argument(
        "bundle",
        nargs="?",
        default=str(BUNDLE_PATH),
        help="Path to Unity AssetBundle file or directory (default: __data)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port number (default: 5000)")
    args = parser.parse_args()

    BUNDLE_PATH = Path(args.bundle)
    if not BUNDLE_PATH.exists():
        print(f"Error: Bundle path does not exist: {BUNDLE_PATH}")
        sys.exit(1)

    print("Starting Unity AssetBundle Viewer Server...")
    print(f"Bundle: {BUNDLE_PATH}")
    # Preload
    _scene_cache = load_scene(BUNDLE_PATH)
    _export_cache = build_export(_scene_cache)
    print(f"Ready! Open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
