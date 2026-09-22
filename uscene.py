"""Light Unity bundle reader: Transform hierarchy + animator path map + clips.

No textures, no meshes -- just what the euler probe and the oracle need.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import UnityPy


def path_hash(path: str) -> int:
    return zlib.crc32(path.encode("utf-8")) & 0xFFFFFFFF


def load_light(src: Path):
    env = UnityPy.load(str(src))
    objs = list(env.objects)

    TR = {}
    go2tr = {}
    go_name = {}
    tr_children = {}
    clips = []
    for o in objs:
        tn = o.type.name
        if tn == "Transform":
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
        elif tn == "GameObject":
            d = o.read()
            go_name[o.path_id] = d.m_Name
        elif tn == "AnimationClip":
            clips.append(o.read())

    animator_go = None
    for o in objs:
        if o.type.name == "Animator":
            animator_go = getattr(o.read().m_GameObject, "path_id", 0)
            break

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

    return dict(
        TR=TR, go2tr=go2tr, go_name=go_name, tr_children=tr_children,
        hash2tr=hash2tr, tr_to_path=tr_to_path, clips=clips,
    )


def bone_names(scene):
    """Same naming rule as unity_to_spine.build_bones."""
    TR, go2tr, go_name = scene["TR"], scene["go2tr"], scene["go_name"]
    tr2go = {t: g for g, t in go2tr.items()}
    used = {}
    tr_name = {}
    for tr in TR:
        nm = (go_name.get(tr2go.get(tr, 0)) or f"bone_{tr}").replace("/", "_")
        if nm in used:
            used[nm] += 1
            nm = f"{nm}#{used[nm]}"
        else:
            used[nm] = 0
        tr_name[tr] = nm
    return tr_name
