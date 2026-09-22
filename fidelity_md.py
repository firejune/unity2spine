"""Write fidelity.md: the short human table that fidelity.json backs.

Deterministic - it only reads fidelity.json and formats it.  Run from the repo
root after fidelity.py:

    python3 fidelity_md.py fidelity.json fidelity.md

The two thresholds are reported side by side on purpose: the worst-bone error is
not cleanly bimodal, so a single pass/fail boolean would be a judgement dressed
up as a measurement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt(value, digits=3):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    doc = json.loads(src.read_text())
    rigs = doc["rigs"]
    strict = doc["thresholds"]["strict"]
    loose = doc["thresholds"]["loose"]
    summary = doc["summary"]
    rot = doc["distribution"]["worst_bone_rotation_deg"]
    pos = doc["distribution"]["worst_bone_position_pct"]
    gap = doc["distribution"]["bimodality"]

    lines = [
        "# Converted Rig Pose Fidelity Report",
        "",
        "Generated automatically from `fidelity.json`. Do not edit manually.",
        "",
        "**Applies to converted rigs.** Measures angular and positional divergence against ground truth Unity curves.",
        "",
        "## Measurement Parameters",
        "",
        f"- What: {doc['what']}",
        f"- Sampling: {doc['sampling']}",
        f"- Units: Rotation = {doc['units']['rotation']} · Position = {doc['units']['position']}",
        "",
        "## Dual Threshold Criteria",
        "",
        f"{doc['thresholds']['why_two']}",
        "",
        "| Criteria | Rotation | Position | Field | Pass Rate |",
        "|---|---:|---:|---|---:|",
        f"| Strict | ≤ {strict['rotation_deg']}° | ≤ {strict['position_pct']}% | "
        f"`{strict['field']}` | **{summary['pose_faithful_strict']} / {summary['rigs']}** |",
        f"| Loose | ≤ {loose['rotation_deg']}° | ≤ {loose['position_pct']}% | "
        f"`{loose['field']}` | **{summary['pose_faithful_loose']} / {summary['rigs']}** |",
        "",
        f"Distribution verdict: `bimodal` = {str(gap.get('bimodal')).lower()} — "
        f"widest splitting band: "
        f"{fmt(gap.get('widest_splitting_band', {}).get('between', [None, None])[0])}° ↔ "
        f"{fmt(gap.get('widest_splitting_band', {}).get('between', [None, None])[1])}° "
        f"(width: {fmt(gap.get('widest_splitting_band', {}).get('log10_width'), 2)} decade, "
        f"below: {gap.get('widest_splitting_band', {}).get('rigs_below')} · "
        f"above: {gap.get('widest_splitting_band', {}).get('rigs_above')}).",
        "",
        "## Worst Bone Error Distribution",
        "",
        "| Axis | n | Median | p75 | p90 | p95 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Rotation (deg) | {rot.get('n', 0)} | {fmt(rot.get('median'))} | {fmt(rot.get('p75'))} | "
        f"{fmt(rot.get('p90'))} | {fmt(rot.get('p95'))} | {fmt(rot.get('max'))} |",
        f"| Position (%) | {pos.get('n', 0)} | {fmt(pos.get('median'), 4)} | {fmt(pos.get('p75'), 4)} | "
        f"{fmt(pos.get('p90'), 4)} | {fmt(pos.get('p95'), 4)} | {fmt(pos.get('max'), 4)} |",
        "",
        "| Rotation Range | " + " | ".join(rot.get("histogram", {}).keys()) + " |",
        "|---|" + "---|" * len(rot.get("histogram", {})),
        "| Rig Count | " + " | ".join(str(v) for v in rot.get("histogram", {}).values()) + " |",
        "",
        "## Rigs Failing Loose Threshold",
        "",
        "Out-of-plane rotation is typically the primary cause — Unity evaluates 3D transforms before flattening, "
        "while Spine evaluates planar 2D transforms along hierarchies. Non-planar counts transforms rotated off the Z-axis in rest pose.",
        "",
        "| Rig | Worst Rotation (deg) | Worst Position (%) | Non-Planar / Total | Worst Bone |",
        "|---|---:|---:|---:|---|",
    ]
    failing = sorted((r for r in rigs if not r["pose_faithful_loose"]),
                     key=lambda r: -(r["rotation_deg"]["worst"] or 0))
    for rig in failing:
        worst = rig.get("worst_bone") or {}
        lines.append(
            f"| `{rig['bundle']}` | {fmt(rig['rotation_deg']['worst'], 2)} | "
            f"{fmt(rig['position_pct']['worst'], 2)} | "
            f"{rig.get('non_planar_rest_transforms')} / {rig.get('transforms')} | "
            f"`{worst.get('bone', '-')}` |")
    if not failing:
        lines.append("| - | - | - | - | None |")

    lines += [
        "",
        "## Rigs Passing Loose But Failing Strict Threshold",
        "",
        "| Rig | Worst Rotation (deg) | Worst Position (%) | Non-Planar / Total |",
        "|---|---:|---:|---:|",
    ]
    middle = sorted((r for r in rigs if r["pose_faithful_loose"] and not r["pose_faithful"]),
                    key=lambda r: -(r["rotation_deg"]["worst"] or 0))
    for rig in middle:
        lines.append(
            f"| `{rig['bundle']}` | {fmt(rig['rotation_deg']['worst'], 2)} | "
            f"{fmt(rig['position_pct']['worst'], 2)} | "
            f"{rig.get('non_planar_rest_transforms')} / {rig.get('transforms')} |")
    if not middle:
        lines.append("| - | - | - | None |")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}: {summary['rigs']} rigs, strict "
          f"{summary['pose_faithful_strict']}, loose {summary['pose_faithful_loose']}")


if __name__ == "__main__":
    main()
