#!/usr/bin/env python3
"""
Build a coarse Allen atlas from the fine CCF annotation.

Outputs:
  - coarse_allen_labels.nii.gz
  - coarse_roi_lookup.csv
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def _load_ontology(json_path: Path):
    data = json.loads(json_path.read_text())
    root = data["msg"][0]

    parent = {}
    depth = {}
    name = {}
    acronym = {}

    def walk(node, d, parent_id):
        nid = int(node["id"])
        parent[nid] = parent_id
        depth[nid] = d
        name[nid] = node.get("name", str(nid))
        acronym[nid] = node.get("acronym", str(nid))
        for child in node.get("children", []):
            walk(child, d + 1, nid)

    walk(root, 0, None)
    return parent, depth, name, acronym


def _ancestor_at_depth(nid, target_depth, parent, depth):
    cur = nid
    if cur not in depth:
        return nid
    while cur is not None and depth[cur] > target_depth:
        cur = parent.get(cur)
    return cur if cur is not None else nid


def _build_mapping_for_depth(
    label_ids,
    label_counts,
    parent,
    depth,
    target_depth,
    min_voxels,
):
    # Initial mapping to target depth.
    mapping = {rid: _ancestor_at_depth(rid, target_depth, parent, depth) for rid in label_ids}

    # Iteratively promote small regions to their parent.
    for _ in range(5):
        counts = {}
        for rid, anc in mapping.items():
            counts[anc] = counts.get(anc, 0) + label_counts[rid]

        changed = False
        for rid, anc in list(mapping.items()):
            if counts.get(anc, 0) >= min_voxels:
                continue
            cur = anc
            while cur is not None and counts.get(cur, 0) < min_voxels:
                cur = parent.get(cur)
            if cur is None:
                cur = anc
            if cur != anc:
                mapping[rid] = cur
                changed = True
        if not changed:
            break

    counts = {}
    for rid, anc in mapping.items():
        counts[anc] = counts.get(anc, 0) + label_counts[rid]
    return mapping, counts


def _pick_best_depth(
    label_ids,
    label_counts,
    parent,
    depth,
    start_depth,
    min_rois,
    max_rois,
    min_voxels,
):
    target_mid = (min_rois + max_rois) / 2
    best = None

    for d in range(start_depth, -1, -1):
        mapping, counts = _build_mapping_for_depth(
            label_ids, label_counts, parent, depth, d, min_voxels
        )
        n_rois = len(counts)
        score = 0 if min_rois <= n_rois <= max_rois else abs(n_rois - target_mid)
        candidate = (score, -d, n_rois, d, mapping, counts)
        if best is None or candidate < best:
            best = candidate
        if min_rois <= n_rois <= max_rois:
            break
    return best


def main():
    default_out = Path("allen_ccf")
    default_labels = default_out / "allen_labels_100um.nii.gz"
    default_ont = default_out / "allen_structure_graph.json"

    ap = argparse.ArgumentParser(description="Build coarse Allen atlas (~40-60 ROIs).")
    ap.add_argument("--labels", type=Path, default=default_labels,
                    help="Fine Allen label image (100um).")
    ap.add_argument("--ontology", type=Path, default=default_ont,
                    help="Allen structure graph JSON.")
    ap.add_argument("--out-dir", type=Path, default=default_out,
                    help="Output directory.")
    ap.add_argument("--start-depth", type=int, default=5,
                    help="Ontology depth to start collapsing from.")
    ap.add_argument("--min-rois", type=int, default=40,
                    help="Minimum target ROI count.")
    ap.add_argument("--max-rois", type=int, default=60,
                    help="Maximum target ROI count.")
    ap.add_argument("--min-voxels", type=int, default=200,
                    help="Minimum voxel count per ROI; smaller ROIs are merged up.")
    args = ap.parse_args()

    if not args.labels.exists():
        raise FileNotFoundError(f"Label image not found: {args.labels}")
    if not args.ontology.exists():
        raise FileNotFoundError(f"Ontology JSON not found: {args.ontology}")

    parent, depth, name, acronym = _load_ontology(args.ontology)

    label_img = nib.load(args.labels)
    label_data = label_img.get_fdata().astype(np.int32)
    label_flat = label_data.reshape(-1)
    label_ids, label_counts = np.unique(label_flat[label_flat > 0], return_counts=True)
    label_ids = [int(x) for x in label_ids.tolist()]
    label_counts = {int(i): int(c) for i, c in zip(label_ids, label_counts)}

    best = _pick_best_depth(
        label_ids=label_ids,
        label_counts=label_counts,
        parent=parent,
        depth=depth,
        start_depth=args.start_depth,
        min_rois=args.min_rois,
        max_rois=args.max_rois,
        min_voxels=args.min_voxels,
    )
    _, _, n_rois, chosen_depth, mapping, counts = best

    # Allen IDs can be large and sparse. Map only IDs present in the image.
    coarse_flat = np.zeros(label_flat.shape, dtype=np.int32)
    positive = label_flat > 0
    mapped = np.array([mapping.get(rid, rid) for rid in label_ids], dtype=np.int32)
    coarse_flat[positive] = mapped[np.searchsorted(label_ids, label_flat[positive])]
    coarse_data = coarse_flat.reshape(label_data.shape)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_labels = args.out_dir / "coarse_allen_labels.nii.gz"
    coarse_image = nib.Nifti1Image(coarse_data, label_img.affine, label_img.header)
    coarse_image.set_data_dtype(np.int32)
    nib.save(coarse_image, out_labels)

    # Lookup table
    rows = []
    for cid, nvox in sorted(counts.items(), key=lambda x: x[0]):
        rows.append({
            "coarse_id": int(cid),
            "acronym": acronym.get(cid, str(cid)),
            "name": name.get(cid, str(cid)),
            "depth": int(depth.get(cid, -1)),
            "parent_id": parent.get(cid),
            "n_voxels": int(nvox),
        })
    out_lookup = args.out_dir / "coarse_roi_lookup.csv"
    pd.DataFrame(rows).to_csv(out_lookup, index=False)

    print(f"Coarse atlas saved: {out_labels}")
    print(f"Lookup table saved: {out_lookup}")
    print(f"Selected depth={chosen_depth}, n_rois={n_rois}, min_voxels={args.min_voxels}")


if __name__ == "__main__":
    main()
