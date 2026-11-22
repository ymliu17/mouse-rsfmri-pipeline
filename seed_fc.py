#!/usr/bin/env python3
"""
Seed-to-voxel functional connectivity from Allen-space cleaned fMRI.

Outputs:
  - seed_<name>_rmap.nii.gz
  - seed_<name>_zmap.nii.gz
"""

import argparse
import difflib
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def _load_lookup(path: Path):
    if path is None:
        return {}, {}
    df = pd.read_csv(path)
    id_col = None
    for c in ["coarse_id", "id", "structure_id", "roi_id"]:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        return {}, {}
    name_col = "acronym" if "acronym" in df.columns else "name" if "name" in df.columns else None
    id_to_name = {}
    name_to_id = {}
    for _, row in df.iterrows():
        rid = int(row[id_col])
        nm = str(row[name_col]) if name_col else str(rid)
        id_to_name[rid] = nm
        name_to_id[nm.lower()] = rid
    return id_to_name, name_to_id


def _normalize(name: str) -> str:
    return "".join(c.lower() for c in name if c.isalnum())


def _best_match(requested: str, lookup_names: list):
    req_norm = _normalize(requested)
    if not req_norm:
        return None, "empty", 0.0

    # Exact normalized match
    for n in lookup_names:
        if _normalize(n) == req_norm:
            return n, "exact", 1.0

    # Substring match
    for n in lookup_names:
        if req_norm in _normalize(n) or _normalize(n) in req_norm:
            return n, "substring", 0.8

    # Fuzzy match
    candidates = difflib.get_close_matches(requested, lookup_names, n=1, cutoff=0.6)
    if candidates:
        score = difflib.SequenceMatcher(None, requested.lower(), candidates[0].lower()).ratio()
        return candidates[0], "fuzzy", float(score)

    return None, "none", 0.0


def _sanitize(name: str):
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name)


def _seed_timeseries(img, seed_mask):
    dataobj = img.dataobj
    t_len = img.shape[3]
    mask_flat = seed_mask.reshape(-1)
    idx = np.flatnonzero(mask_flat)
    if idx.size == 0:
        raise RuntimeError("Seed mask is empty.")
    ts = np.zeros(t_len, dtype=np.float64)
    for t in range(t_len):
        vol = np.asarray(dataobj[..., t], dtype=np.float32).reshape(-1)
        ts[t] = float(vol[idx].mean())
    return ts, int(idx.size)


def _resolve_seed_names(requested_list, name_to_id, id_to_name):
    lookup_names = list(name_to_id.keys())
    # Use original case names if available in id_to_name
    id_to_pretty = {v.lower(): v for v in id_to_name.values()}

    mappings = []
    resolved = []
    for req in requested_list:
        matched_key, match_type, score = _best_match(req, lookup_names)
        if matched_key is None:
            mappings.append({
                "requested": req,
                "matched_name": "",
                "matched_id": "",
                "match_type": "none",
                "score": score,
            })
            continue
        rid = name_to_id[matched_key]
        pretty = id_to_pretty.get(matched_key, id_to_name.get(rid, matched_key))
        mappings.append({
            "requested": req,
            "matched_name": pretty,
            "matched_id": rid,
            "match_type": match_type,
            "score": score,
        })
        resolved.append((rid, pretty))
    return resolved, mappings


def main():
    ap = argparse.ArgumentParser(description="Seed-to-voxel FC (Pearson r + Fisher z).")
    ap.add_argument("--cleaned", type=Path, required=True,
                    help="Allen-space cleaned 4D NIfTI (e.g., *_bold_cleaned.nii.gz).")
    ap.add_argument("--labels", type=Path, required=True,
                    help="ROI label image (e.g., coarse_allen_labels.nii.gz).")
    ap.add_argument("--lookup", type=Path, default=None,
                    help="Optional lookup CSV for ROI names.")
    ap.add_argument("--seed-id", type=int, default=None,
                    help="Seed ROI ID.")
    ap.add_argument("--seed-name", type=str, default=None,
                    help="Seed ROI acronym/name (from lookup).")
    ap.add_argument("--seed-mask", type=Path, default=None,
                    help="Binary seed mask NIfTI.")
    ap.add_argument("--default-seeds", action="store_true",
                    help="Run common mouse rs-fMRI seeds: ACC/ACA, mPFC, RSP, Hippocampus, Thalamus, Striatum.")
    ap.add_argument("--seed-list", type=str, default=None,
                    help="Comma-separated seed names to run (uses lookup matching).")
    ap.add_argument("--mapping-out", type=Path, default=None,
                    help="Optional CSV to save requested→matched ROI mapping.")
    ap.add_argument("--brain-mask", type=Path, default=None,
                    help="Optional brain mask NIfTI; defaults to labels>0.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output directory.")
    ap.add_argument("--min-seed-voxels", type=int, default=50,
                    help="Minimum seed voxel count.")
    args = ap.parse_args()

    if not args.cleaned.exists():
        raise FileNotFoundError(f"Cleaned EPI not found: {args.cleaned}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Label image not found: {args.labels}")

    id_to_name, name_to_id = _load_lookup(args.lookup)

    multi_mode = args.default_seeds or (args.seed_list is not None)
    seed_spec = sum(x is not None for x in [args.seed_id, args.seed_name, args.seed_mask])
    if multi_mode and seed_spec > 0:
        raise ValueError("Use either --default-seeds/--seed-list OR a single seed (--seed-id/--seed-name/--seed-mask).")
    if not multi_mode and seed_spec != 1:
        raise ValueError("Provide exactly one seed: --seed-id, --seed-name, or --seed-mask.")

    labels_img = nib.load(args.labels)
    labels = labels_img.get_fdata().astype(np.int32)

    img = nib.load(args.cleaned, keep_file_open=True)
    if len(img.shape) != 4 or img.shape[3] < 2:
        ap.error("Cleaned EPI must be 4-D with at least two volumes.")

    def validate_grid(image, name):
        if image.shape != img.shape[:3] or not np.allclose(image.affine, img.affine, atol=1e-4):
            ap.error(f"{name} must share the cleaned EPI voxel grid.")

    validate_grid(labels_img, "Labels")
    if args.seed_mask:
        seed_image = nib.load(args.seed_mask)
        validate_grid(seed_image, "Seed mask")

    if args.brain_mask:
        brain_image = nib.load(args.brain_mask)
        validate_grid(brain_image, "Brain mask")
        brain_mask = brain_image.get_fdata() > 0
    else:
        brain_mask = labels > 0

    mask_flat = brain_mask.reshape(-1)
    mask_idx = np.flatnonzero(mask_flat)
    if mask_idx.size == 0:
        raise RuntimeError("Brain mask is empty.")

    dataobj = img.dataobj
    t_len = img.shape[3]

    sum_y = np.zeros(mask_idx.size, dtype=np.float64)
    sum_y2 = np.zeros(mask_idx.size, dtype=np.float64)

    for t in range(t_len):
        vol = np.asarray(dataobj[..., t], dtype=np.float32).reshape(-1)
        y = vol[mask_idx].astype(np.float64)
        sum_y += y
        sum_y2 += y * y

    mean_y = sum_y / t_len
    var_y = sum_y2 - t_len * mean_y * mean_y
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if multi_mode:
        if not name_to_id:
            raise ValueError("Lookup CSV is required for --default-seeds or --seed-list.")
        if args.default_seeds:
            requested = [
                "ACC", "ACA", "mPFC", "RSP", "Hippocampus", "Thalamus", "Striatum",
            ]
        else:
            requested = [s.strip() for s in args.seed_list.split(",") if s.strip()]

        resolved, mappings = _resolve_seed_names(requested, name_to_id, id_to_name)
        mapping_out = args.mapping_out or (args.out_dir / "seed_mapping.csv")
        pd.DataFrame(mappings).to_csv(mapping_out, index=False)
        print(f"Seed mapping saved: {mapping_out}")

        # Deduplicate seeds by ID
        seen = set()
        seed_items = []
        for rid, pretty in resolved:
            if rid in seen:
                continue
            seen.add(rid)
            seed_items.append((rid, pretty))
    else:
        if args.seed_mask:
            seed_items = [(None, args.seed_mask.stem)]
        elif args.seed_name:
            key = args.seed_name.lower()
            if key not in name_to_id:
                raise ValueError(f"Seed name not found in lookup: {args.seed_name}")
            seed_items = [(name_to_id[key], id_to_name.get(name_to_id[key], args.seed_name))]
        else:
            seed_items = [(args.seed_id, id_to_name.get(args.seed_id, str(args.seed_id)))]

    for seed_id, seed_name in seed_items:
        if args.seed_mask:
            seed_mask = seed_image.get_fdata() > 0
        else:
            seed_mask = labels == seed_id

        seed_name = _sanitize(seed_name)
        seed_ts, seed_voxels = _seed_timeseries(img, seed_mask)
        if seed_voxels < args.min_seed_voxels:
            print(f"Warning: seed {seed_name} too small ({seed_voxels} voxels); skipping.")
            continue
        if seed_ts.std() < 1e-6:
            print(f"Warning: seed {seed_name} time series is nearly constant; skipping.")
            continue

        sum_xy = np.zeros(mask_idx.size, dtype=np.float64)
        for t in range(t_len):
            vol = np.asarray(dataobj[..., t], dtype=np.float32).reshape(-1)
            y = vol[mask_idx].astype(np.float64)
            sum_xy += y * seed_ts[t]

        mean_x = float(seed_ts.mean())
        var_x = float((seed_ts * seed_ts).sum() - t_len * mean_x * mean_x)
        denom = np.sqrt(np.maximum(var_y, 0.0) * max(var_x, 1e-12))
        denom[denom == 0] = np.nan
        r = (sum_xy - t_len * mean_x * mean_y) / denom
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        r = np.clip(r, -0.9999, 0.9999)
        z = np.arctanh(r)

        rmap = np.zeros(mask_flat.shape[0], dtype=np.float32)
        zmap = np.zeros(mask_flat.shape[0], dtype=np.float32)
        rmap[mask_idx] = r.astype(np.float32)
        zmap[mask_idx] = z.astype(np.float32)
        rmap = rmap.reshape(labels.shape)
        zmap = zmap.reshape(labels.shape)

        r_path = args.out_dir / f"seed_{seed_name}_rmap.nii.gz"
        z_path = args.out_dir / f"seed_{seed_name}_zmap.nii.gz"
        nib.save(nib.Nifti1Image(rmap, img.affine, img.header), r_path)
        nib.save(nib.Nifti1Image(zmap, img.affine, img.header), z_path)

        print(f"Saved: {r_path}")
        print(f"Saved: {z_path}")
        print(f"QC: seed={seed_name}, voxels={seed_voxels}, "
              f"r[min,max]=({r.min():.3f},{r.max():.3f}), "
              f"z[min,max]=({z.min():.3f},{z.max():.3f})")


if __name__ == "__main__":
    main()
