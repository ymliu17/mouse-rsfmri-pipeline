#!/usr/bin/env python3
"""
ROI-to-ROI functional connectivity from Allen-space cleaned fMRI.

Outputs:
  - roi_timeseries.tsv
  - fc_corr.tsv
  - fc_z.tsv
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def _load_lookup(path: Path):
    if path is None:
        return {}
    df = pd.read_csv(path)
    id_col = None
    for c in ["coarse_id", "id", "structure_id", "roi_id"]:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        return {}
    acr_col = "acronym" if "acronym" in df.columns else None
    mapping = {}
    for _, row in df.iterrows():
        rid = int(row[id_col])
        if acr_col:
            mapping[rid] = str(row[acr_col])
    return mapping


def _compute_timeseries(img, labels, label_ids, counts):
    dataobj = img.dataobj
    t_len = img.shape[3]
    ts = np.zeros((t_len, len(label_ids)), dtype=np.float32)
    selected = np.isin(labels, label_ids)
    indices = np.searchsorted(label_ids, labels[selected])
    for t in range(t_len):
        vol = np.asarray(dataobj[..., t], dtype=np.float32)
        sums = np.bincount(indices, weights=vol[selected], minlength=len(label_ids))
        ts[t, :] = sums / counts
    return ts


def main():
    ap = argparse.ArgumentParser(description="Compute ROI-to-ROI FC (Pearson r + Fisher z).")
    ap.add_argument("--cleaned", type=Path, required=True,
                    help="Allen-space cleaned 4D NIfTI (e.g., *_bold_cleaned.nii.gz).")
    ap.add_argument("--labels", type=Path, required=True,
                    help="ROI label image (e.g., coarse_allen_labels.nii.gz).")
    ap.add_argument("--lookup", type=Path, default=None,
                    help="Optional lookup CSV for ROI acronyms.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output directory.")
    ap.add_argument("--min-voxels", type=int, default=50,
                    help="Exclude ROIs with fewer than this voxel count.")
    args = ap.parse_args()

    if not args.cleaned.exists():
        raise FileNotFoundError(f"Cleaned EPI not found: {args.cleaned}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Label image not found: {args.labels}")

    img = nib.load(args.cleaned, keep_file_open=True)
    labels_img = nib.load(args.labels)
    if (len(img.shape) != 4 or img.shape[3] < 2 or labels_img.shape != img.shape[:3]
            or not np.allclose(labels_img.affine, img.affine, atol=1e-4)):
        ap.error("Cleaned EPI needs at least two volumes and must share the atlas voxel grid.")
    labels = labels_img.get_fdata().astype(np.int32)

    label_flat = labels.reshape(-1)
    label_ids, counts = np.unique(label_flat[label_flat > 0], return_counts=True)
    label_ids = [int(x) for x in label_ids.tolist()]
    counts = np.array(counts, dtype=np.float64)

    keep = counts >= args.min_voxels
    if not np.all(keep):
        dropped = int(np.sum(~keep))
        print(f"Warning: dropping {dropped} ROIs smaller than {args.min_voxels} voxels")
        label_ids = [rid for rid, k in zip(label_ids, keep) if k]
        counts = counts[keep]

    if len(label_ids) == 0:
        raise RuntimeError("No ROIs left after size filtering.")

    ts = _compute_timeseries(img, labels, label_ids, counts)

    # Drop constant columns
    std = ts.std(axis=0)
    keep = std > 1e-6
    if not np.all(keep):
        dropped = int(np.sum(~keep))
        print(f"Warning: dropping {dropped} constant ROIs")
        ts = ts[:, keep]
        label_ids = [rid for rid, k in zip(label_ids, keep) if k]

    if not label_ids:
        ap.error("All ROI time series are constant.")
    centered = ts.astype(np.float64) - ts.mean(axis=0, dtype=np.float64)
    ts_z = centered / np.linalg.norm(centered, axis=0)
    r = ts_z.T @ ts_z
    np.fill_diagonal(r, 1.0)
    r = np.clip(r, -0.9999, 0.9999)
    z = np.arctanh(r)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts_path = args.out_dir / "roi_timeseries.tsv"
    r_path = args.out_dir / "fc_corr.tsv"
    z_path = args.out_dir / "fc_z.tsv"

    name_map = _load_lookup(args.lookup)
    col_names = [name_map.get(rid, str(rid)) for rid in label_ids]

    pd.DataFrame(ts, columns=col_names).to_csv(ts_path, sep="\t", index=False, float_format="%.6f")
    pd.DataFrame(r, index=col_names, columns=col_names).to_csv(
        r_path, sep="\t", float_format="%.6f"
    )
    pd.DataFrame(z, index=col_names, columns=col_names).to_csv(
        z_path, sep="\t", float_format="%.6f"
    )

    finite = np.isfinite(z)
    print(f"Saved: {ts_path}")
    print(f"Saved: {r_path}")
    print(f"Saved: {z_path}")
    print(f"QC: n_rois={len(label_ids)}, r[min,max]=({r.min():.3f},{r.max():.3f}), "
          f"z[min,max]=({z[finite].min():.3f},{z[finite].max():.3f})")


if __name__ == "__main__":
    main()
