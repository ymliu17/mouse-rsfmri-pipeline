#!/usr/bin/env python3
"""Convert Allen CCFv3 NRRDs to templates, labels and a mask for the pipeline."""

import argparse
from pathlib import Path

import nibabel as nib
import nrrd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nrrd-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("allen_ccf"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for resolution, suffix in ((25, ""), (100, "_100um")):
        template, _ = nrrd.read(str(args.nrrd_dir / f"average_template_{resolution}.nrrd"))
        mm = resolution / 1000
        # Reproduce the source workflow's voxel-axis coordinate convention.
        affine = np.array([[-mm, 0, 0, (template.shape[0] - 1) * mm],
                           [0, -mm, 0, (template.shape[1] - 1) * mm],
                           [0, 0, mm, 0],
                           [0, 0, 0, 1]])
        if resolution == 100:
            labels, _ = nrrd.read(str(args.nrrd_dir / "annotation_100.nrrd"))
            if labels.shape != template.shape:
                parser.error("The 100 µm average template and annotation must share a grid.")
            nib.save(nib.Nifti1Image(labels.astype(np.int32), affine),
                     args.out_dir / "allen_labels_100um.nii.gz")
            nib.save(nib.Nifti1Image((labels > 0).astype(np.uint8), affine),
                     args.out_dir / "allen_brain_mask_100um.nii.gz")
        nib.save(nib.Nifti1Image(template.astype(np.float32), affine),
                 args.out_dir / f"allen_template{suffix}.nii.gz")


if __name__ == "__main__":
    main()
