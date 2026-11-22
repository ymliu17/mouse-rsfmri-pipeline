#!/usr/bin/env python3
"""Collect the upper triangle of matching FC matrices into a CSV."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derivatives", type=Path, default=Path("derivatives"))
    parser.add_argument("--out", type=Path, default=Path("fc_all_subjects.csv"))
    parser.add_argument("--groups", type=Path, help="Optional TSV with subject and group columns.")
    parser.add_argument("--run", type=int, default=1, help="Run number to collect (default: 1).")
    args = parser.parse_args()
    groups = {}
    if args.groups:
        table = pd.read_csv(args.groups, sep="\t", dtype=str, keep_default_na=False)
        if not {"subject", "group"}.issubset(table.columns):
            parser.error("--groups requires subject and group columns")
        ids = table["subject"].str.removeprefix("sub-").str.replace("-", "", regex=False)
        if ids.duplicated().any() or ids.eq("").any():
            parser.error("--groups contains missing or duplicate subject IDs")
        groups = dict(zip(ids, table["group"]))

    rows = []
    roi_order = None
    pattern = f"sub-*/connectivity/*_run-{args.run:02d}_fc_matrix_fisherz.tsv"
    for path in sorted(args.derivatives.glob(pattern)):
        subject = path.parent.parent.name.removeprefix("sub-")
        fc = pd.read_csv(path, sep="\t", index_col=0)
        labels = fc.columns.tolist()
        if fc.index.astype(str).tolist() != labels or not labels:
            parser.error(f"FC row and column labels differ: {path}")
        if roi_order is None:
            roi_order = labels
        if labels != roi_order:
            parser.error(f"ROI labels/order differ across FC matrices: {path}")
        row = {"subject": subject}
        if args.groups:
            row["group"] = groups.get(subject, "")
        idx_i, idx_j = np.triu_indices(len(labels), k=1)
        row.update({f"roi-{i}_roi-{j}": value
                    for i, j, value in zip(idx_i, idx_j, fc.to_numpy()[idx_i, idx_j])})
        rows.append(row)

    if not rows:
        parser.error(f"No FC matrices for run {args.run} found under {args.derivatives}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False, float_format="%.6f")
    print(f"Saved {args.out} ({len(rows)} subjects)")


if __name__ == "__main__":
    main()
