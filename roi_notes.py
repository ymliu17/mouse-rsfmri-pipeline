#!/usr/bin/env python3
"""
Generate roi_notes.csv: one row per ROI with acronym, full name, category,
and number of leaf atlas regions merged into it.
"""

import argparse
import json
import pandas as pd
from pathlib import Path

# Manual lookup for IDs missing from the ontology JSON
# (these are root/summary nodes not fully described in the standard graph)
MANUAL = {
    0:   ("root",  "Outside brain / background",         "background"),
    5:   ("WB",    "Whole brain",                        "whole brain"),
    13:  ("CH+",   "Cerebrum (alternate node)",          "gray matter"),
    24:  ("MB",    "Midbrain (summary node)",            "gray matter"),
    32:  ("HB",    "Hindbrain (summary node)",           "gray matter"),
    40:  ("SC",    "Spinal cord (summary node)",         "gray matter"),
    172: ("VS+",   "Ventricular system (alt node)",      "ventricle"),
    176: ("FT",    "Fiber tracts (summary node)",        "fiber tract"),
    265: ("AQ+",   "Cerebral aqueduct (alt node)",       "ventricle"),
    273: ("V4+",   "Fourth ventricle (alt node)",        "ventricle"),
    282: ("c+",    "Central canal (alt node)",           "ventricle"),
}

CATEGORY_MAP = {
    8:    "gray matter",   # grey
    343:  "gray matter",   # BS  Brain stem
    512:  "gray matter",   # CB  Cerebellum
    567:  "gray matter",   # CH  Cerebrum
    73:   "ventricle",     # VS  ventricular systems
    81:   "ventricle",     # VL  lateral ventricle
    124:  "ventricle",     # IVF interventricular foramen
    129:  "ventricle",     # V3  third ventricle
    140:  "ventricle",     # AQ  cerebral aqueduct
    145:  "ventricle",     # V4  fourth ventricle
    164:  "ventricle",     # c   central canal
    960:  "fiber tract",   # cbf
    967:  "fiber tract",   # cm  cranial nerves
    983:  "fiber tract",   # lfbs
    991:  "fiber tract",   # mfbs
    1000: "fiber tract",   # eps
    1040: "other",         # grv of CBX
}

def build_full_map(node, result=None):
    if result is None:
        result = {}
    result[node["id"]] = {"acronym": node.get("acronym", "?"),
                          "name":    node.get("name", "?")}
    for c in node.get("children", []):
        build_full_map(c, result)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-ids", type=Path, required=True, help="One *_roi_ids_merged.tsv.")
    parser.add_argument("--ontology", type=Path,
                        default=Path("allen_ccf/allen_structure_graph.json"))
    parser.add_argument("--out", type=Path, default=Path("roi_notes.csv"))
    args = parser.parse_args()

    data = json.loads(args.ontology.read_text())
    lookup = build_full_map(data["msg"][0])
    roi_df = pd.read_csv(args.roi_ids, sep="\t")

    records = []
    for _, row in roi_df.iterrows():
        aid = int(row["ancestor_id"])
        acr = row["acronym"]
        n   = int(row["n_rois_merged"])

        info      = lookup.get(aid, {})
        full_name = info.get("name", MANUAL.get(aid, ("?", "unknown", "other"))[1])
        cat       = CATEGORY_MAP.get(aid, MANUAL.get(aid, ("?", "?", "other"))[2])

        records.append({
            "roi_index":      len(records),
            "ancestor_id":    aid,
            "acronym":        acr,   # use numeric label as-is (matches FC matrix columns)
            "full_name":      full_name,
            "category":       cat,
            "n_leaf_regions": n,
        })

    df = pd.DataFrame(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
