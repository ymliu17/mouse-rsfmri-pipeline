#!/usr/bin/env python3
"""Preprocess mouse rs-fMRI NIfTI runs and compute Allen CCF connectivity."""

import argparse
import csv
import math
import re
import json
import logging
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS (paths and acquisition parameters can be overridden on the CLI)
# ─────────────────────────────────────────────────────────────────────────────

DICOM_DIR = Path("NPIL_DICOM")
DERIV_DIR = Path("derivatives")
QC_DIR = Path("QC")
ATLAS_DIR = Path("allen_ccf")
ALLEN_TEMPLATE = ATLAS_DIR / "allen_template.nii.gz"
ALLEN_TEMPLATE_FUNC = ATLAS_DIR / "allen_template_100um.nii.gz"
ALLEN_LABELS_FUNC = ATLAS_DIR / "allen_labels_100um.nii.gz"
ALLEN_BRAIN_MASK_FUNC = ATLAS_DIR / "allen_brain_mask_100um.nii.gz"
ALLEN_ONTOLOGY_JSON = ATLAS_DIR / "allen_structure_graph.json"

TR = None  # Use the NIfTI header by default (AFNI).
N_TRIM = 10
HP_FREQ = 0.01
LP_FREQ = 0.10
ROI_MERGE_DEPTH = 2
BET_F_ANAT = 0.35
BET_F_FUNC = 0.30
FD_RADIUS = 50.0  # Historical rotational conversion, in mm; configurable.

# ANTs registration parameters (Rigid, Mattes MI, good for EPI→T2)
ANTS_METRIC_WEIGHT = 1
ANTS_BINS          = 32
ANTS_SAMPLING      = "Regular"
ANTS_SAMPLING_PCT  = 0.25
ANTS_CONVERGENCE   = "1000x500x250x100"
ANTS_SHRINK        = "8x4x2x1"
ANTS_SMOOTH        = "3x2x1x0vox"

# ANTs SyN parameters (T2 → Allen CCF deformable stage)
ANTS_SYN_CONVERGENCE = "100x70x50x20"
ANTS_SYN_SHRINK      = "8x4x2x1"
ANTS_SYN_SMOOTH      = "3x2x1x0vox"

# QC thresholds
FD_THRESHOLD  = 0.10   # mm – strict for mouse (brain ~12 mm)
NCC_THRESHOLD = 0.70   # normalized cross-correlation for registration QC

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL-PATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _find_tool(name: str, variable: str, subdirectory: str = '') -> Path:
    """Return path to an executable, searching search_dir then PATH."""
    location = os.environ.get(variable)
    if location:
        candidate = Path(location).expanduser() / subdirectory / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.absolute()
    found = shutil.which(name)
    if found:
        return Path(found)
    raise FileNotFoundError(f"Cannot find {name}. Add it to PATH or set {variable}.")


def fsl_bin(tool: str) -> Path:
    return _find_tool(tool, "FSLDIR", "bin")


def ants_bin(tool: str) -> Path:
    return _find_tool(tool, "ANTSPATH")


def afni_bin(tool: str) -> Path:
    return _find_tool(tool, "AFNI_DIR")


def run(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command, log it, raise on failure."""
    cmd_str = " ".join(str(c) for c in cmd)
    log.info(f"    $ {cmd_str}")
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if result.stdout.strip():
        log.debug(result.stdout.strip())
    if result.returncode != 0:
        log.error(result.stderr.strip())
        if check:
            raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd_str}")
    return result


def exists(p) -> bool:
    return Path(p).exists()


def remove_if(p: Path, cleanup: bool):
    """Delete a file when --cleanup is active; log the removal."""
    if cleanup and Path(p).exists():
        Path(p).unlink()
        log.info(f"    [cleanup] removed {Path(p).name}")


def safe_input(src: Path, safe_dir: Path, name: str) -> Path:
    """Give FSL a stable input path when the source path contains whitespace."""
    src = src.resolve()
    if not any(char.isspace() for char in str(src)):
        return src
    suffix = ".nii.gz" if src.name.endswith(".nii.gz") else ".nii"
    link = safe_dir / f"{name}{suffix}"
    if link.is_symlink() and link.resolve() != src:
        link.unlink()
    if not link.is_symlink():
        link.symlink_to(src)
    return link


def makedirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# ALLEN ONTOLOGY HELPERS  (ROI hierarchy merging)
# ─────────────────────────────────────────────────────────────────────────────

def ensure_allen_ontology():
    """Download Allen CCF structure graph JSON once and cache it locally."""
    if ALLEN_ONTOLOGY_JSON.exists():
        return
    url = "https://api.brain-map.org/api/v2/structure_graph_download/1.json"
    log.info(f"  Downloading Allen structure graph → {ALLEN_ONTOLOGY_JSON.name}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            ALLEN_ONTOLOGY_JSON.parent.mkdir(parents=True, exist_ok=True)
            data = json.load(resp)
            with TemporaryDirectory(dir=ALLEN_ONTOLOGY_JSON.parent) as tmp:
                staged = Path(tmp) / ALLEN_ONTOLOGY_JSON.name
                staged.write_text(json.dumps(data))
                staged.replace(ALLEN_ONTOLOGY_JSON)
        log.info("  Allen ontology downloaded.")
    except Exception as e:
        raise RuntimeError(
            f"Could not download Allen ontology from {url}: {e}\n"
            "Check internet access or manually place the JSON at:\n"
            f"  {ALLEN_ONTOLOGY_JSON}"
        )


def _traverse_ontology(node, target_depth, anc_map, name_map, depth=0, ancestor=None):
    nid  = node["id"]
    name_map[nid] = node.get("acronym", str(nid))
    if depth == target_depth:
        ancestor = nid
    anc_map[nid] = ancestor if ancestor is not None else nid
    for child in node.get("children", []):
        _traverse_ontology(child, target_depth, anc_map, name_map, depth + 1, ancestor)


def build_roi_maps(depth: int):
    """
    Parse the cached Allen ontology JSON and return:
      ancestor_map : {structure_id → ancestor_id at target_depth}
      name_map     : {structure_id → acronym string}
    """
    data = json.loads(ALLEN_ONTOLOGY_JSON.read_text())
    root = data["msg"][0]
    ancestor_map, name_map = {}, {}
    _traverse_ontology(root, depth, ancestor_map, name_map)
    return ancestor_map, name_map


def subj_label(raw_id: str) -> str:
    """'CNO-LPS1' → 'sub-CNOLPS1'"""
    return "sub-" + raw_id.replace("-", "")


# ─────────────────────────────────────────────────────────────────────────────
# DATA COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def collect_subjects() -> dict:
    """
    Walk DICOM_DIR and collect fMRI NIfTI files and T2 anatomicals for each
    subject. Handles both nested (ScanTypeDir/) and flat directory layouts.

    Returns
    -------
    dict : { raw_id → {'anat': Path, 'func': [Path, ...]} }
    """
    subjects = {}
    for subj_dir in sorted(DICOM_DIR.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name.startswith("."):
            continue
        raw_id = subj_dir.name

        all_nii = sorted(subj_dir.rglob("*.nii.gz"))

        # fMRI: T2* EPI runs (exclude ghost / adjustment acquisitions)
        func_files = [
            f for f in all_nii
            if "T2star_FID_EPI" in f.name and "ADJ_GHOST" not in f.name
            and "B0Map" not in f.name
        ]
        # T2 anatomical (3-D isotropic)
        anat_files = [f for f in all_nii if "T2_RAREvfl_3D_iso" in f.name]

        if not func_files:
            log.warning(f"[{raw_id}] No fMRI NIfTI found – skipping.")
            continue
        if not anat_files:
            log.warning(f"[{raw_id}] No T2 anatomical found – skipping.")
            continue

        # Last T2 = repeated/averaged acquisition (usually best quality)
        # Default: use only the first fMRI run
        subjects[raw_id] = {"anat": anat_files[-1], "func": func_files[:1]}
        log.info(f"  {raw_id}: {len(func_files)} run(s) found, using first | T2={anat_files[-1].name}")

    return subjects


def read_manifest(path: Path) -> dict:
    subjects = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"subject", "anat", "func"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path}: missing TSV columns {sorted(required)}")
        for line, row in enumerate(reader, start=2):
            subject, anat, func = ((row[key] or "").strip() for key in ("subject", "anat", "func"))
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", subject) or not anat or not func:
                raise ValueError(f"{path}:{line}: invalid subject ID or missing input path")
            anat, func = [(path.parent / Path(item).expanduser()).resolve() for item in (anat, func)]
            info = subjects.setdefault(subject, {"anat": anat, "func": []})
            if info["anat"] != anat or func in info["func"]:
                raise ValueError(f"{path}:{line}: conflicting anatomy or duplicate functional run")
            info["func"].append(func)
    return subjects


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – TRIM VOLUMES
# ─────────────────────────────────────────────────────────────────────────────

def trim_volumes(in_file: Path, out_file: Path, skip: bool = False):
    """
    Remove the first N_TRIM volumes (steady-state settling + stimulus artifact).
    Command: fslroi <in> <out> <start_vol> <n_vols>
    """
    if skip and exists(out_file):
        log.info("      [skip] Trimmed file exists.")
        return
    n_vols = nib.load(in_file).shape[3]
    run([fsl_bin("fslroi"), in_file, out_file, N_TRIM, n_vols - N_TRIM])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – MOTION CORRECTION
# ─────────────────────────────────────────────────────────────────────────────

def motion_correction(in_file: Path, out_prefix: Path, skip: bool = False):
    """
    mcflirt: rigid-body motion correction, saves .par (6 params) and RMS files.
    Reference volume = middle volume of the trimmed run.

    Output files:
      <out_prefix>.nii.gz  – motion-corrected 4-D image
      <out_prefix>.par     – motion parameters  (N_vols × 6: rx ry rz tx ty tz)
      <out_prefix>_abs.rms – absolute RMS displacement
      <out_prefix>_rel.rms – relative RMS displacement
    """
    mc_nii = Path(str(out_prefix) + ".nii.gz")
    if skip and exists(mc_nii) and exists(f"{out_prefix}.par"):
        log.info("      [skip] MC output exists.")
        return
    n_vols  = nib.load(in_file).shape[3]
    ref_vol = n_vols // 2
    run([
        fsl_bin("mcflirt"),
        "-in",   in_file,
        "-o",    out_prefix,
        "-plots",           # write .par
        "-rmsrel",          # relative RMS
        "-rmsabs",          # absolute RMS
        "-spline_final",    # spline interpolation
        "-refvol", ref_vol,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# STEP A – T2 SKULL STRIPPING  (runs once per subject, before EPI steps)
# ─────────────────────────────────────────────────────────────────────────────

def skull_strip_t2(t2_raw: Path, t2_brain: Path, skip: bool = False):
    """
    Skull-strip the T2 anatomical with FSL BET.

    Output files:
      <t2_brain>        – skull-stripped T2
      <t2_brain>_mask   – binary brain mask
    """
    if skip and exists(t2_brain):
        log.info("      [skip] T2 brain exists.")
        return
    run([
        fsl_bin("bet"), t2_raw, t2_brain,
        "-f", BET_F_ANAT,
        "-m",    # save binary mask
        "-R",    # robust brain-centre estimation
    ])


# ─────────────────────────────────────────────────────────────────────────────
# STEP B – T2 → ALLEN CCF REGISTRATION  (runs once per subject)
# ─────────────────────────────────────────────────────────────────────────────

def register_t2_to_allen(t2_brain: Path, out_prefix: Path, skip: bool = False):
    """
    ANTs Rigid + SyN registration: T2 brain (moving) → Allen CCF template (fixed).

    Two stages:
      Stage 1 – Rigid:  linear alignment with Mattes MI
      Stage 2 – SyN:    deformable warp with cross-correlation (CC)

    Output files:
      <out_prefix>0GenericAffine.mat   – rigid transform
      <out_prefix>1Warp.nii.gz        – SyN forward warp field
      <out_prefix>1InverseWarp.nii.gz – SyN inverse warp field
      <out_prefix>Warped.nii.gz       – T2 in Allen space (QC)
    """
    affine = Path(f"{out_prefix}0GenericAffine.mat")
    if skip and all(exists(file) for file in (affine, Path(f"{out_prefix}Warped.nii.gz"),
                                                Path(f"{out_prefix}1Warp.nii.gz"))):
        log.info("      [skip] T2→Allen registration exists.")
        return
    run([
        ants_bin("antsRegistration"),
        "--dimensionality", 3,
        "--float", 1,
        "--output",                      f"[{out_prefix},{out_prefix}Warped.nii.gz]",
        "--interpolation",               "Linear",
        "--winsorize-image-intensities", "[0.005,0.995]",
        "--use-histogram-matching",      0,
        "--initial-moving-transform",    f"[{ALLEN_TEMPLATE},{t2_brain},1]",
        # Stage 1: Rigid
        "--transform",   "Rigid[0.1]",
        "--metric",
            f"MI[{ALLEN_TEMPLATE},{t2_brain},"
            f"{ANTS_METRIC_WEIGHT},{ANTS_BINS},"
            f"{ANTS_SAMPLING},{ANTS_SAMPLING_PCT}]",
        "--convergence",      f"[{ANTS_CONVERGENCE},1e-6,10]",
        "--shrink-factors",   ANTS_SHRINK,
        "--smoothing-sigmas", ANTS_SMOOTH,
        # Stage 2: SyN
        "--transform",   "SyN[0.1,3,0]",
        "--metric",      f"CC[{ALLEN_TEMPLATE},{t2_brain},1,4]",
        "--convergence",      f"[{ANTS_SYN_CONVERGENCE},1e-6,10]",
        "--shrink-factors",   ANTS_SYN_SHRINK,
        "--smoothing-sigmas", ANTS_SYN_SMOOTH,
        "-v", 1,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – MEAN EPI + BRAIN EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def mean_and_bet(mc_file: Path, mean_file: Path, brain_file: Path, skip: bool = False):
    """
    Compute temporal mean of motion-corrected EPI, then skull-strip it.
    The brain-extracted mean is used as the moving image for ANTs registration.

    Output files:
      <mean_file>          – temporal mean (no skull strip)
      <brain_file>         – skull-stripped mean EPI
      <brain_file>_mask    – binary brain mask (from -m flag)
    """
    if skip and exists(brain_file):
        log.info("      [skip] Mean brain exists.")
        return
    run([fsl_bin("fslmaths"), mc_file, "-Tmean", mean_file])
    run([
        fsl_bin("bet"), mean_file, brain_file,
        "-f", BET_F_FUNC,
        "-m",    # save binary mask  → <brain_file>_mask.nii.gz
        "-R",    # robust brain-centre estimation
    ])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – EPI → T2 RIGID REGISTRATION  (ANTs)
# ─────────────────────────────────────────────────────────────────────────────

def register_epi_to_t2(
    mean_brain: Path,
    t2_brain:   Path,
    out_prefix: Path,
    skip: bool = False,
):
    """
    ANTs rigid registration: mean EPI (moving) → T2 brain (fixed).
    Uses Mattes mutual information (robust for EPI↔T2 contrast mismatch).

    Output files:
      <out_prefix>0GenericAffine.mat   – rigid transform (EPI → T2)
      <out_prefix>Warped.nii.gz        – mean EPI warped into T2 space (QC)

    Exact command:
      antsRegistration --dimensionality 3 --float 1
        --output [<out_prefix>, <out_prefix>Warped.nii.gz]
        --interpolation Linear
        --winsorize-image-intensities [0.005,0.995]
        --use-histogram-matching 0
        --initial-moving-transform [T2_brain, mean_brain, 1]
        --transform Rigid[0.1]
        --metric MI[T2_brain, mean_brain, 1, 32, Regular, 0.25]
        --convergence [1000x500x250x100, 1e-6, 10]
        --shrink-factors 8x4x2x1
        --smoothing-sigmas 3x2x1x0vox
    """
    affine = Path(f"{out_prefix}0GenericAffine.mat")
    if skip and exists(affine) and exists(f"{out_prefix}Warped.nii.gz"):
        log.info("      [skip] EPI→T2 registration exists.")
        return
    run([
        ants_bin("antsRegistration"),
        "--dimensionality", 3,
        "--float", 1,
        "--output",                f"[{out_prefix},{out_prefix}Warped.nii.gz]",
        "--interpolation",         "Linear",
        "--winsorize-image-intensities", "[0.005,0.995]",
        "--use-histogram-matching", 0,
        "--initial-moving-transform",
            f"[{t2_brain},{mean_brain},1]",     # centre-of-mass init
        "--transform",   "Rigid[0.1]",
        "--metric",
            f"MI[{t2_brain},{mean_brain},"
            f"{ANTS_METRIC_WEIGHT},{ANTS_BINS},"
            f"{ANTS_SAMPLING},{ANTS_SAMPLING_PCT}]",
        "--convergence", f"[{ANTS_CONVERGENCE},1e-6,10]",
        "--shrink-factors",  ANTS_SHRINK,
        "--smoothing-sigmas", ANTS_SMOOTH,
        "-v", 1,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – APPLY EPI→T2→ALLEN TRANSFORMS  (ANTs 4-D)
# ─────────────────────────────────────────────────────────────────────────────

def warp_epi_to_allen(
    mc_file:        Path,
    func2t2_affine: Path,
    t2_to_allen_affine: Path,
    t2_to_allen_warp:   Path,
    out_file:       Path,
    skip: bool = False,
):
    """
    Apply the concatenated transform chain  EPI → T2 → Allen CCF
    to the full 4-D motion-corrected EPI timeseries.

    Transform order for ANTs (applied right-to-left, i.e. last specified first):
      1. func2t2_affine              (EPI → T2,    applied first to moving vol)
      2. t2_to_allen_affine          (T2  → Allen, affine part)
      3. t2_to_allen_warp            (T2  → Allen, non-linear warp, applied last)

    Exact command:
      antsApplyTransforms
        --dimensionality 3
        --input-image-type 3          ← time series flag
        --input mc_func.nii.gz
        --reference-image allen_template.nii.gz
        --output func_allen.nii.gz
        --transform t2_to_allen_1Warp.nii.gz
        --transform t2_to_allen_0GenericAffine.mat
        --transform [func2anat_0GenericAffine.mat, 0]
        --interpolation LanczosWindowedSinc
    """
    if skip and exists(out_file):
        log.info("      [skip] Allen-space EPI exists.")
        return
    run([
        ants_bin("antsApplyTransforms"),
        "--dimensionality", 3,
        "--input-image-type", 3,               # 4-D time series
        "--input",            mc_file,
        "--reference-image",  ALLEN_TEMPLATE_FUNC,   # 100µm: feasible output size
        "--output",           out_file,
        # Transforms applied right-to-left (last listed = applied first)
        "--transform", t2_to_allen_warp,
        "--transform", t2_to_allen_affine,
        "--transform", f"[{func2t2_affine},0]",  # 0 = forward direction
        "--interpolation", "LanczosWindowedSinc",
        "-v", 1,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 – NUISANCE REGRESSION + BANDPASS  (AFNI 3dTproject)
# ─────────────────────────────────────────────────────────────────────────────

def make_nuisance_ort(motion_par: Path, ort_file: Path, skip: bool = False):
    """
    Build AFNI 1D ort file:  6 motion params  +  6 temporal derivatives
    (columns = regressors, rows = time points).

    Output: <ort_file>  (plain text, whitespace-separated)
    """
    if skip and exists(ort_file) and Path(ort_file).stat().st_size > 0:
        return
    motion = np.loadtxt(motion_par)                        # (T, 6)
    diff   = np.vstack([np.zeros((1, 6)), np.diff(motion, axis=0)])  # (T, 6)
    ort    = np.hstack([motion, diff])                     # (T, 12)
    np.savetxt(ort_file, ort, fmt="%.8f")


def nuisance_and_bandpass(
    allen_file: Path,
    ort_file:   Path,
    out_file:   Path,
    tr: float,
    skip: bool = False,
):
    """
    AFNI 3dTproject: simultaneous nuisance regression and bandpass filtering.
    Avoids the spectral re-introduction artifacts of sequential filtering.

    Regressors: motion (6) + derivatives (6) supplied via -ort.
    Trend:      -polort 2  removes up to quadratic baseline drift.
    Bandpass:   -passband 0.01 0.1 Hz.
    Mask:       Allen brain mask (keeps only brain voxels).

    Exact command:
      3dTproject
        -input  func_allen.nii.gz
        -prefix func_cleaned.nii.gz
        -ort    motion_ort.1D
        -passband 0.01 0.1
        -mask   allen_brain_mask.nii.gz
        -polort 2
        -verb
    """
    if skip and exists(out_file):
        log.info("      [skip] Cleaned EPI exists.")
        return
    run([
        afni_bin("3dTproject"),
        "-input",    allen_file,
        "-prefix",   out_file,
        "-ort",      ort_file,
        "-passband", HP_FREQ, LP_FREQ,
        "-dt",       tr,
        "-mask",     ALLEN_BRAIN_MASK_FUNC,
        "-polort",   2,
        "-verb",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 – ROI MEAN TIME-SERIES  (AFNI 3dROIstats)
# ─────────────────────────────────────────────────────────────────────────────

def extract_roi_timeseries(
    cleaned_file: Path,
    roi_ts_txt:   Path,
    roi_ts_npy:   Path,
    roi_ids_npy:  Path,
    skip: bool = False,
):
    """Measure each in-mask ROI using consecutive short labels for AFNI."""
    if skip and all(exists(p) for p in (roi_ts_txt, roi_ts_npy, roi_ids_npy)):
        log.info("      [skip] ROI timeseries exists.")
        return

    atlas = nib.load(ALLEN_LABELS_FUNC)
    mask = nib.load(ALLEN_BRAIN_MASK_FUNC)
    cleaned = nib.load(cleaned_file)
    if (atlas.shape != cleaned.shape[:3]
            or not np.allclose(atlas.affine, cleaned.affine, atol=1e-4)):
        raise ValueError("Cleaned EPI and the functional atlas must share a voxel grid.")
    labels = np.asanyarray(atlas.dataobj)
    in_roi = (np.asanyarray(mask.dataobj) > 0) & (labels > 0)
    roi_ids, indices = np.unique(labels[in_roi], return_inverse=True)
    if not 0 < roi_ids.size <= np.iinfo(np.int16).max:
        raise ValueError(f"Invalid number of in-mask ROIs: {roi_ids.size}")
    short_labels = np.zeros(atlas.shape, dtype=np.int16)
    short_labels[in_roi] = indices + 1

    with TemporaryDirectory(dir=roi_ts_txt.parent) as tmp:
        roi_mask = Path(tmp) / "roi_mask.nii.gz"
        nib.save(nib.Nifti1Image(short_labels, atlas.affine), roi_mask)
        result = run([afni_bin("3dROIstats"), "-quiet", "-mask", roi_mask, cleaned_file])
    ts_matrix = np.loadtxt(result.stdout.splitlines(), ndmin=2, dtype=np.float32)
    if ts_matrix.shape != (cleaned.shape[3], roi_ids.size):
        raise ValueError(f"Unexpected 3dROIstats output shape: {ts_matrix.shape}")
    roi_ts_txt.write_text(result.stdout)
    np.save(roi_ts_npy, ts_matrix)
    np.save(roi_ids_npy, roi_ids)
    log.info("      ROI timeseries shape: %s", ts_matrix.shape)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7b – MERGE ROIs BY ALLEN HIERARCHY
# ─────────────────────────────────────────────────────────────────────────────

def merge_roi_timeseries(
    roi_ts_npy:     Path,
    roi_ids_npy:    Path,
    merged_ts_npy:  Path,
    merged_ids_tsv: Path,
    skip: bool = False,
):
    """
    Collapse leaf-level ROI timeseries to summary regions by averaging all
    ROIs that share the same Allen CCF ancestor at ROI_MERGE_DEPTH.

    Output files:
      <merged_ts_npy>   – (T, M) float32 array  (M << N)
      <merged_ids_tsv>  – TSV: ancestor_id | acronym | n_rois_merged

    Returns (merged_ts, ancestor_ids, acronyms).
    """
    if skip and exists(merged_ts_npy) and exists(merged_ids_tsv):
        log.info("      [skip] Merged ROI timeseries exists.")
        df = pd.read_csv(merged_ids_tsv, sep="\t")
        return np.load(merged_ts_npy), df["ancestor_id"].tolist(), df["acronym"].tolist()

    ts      = np.load(roi_ts_npy).astype(np.float64)   # (T, N)
    roi_ids = np.load(roi_ids_npy).tolist()             # N ints

    ancestor_map, name_map = build_roi_maps(ROI_MERGE_DEPTH)

    ancestors        = [ancestor_map.get(rid, rid) for rid in roi_ids]
    unique_ancestors = sorted(set(ancestors))

    T      = ts.shape[0]
    merged = np.zeros((T, len(unique_ancestors)), dtype=np.float32)
    counts = []
    for col, anc in enumerate(unique_ancestors):
        cols = [i for i, a in enumerate(ancestors) if a == anc]
        merged[:, col] = ts[:, cols].mean(axis=1)
        counts.append(len(cols))

    acronyms = [name_map.get(a, str(a)) for a in unique_ancestors]
    np.save(merged_ts_npy, merged)
    pd.DataFrame({
        "ancestor_id":   unique_ancestors,
        "acronym":       acronyms,
        "n_rois_merged": counts,
    }).to_csv(merged_ids_tsv, sep="\t", index=False)

    log.info(
        f"      ROI merge depth={ROI_MERGE_DEPTH} "
        f"{len(roi_ids)} leaf ROIs → {len(unique_ancestors)} summary regions"
    )
    return merged, unique_ancestors, acronyms


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 – CONNECTIVITY MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def compute_fc_matrix(
    roi_ts_npy:  Path,
    fc_npy:      Path,
    fc_tsv:      Path,
    roi_labels:  list = None,
    skip: bool = False,
):
    """
    Compute ROI-by-ROI Pearson correlation matrix and apply Fisher z-transform.

    Steps:
      1. Load (T, N_rois) time-series matrix.
      2. Compute N_rois × N_rois Pearson r using numpy (vectorised).
      3. Fisher z-transform: z = arctanh(r)  [clips |r| < 0.9999 to avoid inf].
      4. Save:
           <fc_npy>  – numpy array (N_rois, N_rois) Fisher z values
           <fc_tsv>  – pandas DataFrame with ROI label indices as row/column names

    Expected output files:
      sub-CTL1_run-01_fc_matrix_fisherz.npy
      sub-CTL1_run-01_fc_matrix_fisherz.tsv
    """
    if skip and exists(fc_npy) and exists(fc_tsv):
        log.info("      [skip] FC matrix exists.")
        return

    ts = np.load(roi_ts_npy).astype(np.float64)   # (T, N_rois)
    T, N = ts.shape
    log.info(f"      Computing {N}×{N} FC matrix from {T} time points")

    # Vectorised Pearson r:  correlate each pair of z-scored signals
    if T < 2 or not np.isfinite(ts).all():
        raise ValueError("FC requires at least two time points with finite signals.")
    centered = ts - ts.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    ts_z = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)
    r_matrix = ts_z.T @ ts_z
    np.fill_diagonal(r_matrix, 1.0)
    r_matrix = np.clip(r_matrix, -0.9999, 0.9999)

    # Fisher z-transform
    z_matrix = np.arctanh(r_matrix)

    # Use provided labels or fall back to atlas-derived IDs
    if roi_labels is not None and len(roi_labels) == N:
        row_col_ids = roi_labels
    else:
        label_img = nib.load(ALLEN_LABELS_FUNC)
        roi_ids   = sorted(np.unique(label_img.get_fdata().astype(int)))
        roi_ids   = [r for r in roi_ids if r != 0]
        if len(roi_ids) != N:
            log.warning(
                f"      ROI count mismatch: atlas has {len(roi_ids)} labels "
                f"but timeseries has {N} columns. Using numeric indices."
            )
            roi_ids = list(range(N))
        row_col_ids = roi_ids

    np.save(fc_npy, z_matrix)
    df = pd.DataFrame(z_matrix, index=row_col_ids, columns=row_col_ids)
    df.to_csv(fc_tsv, sep="\t", float_format="%.6f")
    log.info(f"      FC matrix saved → {fc_npy.name}  {fc_tsv.name}")


# ─────────────────────────────────────────────────────────────────────────────
# QC – THREE ESSENTIAL CHECKS
# ─────────────────────────────────────────────────────────────────────────────

# ── QC-A: Framewise Displacement ─────────────────────────────────────────────

def qc_motion(motion_par: Path, qc_dir: Path, run_label: str) -> float:
    """
    QC-A: Framewise Displacement (Power et al. 2012).
    Sum absolute translations and convert absolute rotations to millimeters
    using the configured radius.

    Saves:
      <qc_dir>/<run_label>_FD.png  – FD time series with threshold line
      <qc_dir>/<run_label>_FD.txt  – mean FD and percent volumes above threshold

    Returns mean FD (mm).
    """
    makedirs(qc_dir)
    params = np.loadtxt(motion_par)                    # (T, 6): rx ry rz tx ty tz
    diff   = np.abs(np.diff(params, axis=0))
    fd = diff[:, :3].sum(axis=1) * FD_RADIUS + diff[:, 3:].sum(axis=1)
    mean_fd    = float(np.mean(fd))
    pct_spikes = float(100 * np.mean(fd > FD_THRESHOLD))

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(fd, color="steelblue", linewidth=0.8, label="FD (mm)")
    ax.axhline(FD_THRESHOLD, color="red", linewidth=1, linestyle="--",
               label=f"Threshold {FD_THRESHOLD} mm")
    ax.set(xlabel="Volume", ylabel="FD (mm)",
           title=f"{run_label} | Mean FD={mean_fd:.3f} mm | "
                 f">{FD_THRESHOLD} mm: {pct_spikes:.1f}%")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(qc_dir / f"{run_label}_FD.png", dpi=150)
    plt.close(fig)

    summary = (f"mean_FD_mm={mean_fd:.4f}\n"
               f"pct_volumes_above_thr={pct_spikes:.2f}\n"
               f"n_volumes={len(fd)}\n"
               f"threshold_mm={FD_THRESHOLD}\n"
               f"radius_mm={FD_RADIUS}\n")
    (qc_dir / f"{run_label}_FD.txt").write_text(summary)

    flag = "PASS" if mean_fd < FD_THRESHOLD else "WARN"
    log.info(f"      QC-A [{flag}] FD: mean={mean_fd:.3f} mm, "
             f"spikes={pct_spikes:.1f}%")
    return mean_fd


# ── QC-B: Registration Quality (NCC) ─────────────────────────────────────────

def qc_registration(
    mean_brain_in_t2: Path,
    t2_brain:         Path,
    qc_dir:           Path,
    run_label:        str,
) -> float:
    """
    QC-B: Normalized Cross-Correlation between mean EPI warped into T2 space
    and the T2 brain.  NCC ∈ [-1,1]; values > 0.70 indicate good alignment.

    Saves:
      <qc_dir>/<run_label>_reg_NCC.txt  – NCC value and pass/fail flag
      <qc_dir>/<run_label>_reg_edge.png – edge overlay (middle slice)

    Returns NCC value.
    """
    makedirs(qc_dir)
    if not exists(mean_brain_in_t2):
        log.warning("      QC-B skipped: warped mean EPI not found.")
        return float("nan")

    epi_data = nib.load(mean_brain_in_t2).get_fdata().flatten()
    t2_data  = nib.load(t2_brain).get_fdata().flatten()

    # Restrict to voxels where both images are non-zero
    mask = (epi_data > 0) & (t2_data > 0)
    if mask.sum() < 100:
        log.warning("      QC-B: too few overlapping voxels – check registration.")
        return float("nan")

    epi_m = epi_data[mask]
    t2_m  = t2_data[mask]
    ncc   = float(np.corrcoef(epi_m, t2_m)[0, 1])
    flag  = "PASS" if ncc > NCC_THRESHOLD else "WARN"
    log.info(f"      QC-B [{flag}] Registration NCC = {ncc:.4f}")

    # Edge overlay on the middle axial slice
    epi_vol = nib.load(mean_brain_in_t2).get_fdata()
    t2_vol  = nib.load(t2_brain).get_fdata()
    # Match in-plane: take the slice with the largest T2 brain area
    z_idx   = int(np.argmax((t2_vol > 0).sum(axis=(0, 1))))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(t2_vol[:, :, z_idx].T,  cmap="gray", origin="lower")
    axes[0].set_title("T2 brain")
    axes[1].imshow(t2_vol[:, :, z_idx].T,  cmap="gray", origin="lower")
    axes[1].imshow(epi_vol[:, :, z_idx].T, cmap="hot",  origin="lower", alpha=0.4)
    axes[1].set_title(f"T2 + EPI overlay | NCC={ncc:.3f}")
    for ax in axes:
        ax.axis("off")
    plt.suptitle(run_label)
    plt.tight_layout()
    fig.savefig(qc_dir / f"{run_label}_reg_edge.png", dpi=150)
    plt.close(fig)

    (qc_dir / f"{run_label}_reg_NCC.txt").write_text(
        f"NCC={ncc:.6f}\nflag={flag}\nthreshold={NCC_THRESHOLD}\n"
    )
    return ncc


# ── QC-C: Connectivity Matrix Sanity Check ────────────────────────────────────

def qc_connectivity(
    fc_npy:    Path,
    qc_dir:    Path,
    subj_label_: str,
    run_label: str,
):
    """
    QC-C: Three sanity checks on the FC matrix.
      i.  Distribution: Fisher-z values should be roughly symmetric, most in [-3, 3].
      ii. No NaN / Inf values.
      iii. Report the upper-triangle mean Fisher z as a descriptive QC metric.

    Saves:
      <qc_dir>/<subj>_<run>_FC_heatmap.png
      <qc_dir>/<subj>_<run>_FC_distribution.png
      <qc_dir>/<subj>_<run>_FC_sanity.txt
    """
    makedirs(qc_dir)
    z = np.load(fc_npy)
    N = z.shape[0]

    # i. Finite values
    n_nan = int(np.sum(~np.isfinite(z)))
    flag_nan = "PASS" if n_nan == 0 else "FAIL"

    # ii. Off-diagonal upper-triangle stats
    idx = np.triu_indices(N, k=1)
    z_upper = z[idx]
    finite_z = z_upper[np.isfinite(z_upper)]
    mean_z = float(finite_z.mean()) if finite_z.size else float("nan")
    flag_sign = "PASS" if mean_z > 0 else "WARN"

    log.info(f"      QC-C [{flag_nan}] NaN/Inf count: {n_nan}")
    log.info(f"      QC-C [{flag_sign}] Mean off-diag Fisher z: {mean_z:.3f}")

    prefix = f"{subj_label_}_{run_label}"

    # FC heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(z, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    plt.colorbar(im, ax=ax, label="Fisher z")
    ax.set_title(f"{prefix} | FC matrix ({N}×{N})")
    ax.set_xlabel("ROI index")
    ax.set_ylabel("ROI index")
    plt.tight_layout()
    fig.savefig(qc_dir / f"{prefix}_FC_heatmap.png", dpi=150)
    plt.close(fig)

    # Distribution histogram
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(finite_z, bins=80, color="steelblue", edgecolor="none")
    ax.axvline(0, color="black", linewidth=1)
    ax.set(xlabel="Fisher z", ylabel="Count",
           title=f"{prefix} | FC distribution | mean={mean_z:.3f}")
    plt.tight_layout()
    fig.savefig(qc_dir / f"{prefix}_FC_distribution.png", dpi=150)
    plt.close(fig)

    summary = (
        f"n_rois={N}\n"
        f"n_nan_inf={n_nan}\n"
        f"mean_offdiag_fisherz={mean_z:.6f}\n"
        f"flag_nan={flag_nan}\n"
        f"flag_sign={flag_sign}\n"
    )
    (qc_dir / f"{prefix}_FC_sanity.txt").write_text(summary)


# ─────────────────────────────────────────────────────────────────────────────
# DIRECTORY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _anat_dir(raw_id: str) -> Path:
    return DERIV_DIR / subj_label(raw_id) / "anat"


def _func_dir(raw_id: str, run_idx: int) -> Path:
    return DERIV_DIR / subj_label(raw_id) / "func" / f"run-{run_idx:02d}"


def _conn_dir(raw_id: str) -> Path:
    return DERIV_DIR / subj_label(raw_id) / "connectivity"


# ─────────────────────────────────────────────────────────────────────────────
# PER-SUBJECT PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_subject(raw_id: str, info: dict, args):
    lbl     = subj_label(raw_id)
    skip    = args.skip_existing
    cleanup = args.cleanup
    log.info(f"\n{'='*60}")
    log.info(f"  Subject: {raw_id}  ({lbl})")
    log.info(f"{'='*60}")

    anat_dir = _anat_dir(raw_id)
    conn_dir = _conn_dir(raw_id)
    subj_qc  = QC_DIR / lbl
    makedirs(anat_dir, conn_dir, subj_qc)

    t2_raw           = safe_input(info["anat"], anat_dir, "input_anat")
    t2_brain         = anat_dir / f"{lbl}_T2w_brain.nii.gz"
    t2_to_allen_pfx  = anat_dir / f"{lbl}_T2w_to_allen_"
    t2_to_allen_aff  = anat_dir / f"{lbl}_T2w_to_allen_0GenericAffine.mat"
    t2_to_allen_warp = anat_dir / f"{lbl}_T2w_to_allen_1Warp.nii.gz"

    # ── Step A: T2 skull stripping ────────────────────────────────────────
    log.info("  [A] T2 skull stripping (BET)")
    skull_strip_t2(t2_raw, t2_brain, skip=skip)

    # ── Step B: T2 → Allen CCF registration ──────────────────────────────
    log.info("  [B] T2 → Allen CCF registration (ANTs Rigid+SyN)")
    register_t2_to_allen(t2_brain, t2_to_allen_pfx, skip=skip)

    for run_idx, func_raw_orig in enumerate(info["func"], start=1):
        fdir = _func_dir(raw_id, run_idx)
        makedirs(fdir)
        func_raw = safe_input(func_raw_orig, fdir, "input_func")
        log.info(f"\n  ── Run {run_idx:02d}: {func_raw_orig.name}")

        p     = fdir / f"{lbl}_run-{run_idx:02d}"
        rl    = f"{lbl}_run-{run_idx:02d}"   # run label for QC naming

        # Expected file names at each stage
        trim_file        = Path(f"{p}_bold_trim.nii.gz")
        mc_prefix        = Path(f"{p}_bold_mc")
        mc_file          = Path(f"{p}_bold_mc.nii.gz")
        mc_par           = Path(f"{p}_bold_mc.par")
        mean_file        = Path(f"{p}_bold_mean.nii.gz")
        mean_brain       = Path(f"{p}_bold_mean_brain.nii.gz")
        func2t2_prefix   = Path(f"{p}_func2anat_")
        func2t2_affine   = Path(f"{p}_func2anat_0GenericAffine.mat")
        mean_in_t2       = Path(f"{p}_func2anat_Warped.nii.gz")    # ANTs output
        allen_file       = Path(f"{p}_bold_allen.nii.gz")
        ort_file         = Path(f"{p}_motion_ort.1D")
        cleaned_file     = Path(f"{p}_bold_cleaned.nii.gz")
        roi_ts_txt       = conn_dir / f"{rl}_roi_timeseries.txt"
        roi_ts_npy       = conn_dir / f"{rl}_roi_timeseries.npy"
        roi_ids_npy      = conn_dir / f"{rl}_roi_ids.npy"
        merged_ts_npy    = conn_dir / f"{rl}_roi_timeseries_merged.npy"
        merged_ids_tsv   = conn_dir / f"{rl}_roi_ids_merged.tsv"
        fc_npy           = conn_dir / f"{rl}_fc_matrix_fisherz.npy"
        fc_tsv           = conn_dir / f"{rl}_fc_matrix_fisherz.tsv"

        # ── Step 1: Trim ─────────────────────────────────────────────────
        log.info(f"  [1] Trim first {N_TRIM} volumes")
        trim_volumes(func_raw, trim_file, skip=skip)

        # ── Step 2: Motion correction ─────────────────────────────────────
        log.info("  [2] Motion correction (MCFLIRT)")
        motion_correction(trim_file, mc_prefix, skip=skip)
        remove_if(trim_file, cleanup)          # no longer needed after MC

        # ── Step 3: Mean EPI + BET ─────────────────────────────────────────
        log.info("  [3] Mean EPI + brain extraction (BET)")
        mean_and_bet(mc_file, mean_file, mean_brain, skip=skip)

        # ── Step 4: ANTs EPI → T2 ─────────────────────────────────────────
        log.info("  [4] EPI → T2 registration (ANTs Rigid)")
        register_epi_to_t2(mean_brain, t2_brain, func2t2_prefix, skip=skip)

        # ── Step 5: Warp 4-D EPI to Allen space ───────────────────────────
        log.info("  [5] Warp EPI → Allen space (antsApplyTransforms)")
        warp_epi_to_allen(
            mc_file, func2t2_affine,
            t2_to_allen_aff, t2_to_allen_warp,
            allen_file, skip=skip,
        )

        # ── Step 6: Nuisance regression + bandpass ─────────────────────────
        log.info("  [6] Nuisance regression + bandpass (3dTproject)")
        make_nuisance_ort(mc_par, ort_file, skip=skip)
        nuisance_and_bandpass(allen_file, ort_file, cleaned_file,
                              tr=func_tr(nib.load(func_raw)), skip=skip)
        remove_if(mc_file, cleanup)            # no longer needed after warp+clean
        remove_if(allen_file, cleanup)         # no longer needed after cleaning

        roi_checkpoint = conn_dir / f"{rl}_roi_settings_v2.json"
        roi_settings = json.dumps({"merge_depth": ROI_MERGE_DEPTH})
        skip_roi = (skip and roi_checkpoint.exists()
                    and roi_checkpoint.read_text() == roi_settings)
        if not skip_roi:
            roi_checkpoint.unlink(missing_ok=True)

        # ── Step 7: ROI time-series ────────────────────────────────────────
        log.info("  [7] ROI mean time-series (3dROIstats)")
        extract_roi_timeseries(cleaned_file, roi_ts_txt, roi_ts_npy, roi_ids_npy, skip=skip_roi)
        remove_if(cleaned_file, cleanup)       # no longer needed after ROI extraction

        # ── Step 7b: Merge ROIs by Allen hierarchy ─────────────────────────
        log.info(f"  [7b] Merge ROIs by Allen hierarchy (depth={ROI_MERGE_DEPTH})")
        _, merged_roi_ids, merged_acronyms = merge_roi_timeseries(
            roi_ts_npy, roi_ids_npy, merged_ts_npy, merged_ids_tsv, skip=skip_roi)

        # ── Step 8: FC matrix (merged summary regions) ─────────────────────
        log.info("  [8] ROI–ROI FC matrix (merged, Pearson r + Fisher z)")
        compute_fc_matrix(merged_ts_npy, fc_npy, fc_tsv,
                          roi_labels=merged_acronyms, skip=skip_roi)
        roi_checkpoint.write_text(roi_settings)

        # ── QC ────────────────────────────────────────────────────────────
        log.info("  [QC] Generating quality-control metrics")
        qc_motion(mc_par, subj_qc, rl)
        qc_registration(mean_in_t2, t2_brain, subj_qc, rl)
        if exists(fc_npy):
            qc_connectivity(fc_npy, subj_qc, lbl, f"run-{run_idx:02d}")

    log.info(f"\n  [{raw_id}] Complete.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Mouse rs-fMRI FC pipeline")
    inputs = p.add_mutually_exclusive_group()
    inputs.add_argument("--manifest", type=Path, help="TSV with subject, anat and func columns.")
    inputs.add_argument("--input-dir", type=Path, default=DICOM_DIR,
                        help="Bruker NIfTI directory (default: NPIL_DICOM).")
    p.add_argument("--atlas-dir", type=Path, default=ATLAS_DIR)
    p.add_argument("--output-dir", type=Path, default=DERIV_DIR)
    p.add_argument("--qc-dir", type=Path, default=QC_DIR)
    p.add_argument("--log-file", type=Path, help="Default: preprocessing.log inside output-dir.")
    p.add_argument("--subjects", nargs="+", metavar="ID", help="Select these subject IDs.")
    p.add_argument("--tr", type=float, help="TR in seconds (default: read each NIfTI header).")
    p.add_argument("--trim", type=int, default=N_TRIM, help="Initial volumes to discard.")
    p.add_argument("--bandpass", nargs=2, type=float, default=(HP_FREQ, LP_FREQ),
                   metavar=("LOW", "HIGH"), help="Bandpass cutoffs in Hz.")
    p.add_argument("--roi-depth", type=int, default=ROI_MERGE_DEPTH)
    p.add_argument("--bet-f-anat", type=float, default=BET_F_ANAT)
    p.add_argument("--bet-f-func", type=float, default=BET_F_FUNC)
    p.add_argument("--fd-radius", type=float, default=FD_RADIUS, help="FD rotation radius in mm.")
    p.add_argument("--skip-existing", action="store_true", help="Resume with the same inputs and settings.")
    p.add_argument("--cleanup", action="store_true",
                   help="Remove large intermediate volumes, including cleaned EPI.")
    p.add_argument("--qc-only", action="store_true", help="Regenerate QC from existing outputs.")
    p.add_argument("--check-only", action="store_true", help="Validate inputs and tools, then exit.")
    args = p.parse_args()
    low, high = args.bandpass
    if not (math.isfinite(high) and 0 < low < high):
        p.error("--bandpass requires two increasing positive frequencies")
    if args.tr is not None and not (math.isfinite(args.tr) and 0 < args.tr < 0.5 / high):
        p.error("--tr must be positive and the high cutoff must be below Nyquist")
    if args.trim < 0 or args.roi_depth < 0:
        p.error("--trim and --roi-depth must be nonnegative")
    if not all(0 < f < 1 for f in (args.bet_f_anat, args.bet_f_func)):
        p.error("BET thresholds must be between 0 and 1")
    if not (math.isfinite(args.fd_radius) and args.fd_radius > 0):
        p.error("--fd-radius must be positive")
    if args.check_only and args.qc_only:
        p.error("--check-only and --qc-only are mutually exclusive")
    return args


def configure(args):
    global DICOM_DIR, DERIV_DIR, QC_DIR, TR, N_TRIM, HP_FREQ, LP_FREQ
    global ROI_MERGE_DEPTH, BET_F_ANAT, BET_F_FUNC, FD_RADIUS
    global ALLEN_TEMPLATE, ALLEN_TEMPLATE_FUNC, ALLEN_LABELS_FUNC
    global ALLEN_BRAIN_MASK_FUNC, ALLEN_ONTOLOGY_JSON

    DICOM_DIR = args.input_dir.expanduser().resolve()
    DERIV_DIR = args.output_dir.expanduser().resolve()
    QC_DIR = args.qc_dir.expanduser().resolve()
    atlas_dir = args.atlas_dir.expanduser().resolve()
    ALLEN_TEMPLATE = atlas_dir / ALLEN_TEMPLATE.name
    ALLEN_TEMPLATE_FUNC = atlas_dir / ALLEN_TEMPLATE_FUNC.name
    ALLEN_LABELS_FUNC = atlas_dir / ALLEN_LABELS_FUNC.name
    ALLEN_BRAIN_MASK_FUNC = atlas_dir / ALLEN_BRAIN_MASK_FUNC.name
    ALLEN_ONTOLOGY_JSON = atlas_dir / ALLEN_ONTOLOGY_JSON.name
    TR, N_TRIM = args.tr, args.trim
    HP_FREQ, LP_FREQ = args.bandpass
    ROI_MERGE_DEPTH = args.roi_depth
    BET_F_ANAT, BET_F_FUNC, FD_RADIUS = args.bet_f_anat, args.bet_f_func, args.fd_radius


def func_tr(image):
    if TR is not None:
        return TR
    _, unit = image.header.get_xyzt_units()
    seconds = {"sec": 1.0, "msec": 0.001, "usec": 0.000001}
    return image.header.get_zooms()[3] * seconds.get(unit, 0)


def check_inputs(subjects):
    if any(c.isspace() for c in str(DERIV_DIR)) or any(
            c in str(path) for path in (DERIV_DIR, ALLEN_TEMPLATE) for c in "[],"):
        raise ValueError("Use an output path without whitespace and atlas/output paths without brackets or commas.")
    template = nib.load(ALLEN_TEMPLATE_FUNC)
    labels = nib.load(ALLEN_LABELS_FUNC)
    mask = nib.load(ALLEN_BRAIN_MASK_FUNC)
    if len(nib.load(ALLEN_TEMPLATE).shape) != 3 or len(template.shape) != 3:
        raise ValueError("Allen templates must be three-dimensional.")
    for atlas_file in (labels, mask):
        if atlas_file.shape != template.shape or not np.allclose(atlas_file.affine, template.affine):
            raise ValueError("The functional template, labels and mask must share a voxel grid.")
    for subject, info in subjects.items():
        if len(nib.load(info["anat"]).shape) != 3:
            raise ValueError(f"{subject}: expected a 3-D anatomy: {info['anat']}")
        for func in info["func"]:
            image = nib.load(func)
            if len(image.shape) != 4 or image.shape[3] - N_TRIM < 2:
                raise ValueError(f"{subject}: expected a 4-D EPI with volumes remaining after trimming: {func}")
            if not (0 < func_tr(image) < 0.5 / LP_FREQ):
                raise ValueError(f"{subject}: invalid header TR for bandpass: {func}. Set --tr in seconds.")
    for tool in ("fslroi", "mcflirt", "fslmaths", "bet"):
        fsl_bin(tool)
    if not shutil.which("dc"):
        raise FileNotFoundError("FSL BET robust mode requires dc on PATH. Install the system dc package.")
    for tool in ("antsRegistration", "antsApplyTransforms"):
        ants_bin(tool)
    for tool in ("3dTproject", "3dROIstats"):
        afni_bin(tool)


def main():
    args = parse_args()
    configure(args)
    handlers = [logging.StreamHandler(sys.stdout)]
    if not args.check_only:
        log_file = args.log_file or DERIV_DIR / "preprocessing.log"
        log_file = log_file.expanduser().resolve()
        makedirs(log_file.parent)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                        handlers=handlers)
    os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    os.environ["AFNI_DECONFLICT"] = "OVERWRITE"

    try:
        subjects = read_manifest(args.manifest.expanduser().resolve()) if args.manifest else collect_subjects()
        if args.subjects:
            unknown = sorted(set(args.subjects) - subjects.keys())
            if unknown:
                raise ValueError(f"Unknown or incomplete subjects: {', '.join(unknown)}")
            subjects = {key: subjects[key] for key in dict.fromkeys(args.subjects)}
        if not subjects:
            raise ValueError("No subjects found.")
        labels = [subj_label(key) for key in subjects]
        if len(set(labels)) != len(labels):
            raise ValueError("Subject IDs collide after removing hyphens.")
        if not args.qc_only:
            check_inputs(subjects)
            if not args.check_only:
                ensure_allen_ontology()
    except (OSError, ValueError, RuntimeError, nib.filebasedimages.ImageFileError) as exc:
        log.error("%s", exc)
        return 1

    n_runs = sum(len(info["func"]) for info in subjects.values())
    log.info("%d subject(s), %d run(s)", len(subjects), n_runs)
    if args.check_only:
        log.info("Input headers, atlas grids and executable paths validated.")
        return 0

    if args.qc_only:
        generated = False
        for raw_id, info in subjects.items():
            lbl     = subj_label(raw_id)
            subj_qc = QC_DIR / lbl
            makedirs(subj_qc)
            for run_idx, _ in enumerate(info["func"], start=1):
                rl       = f"{lbl}_run-{run_idx:02d}"
                mc_par   = _func_dir(raw_id, run_idx) / f"{rl}_bold_mc.par"
                mean_t2  = _func_dir(raw_id, run_idx) / f"{rl}_func2anat_Warped.nii.gz"
                t2_brain = _anat_dir(raw_id) / f"{lbl}_T2w_brain.nii.gz"
                fc_npy   = _conn_dir(raw_id) / f"{rl}_fc_matrix_fisherz.npy"
                if exists(mc_par):
                    generated = True
                    qc_motion(mc_par, subj_qc, rl)
                if exists(mean_t2) and exists(t2_brain):
                    generated = True
                    qc_registration(mean_t2, t2_brain, subj_qc, rl)
                if exists(fc_npy):
                    generated = True
                    qc_connectivity(fc_npy, subj_qc, lbl, f"run-{run_idx:02d}")
        if not generated:
            log.error("No derivatives available for QC.")
            return 1
        log.info("QC-only mode complete.")
        return 0

    failed = []
    for raw_id, info in subjects.items():
        try:
            preprocess_subject(raw_id, info, args)
        except Exception as e:
            failed.append(raw_id)
            log.error(f"[{raw_id}] FAILED: {e}", exc_info=True)

    if failed:
        log.error("Processing failed for: %s", ", ".join(failed))
        return 1
    log.info("Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
