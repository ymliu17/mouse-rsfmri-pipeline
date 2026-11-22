# Mouse rs-fMRI functional connectivity pipeline

Preprocess mouse resting-state fMRI from converted NIfTI and compute ROI-to-ROI functional connectivity in Allen CCF space. The workflow was adapted from a Bruker 7T study. It starts with a 3-D T2-weighted image and one or more 4-D EPI runs per subject. Convert scanner data to NIfTI before running.

## Install

Use Python 3.10 or newer and install the pinned Python dependencies in a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/index.html), [ANTs](https://github.com/ANTsX/ANTs), and [AFNI](https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/background_install/install_instructs/index.html) for your system. The runner looks for FSL tools in `$FSLDIR/bin`, ANTs tools in `$ANTSPATH`, and AFNI tools in `$AFNI_DIR`, with a fallback to `PATH`. Initialize FSL according to its installation instructions and ensure `dc` is on `PATH` for FSL BET's robust mode (on Debian/Ubuntu, install the `dc` package).

## Prepare the atlas

Use 25 µm and 100 µm average templates and the 100 µm CCFv3 (2017) annotation from the Allen Institute. The converter reproduces the source workflow's NIfTI coordinate convention and builds the 100 µm mask from positive annotation voxels. Use matching templates, labels and masks when substituting an existing atlas. [Allen reference space API](https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/api/queries/reference_space_api.py)

```bash
mkdir -p data/ccf-nrrd allen_ccf
ccf=https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf
curl -fL "$ccf/average_template/average_template_25.nrrd" -o data/ccf-nrrd/average_template_25.nrrd
curl -fL "$ccf/average_template/average_template_100.nrrd" -o data/ccf-nrrd/average_template_100.nrrd
curl -fL "$ccf/annotation/ccf_2017/annotation_100.nrrd" -o data/ccf-nrrd/annotation_100.nrrd
python prepare_atlas.py --nrrd-dir data/ccf-nrrd
curl -fL https://api.brain-map.org/api/v2/structure_graph_download/1.json -o allen_ccf/allen_structure_graph.json
```

The runner also downloads the ontology on demand if absent. For offline jobs, stage the JSON together with the atlas. Existing atlas directories can be supplied with `--atlas-dir`. Required volumes are `allen_template.nii.gz`, `allen_template_100um.nii.gz`, `allen_labels_100um.nii.gz`, and `allen_brain_mask_100um.nii.gz`.

## Run

Copy [manifest.example.tsv](manifest.example.tsv) to `manifest.tsv` and update its paths. Each row is a run. Use the same anatomy path for all runs of a subject. Relative paths are resolved from the manifest's directory. Runs are numbered in row order. Subject IDs can contain letters, digits, and hyphens. Output IDs omit hyphens for compatibility with existing derivatives.

```bash
python preprocess_fmri.py --manifest manifest.tsv --check-only
python preprocess_fmri.py --manifest manifest.tsv --output-dir derivatives --qc-dir QC
```

For the original Bruker naming convention, use `--input-dir /path/to/NPIL_DICOM`. Discovery takes the first lexically sorted `T2star_FID_EPI` NIfTI (excluding ghost adjustment and B0 maps) and the last lexically sorted `T2_RAREvfl_3D_iso` NIfTI per subject. Use a manifest to select files explicitly. Use output paths without whitespace or brackets/commas for compatibility with external tools.

| Options | Behavior |
| --- | --- |
| `--check-only` | Validate selected NIfTI headers, atlas grids, and tool paths. |
| `--subjects mouse01 mouse02` | Run a subset. For concurrent jobs, select disjoint subjects and give each job its own `--log-file`. |
| `--tr 1.0 --trim 10 --bandpass 0.01 0.10` | Override TR in seconds. The default uses each source NIfTI header. Trim and cutoffs shown are defaults. |
| `--roi-depth 2` | Merge Allen ontology nodes at the selected depth, with the root at depth 0. |
| `--bet-f-anat 0.35 --bet-f-func 0.30 --fd-radius 50` | Historical BET and QC parameters, shown at their defaults. Set the FD radius in mm to match your analysis protocol. |
| `--skip-existing` | Resume with identical source files, run order, atlas and settings. Use a fresh output directory for changed inputs or preprocessing settings. |
| `--qc-only` | Regenerate available QC from derivatives using the same manifest or discovery settings. |
| `--cleanup` | Delete trimmed, motion-corrected, warped and cleaned EPI after use. Omit this option for follow-up analyses that need cleaned EPI. |

The runner performs T2 skull stripping and T2-to-Allen registration, then trims each EPI run, motion-corrects it, aligns the mean EPI to T2, and applies the forward transform chain to the 4-D EPI. AFNI jointly regresses motion, motion derivatives and polynomial drift while applying the bandpass. The runner extracts in-mask ROI means and averages leaf time series equally within each selected ontology ancestor. It saves Fisher-z-transformed Pearson FC and generates FD, registration overlay/NCC, and FC QC. QC metrics are descriptive and thresholds should be reviewed for the acquisition. [AFNI 3dTproject](https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTproject.html) · [AFNI 3dROIstats](https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dROIstats.html)

## Outputs and follow-up

- `derivatives/sub-*/anat/` holds the skull-stripped anatomy and T2-to-CCF transforms.
- `derivatives/sub-*/func/run-*/` holds motion and EPI intermediates.
- `derivatives/sub-*/connectivity/` holds ROI time series, Allen IDs, merged ROI metadata and FC matrices in TSV and NumPy formats.
- `QC/sub-*/` holds per-run QC figures and metrics. The default run log is `derivatives/preprocessing.log`.

```bash
python collect_fc.py --derivatives derivatives --out fc_all_subjects.csv
python build_coarse_atlas.py
python roi_fc.py --cleaned /path/to/run_bold_cleaned.nii.gz \
  --labels allen_ccf/coarse_allen_labels.nii.gz \
  --lookup allen_ccf/coarse_roi_lookup.csv --out-dir roi-results
python seed_fc.py --cleaned /path/to/run_bold_cleaned.nii.gz \
  --labels allen_ccf/coarse_allen_labels.nii.gz --seed-id 385 --out-dir seed-results
python roi_notes.py --roi-ids /path/to/run_roi_ids_merged.tsv
```

`collect_fc.py` reads run 1 by default, requires matching ROI rows and columns across inputs, and writes zero-based upper-triangle column indices. Supply `--run` for another run and `--groups groups.tsv` for a TSV with `subject` and `group` columns. For seed FC, select an ID present in the lookup or pass a seed mask on the cleaned EPI grid.

The ROI extractor remaps Allen IDs to temporary contiguous labels for AFNI and preserves the original IDs in its outputs. The Pearson calculation uses normalized centered vectors. Both corrections can change FC values from the original study scripts. On a resumed legacy run, ROI and FC products are regenerated using the corrected implementations. Keep legacy derivatives separately if you need to compare versions.
