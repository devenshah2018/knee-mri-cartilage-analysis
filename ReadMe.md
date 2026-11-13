# Patellar cartilage thickness with SKM-TEA

Segment knee DICOM images, measure patellar cartilage thickness, validate against
manual annotations and Excel references, and compare visits using Stanford's
pretrained [SKM-TEA](https://github.com/StanfordMIMI/skm-tea) U-Net.

**Start with [`runner.ipynb`](runner.ipynb).** It is the single analysis entry point.
The former marimo runners and duplicate standalone notebook have been retired.
The notebook explains the workflow; [`skm_tea_cdi/`](skm_tea_cdi/) holds reusable code.

## Install

Use **Python 3.11**. Upstream `meddlr==0.0.9` requires `numpy<=1.23.5`;
newer Python/NumPy combinations are not supported by this project.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name skm-tea-cdi --display-name "SKM-TEA CDI (Python 3.11)"
python -m jupyterlab runner.ipynb
```

Windows activation: `.venv\Scripts\activate`. Select the **SKM-TEA CDI
(Python 3.11)** kernel in JupyterLab, or `.venv` in VS Code. A CUDA GPU is optional;
`device="auto"` uses CPU when CUDA is unavailable. Installation and execution were
tested on Apple Silicon/macOS. Platform markers are included for other systems,
but those systems have not been runtime tested.

`requirements.in` lists direct dependencies; `requirements.txt` pins the resolved
transitive dependencies. Neither includes marimo. With uv, use `uv venv --python
3.11` followed by `uv pip install -r requirements.txt`.

**Excel dependencies:** `xlrd` reads legacy `.xls`; `openpyxl` reads `.xlsx` and
writes `analysis.xlsx`. Both are included in the requirements. Microsoft Excel
and LibreOffice are not required. The sample `.xls` files contain empty leading
sheets, so the code explicitly reads **`main`**. See the
[pandas Excel engine documentation](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html).

## Provide your own data

**Copyrighted datasets, clinical workbooks, pretrained weights, and run outputs
are not distributed with this repository.** Obtain inputs you are authorized to
use. Place them in these ignored local directories, or configure your own paths:

```text
datasets/
├── dess-phase-3/<scan-id>/
│   ├── v00/                         # baseline DICOM slice files
│   └── v03/                         # follow-up DICOM slice files
├── u01-phase-3/
│   ├── *<scan-id>_v00_*Patella.txt
│   └── *<scan-id>_v03_*Patella.txt
└── clinical/
    ├── Phase_3_Medial_Patella.xls    # optional in TXT mode
    ├── Phase_3_Lateral_Patella.xls
    ├── KL for Ming May 12 2016.xls   # optional Phase 3 KL reference
    └── combined.xlsx                # optional alternative clinical/KL reference
models/
├── config.yaml
└── model.ckpt
```

The formats were checked against the local Phase 3 images, annotations, and
workbooks used for testing. The documentation uses synthetic examples rather than
copies of those records. Filenames and dataset directory names are configurable.

| Input | Required contents |
|---|---|
| DICOM | One grayscale 2D file per slice, grouped by scan and visit. Numeric filenames supply slice IDs; `.dcm` filenames use `InstanceNumber`. |
| Patella TXT | One annotation per scan and visit: seven-line header, six bone outlines under `Patella`, four endpoint pairs per slice under `POI`. |
| Medial/lateral Excel | Optional in default TXT mode; both required in Excel selection mode. Sheet `main`: `id`, `time`, `knee`, `Start_slice`, `End_slice`, `BoneLength_medial_1` through `_3`. The lateral legacy file uses the same medial-named columns. |
| KL Excel | Optional: `FORMING` sheet with `id`, `side`, `V00XRKL`; or `combined.xlsx` first sheet with `ID`, `side`, `V00XRKL`. |
| Model | Published SKM-TEA `U-Net_E1` configuration and checkpoint, local or remote. |

Clinical reference workbooks belong in `datasets/clinical/`. The real-data
`kl_sample_manifest.csv` stays under `datasets/` as a dataset-selection manifest;
it is also excluded from Git. Other spreadsheets shipped inside the original
annotation package remain with that raw package and are not pipeline inputs.

For TXT-only use, set `medial_excel=None`, `lateral_excel=None`, `kl_excel=None`.
DICOM, both visits' TXT annotations, and the model remain required; the independent
Excel bone comparisons and KL stratification are then skipped.

### Model files

Save the published [configuration](https://huggingface.co/arjundd/skm-tea-models/raw/main/neurips2021/segmentation-models/U-Net_E1/config.yaml)
and [checkpoint](https://huggingface.co/arjundd/skm-tea-models/resolve/main/neurips2021/segmentation-models/U-Net_E1/model.ckpt)
as `models/config.yaml` and `models/model.ckpt`. Alternatively, import
`MODEL_CONFIG_URL` and `MODEL_WEIGHTS_URL` from `skm_tea_cdi.config` and use them as
the notebook's model paths. Local files require no network access.

Only load trusted checkpoints. The published checkpoint uses pickle; the loader
opts into it for that load without globally changing PyTorch behavior. See
[PyTorch serialization](https://docs.pytorch.org/docs/stable/notes/serialization).

## Run

1. Open `runner.ipynb`, select the Python 3.11 kernel, and edit its configuration.
2. Set `scan_id` to your scan directory name, or `None` for the whole cohort.
3. Set the visits and Excel times (sample defaults: `v00`/`v03`, `0`/`24`).
4. Choose **Restart Kernel and Run All Cells**.

Preflight validates both visits and all required reference measurements before
loading the model. Default `annotation_source="txt"` uses the six actual slice
numbers recorded in each annotation; Excel remains an independent bone-length
reference. `annotation_source="excel"` reproduces the original baseline range
formula, while follow-up still uses recorded slices. Some local baseline records
differ from the formula by one slice, so TXT mode is recommended for cohorts.

The pipeline segments each requested slice once and exports the same tables for
single knees and cohorts. Failed masks remain `NaN` with `mask_found=False`.
Scan totals require all 24 valid point measurements. Delta means **follow-up minus
baseline**, and error means **calculated minus manual**.

Thickness, displacement, and bone length are in **pixels**, as in the original
analysis. Scan totals sum thicknesses; they are not the area-valued spreadsheet
`CDI`. Point-level statistics are descriptive because measurements within a knee
are correlated. Review missing masks and incomplete totals before interpreting results.

### Optional whole-cartilage segmentation

Set `RUN_WHOLE_VOLUMES=True` to segment every slice in each selected series. It is
disabled by default because it is slower. Labels are `0=background`, `1=patellar
cartilage`, `2=femoral cartilage`, `3=meniscus`, `4=tibial cartilage`. Meniscus is
reported separately from cartilage. This path preserves the former standalone
runner's raw-image z-score preprocessing; CDI uses its original CLAHE/blur/erosion
preprocessing. Set `RUN_CDI_ANALYSIS=False` and `RUN_WHOLE_VOLUMES=True` for an
image-only run, which requires no Patella TXT or medial/lateral workbooks. Set
`kl_excel=None` if you have no KL references. Knee-qualified IDs or DICOM laterality
supply side identity for optional KL matching; unknown side leaves KL missing.

## Outputs

Everything is saved under `output_dir` (default: ignored `results/`):

| File | Contents |
|---|---|
| `input_inventory.csv` | Scan/visit identities, annotation names, slice counts |
| `point_measurements.csv` | Manual/calculated thickness, errors, displacement, mask status, KL grade |
| `bone_validation.csv` | Calculated, TXT, and Excel bone lengths |
| `longitudinal_changes.csv` | Paired point-level follow-up minus baseline changes |
| `scan_totals.csv` | Complete thickness sums and valid-point counts |
| `agreement.csv` | Bias, MAE, limits of agreement, Pearson/Spearman/Kendall statistics |
| `validation_by_kl.csv` | Validation summaries by visit and KL grade |
| `scan_total_changes.csv` | Paired whole-scan thickness changes |
| `scan_total_agreement.csv` | Scan-level baseline, follow-up, and delta agreement |
| `scan_changes_by_kl.csv` | Scan-total change summaries by baseline KL |
| `analysis.xlsx` | The same CDI tables as separate sheets |
| `run_config.json` | Paths and settings used |
| `*.png` | Slice previews, agreement plots, KL displacement figures |
| `volumes/<scan>/<visit>/segmentation.npy` | Optional integer labels, shape `(slices, rows, columns)` |
| `volume_summary.csv` | Optional tissue/whole-cartilage voxel counts and physical volume estimates |
| `volume_changes.csv` | Paired tissue/whole-cartilage volume, change, and percent change |
| `volume_by_kl.csv`, `volumes.xlsx` | Optional volume summaries by KL and all volume tables |

Physical volumes use DICOM pixel spacing and interslice spacing, falling back to
slice thickness; missing metadata yields `NaN`, never a fabricated voxel size.
Cohort figures preview the first scan, while tables cover all selected scans.

## Code map and verification

```text
runner.ipynb                documented notebook workflow
skm_tea_cdi/
  config.py                 paths and run settings
  annotations.py            TXT parsing, Excel references, KL grades
  inference.py              DICOM indexing, preprocessing, model inference
  geometry.py               masks, edges, arc coordinates, thickness
  analysis.py               shared analysis, validation, statistics, exports
  plotting.py               diagnostic and agreement figures
  volume.py                 optional whole-cartilage segmentation
scripts/execute_notebook.py  fresh-kernel notebook verification
tests/                      synthetic behavior and notebook-hygiene tests
```

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
ruff check skm_tea_cdi scripts tests runner.ipynb
ruff format --check skm_tea_cdi scripts tests runner.ipynb
python scripts/execute_notebook.py                 # historical Excel slice selection
python scripts/execute_notebook.py --txt           # default TXT slice selection
python scripts/execute_notebook.py --no-excel      # no optional reference workbooks
python scripts/execute_notebook.py --txt --all-scans
python scripts/execute_notebook.py --no-excel --volumes-only
```

Notebook execution commands require your local inputs and model. They use the
invoking Python interpreter and save executed copies under `results/verification/`,
leaving the source notebook clean. `--scan-id <id>` selects another knee;
`--volumes` also exercises the optional volume export.

## License

Copyright (c) 2026 Deven Shah. This project's code is released under the
[MIT License](LICENSE). External datasets and pretrained models retain their
respective licenses and terms of use.
