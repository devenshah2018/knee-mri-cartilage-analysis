"""Optional whole-cartilage exports, replacing the standalone segmentation runner.

This path uses raw z-score normalization, as the former whole-cartilage notebook
 did. The patellar CDI analysis uses its historical enhanced-image preprocessing.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from .annotations import kl_grade_for
from .inference import predict

TISSUES = {1: "patellar_cartilage", 2: "femoral_cartilage", 3: "meniscus", 4: "tibial_cartilage"}


def segment_volume(model, paths, device="auto"):
    """Segment a series, supporting 2D slices and grayscale multiframe DICOM."""
    masks = []
    voxel_sizes = []
    spacing_valid = True
    for path in paths:
        dicom = pydicom.dcmread(path)
        pixels = dicom.pixel_array
        if getattr(dicom, "SamplesPerPixel", 1) != 1 or pixels.ndim not in (2, 3):
            raise ValueError(f"Expected grayscale 2D/multiframe DICOM: {path}")
        frames = pixels[None] if pixels.ndim == 2 else pixels
        spacing = getattr(dicom, "PixelSpacing", None)
        depth = getattr(dicom, "SpacingBetweenSlices", getattr(dicom, "SliceThickness", None))
        if spacing is None or depth is None:
            spacing_valid = False
        else:
            sizes = np.array([*map(float, spacing), abs(float(depth))])
            if not np.isfinite(sizes).all() or (sizes <= 0).any():
                spacing_valid = False
            else:
                voxel_sizes.extend([float(np.prod(sizes))] * len(frames))
        for frame in frames:
            _, labels = predict(model, frame.astype(np.float32), device, enhance=False)
            masks.append(labels)
    if not masks:
        raise ValueError("No DICOM frames to segment")
    return np.stack(masks), voxel_sizes if spacing_valid else None


def export_volumes(settings, inputs, model):
    """Export all selected visits, not just the six CDI slices. May take hours."""
    rows = []
    for (scan_id, visit), series in inputs.series.items():
        volume, voxel_sizes = segment_volume(model, series.values(), settings.device)
        output = Path(settings.output_dir) / "volumes" / scan_id / visit
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "segmentation.npy", volume)
        row = {
            "scan_id": scan_id,
            "visit": visit,
            "slices": len(volume),
            "kl_grade": kl_grade_for(inputs.grades, scan_id, inputs.knees[(scan_id, visit)]),
        }
        for label, name in TISSUES.items():
            counts = (volume == label).sum(axis=(1, 2))
            row[f"{name}_voxels"] = int(counts.sum())
            row[f"{name}_mm3"] = (
                float(np.dot(counts, voxel_sizes)) if voxel_sizes is not None else np.nan
            )
        cartilage = ["patellar_cartilage", "femoral_cartilage", "tibial_cartilage"]
        row["whole_cartilage_voxels"] = sum(row[f"{name}_voxels"] for name in cartilage)
        row["whole_cartilage_mm3"] = sum(row[f"{name}_mm3"] for name in cartilage)
        rows.append(row)
    summary = pd.DataFrame(rows)
    changes, by_kl = volume_changes(summary, *settings.visits)
    tables = {"volume_summary": summary, "volume_changes": changes, "volume_by_kl": by_kl}
    for name, table in tables.items():
        table.to_csv(Path(settings.output_dir) / f"{name}.csv", index=False)
    with pd.ExcelWriter(Path(settings.output_dir) / "volumes.xlsx", engine="openpyxl") as book:
        for name, table in tables.items():
            table.to_excel(book, sheet_name=name, index=False)
    return tables


def volume_changes(summary, baseline, followup):
    """Baseline/follow-up, delta and percent change by tissue and whole cartilage.

    Whole cartilage always excludes meniscus. Zero baseline or missing spacing
    leaves percent change undefined rather than reporting an artificial zero.
    """
    metrics = [*TISSUES.values(), "whole_cartilage"]
    values = [f"{name}_mm3" for name in metrics]
    before = summary.loc[summary.visit == baseline, ["scan_id", "kl_grade", *values]]
    after = summary.loc[summary.visit == followup, ["scan_id", *values]]
    paired = before.merge(
        after, on="scan_id", suffixes=("_baseline", "_followup"), how="outer", validate="one_to_one"
    )
    rows = []
    for name in metrics:
        baseline_values = paired[f"{name}_mm3_baseline"] / 1000
        followup_values = paired[f"{name}_mm3_followup"] / 1000
        paired[f"{name}_baseline_cm3"] = baseline_values
        paired[f"{name}_followup_cm3"] = followup_values
        paired[f"{name}_delta_cm3"] = followup_values - baseline_values
        paired[f"{name}_pct_change"] = (
            100 * (followup_values - baseline_values) / baseline_values.where(baseline_values > 0)
        )
        for grade, group in paired.groupby("kl_grade", dropna=False):
            rows.append(
                {
                    "tissue": name,
                    "kl_grade": grade,
                    "scans": len(group),
                    "valid_pairs": int(group[f"{name}_delta_cm3"].notna().sum()),
                    "baseline_mean_cm3": group[f"{name}_baseline_cm3"].mean(),
                    "followup_mean_cm3": group[f"{name}_followup_cm3"].mean(),
                    "delta_mean_cm3": group[f"{name}_delta_cm3"].mean(),
                    "delta_std_cm3": group[f"{name}_delta_cm3"].std(),
                    "pct_change_mean": group[f"{name}_pct_change"].mean(),
                }
            )
    paired = paired.drop(
        columns=[f"{name}_mm3_{visit}" for name in metrics for visit in ("baseline", "followup")]
    )
    return paired, pd.DataFrame(rows)
