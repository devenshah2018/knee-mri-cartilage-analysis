"""One analysis path for single scans and cohorts, at baseline and follow-up."""

from dataclasses import asdict, dataclass
from math import ceil

import numpy as np
import pandas as pd
import pydicom
from scipy import stats

from .annotations import (
    ExcelReferences,
    annotation_index,
    compartment_index,
    kl_grade_for,
    load_kl_grades,
)
from .config import LINE_COORDINATES, SLICE_COORDINATES, Settings
from .geometry import (
    arc_length_pixels,
    compute_thickness_for_cdi_locations,
    find_cdi_locations,
    find_patella_mask,
    trace_edges,
)
from .inference import index_series, predict, read_slice, scan_ids, visit_directory


@dataclass
class Inputs:
    scan_ids: list
    annotations: dict
    references: ExcelReferences
    grades: dict
    series: dict
    selections: dict
    knees: dict
    inventory: pd.DataFrame


def prepare_inputs(settings: Settings) -> Inputs:
    """Validate every selected scan/visit before loading the large model."""
    ids = scan_ids(settings.dicom_dir, settings.scan_id)
    annotations = annotation_index(settings.annotation_dir, ids, settings.visits)
    references = ExcelReferences(settings.medial_excel, settings.lateral_excel)
    grades = load_kl_grades(settings.kl_excel)
    series, selections, knees, inventory = {}, {}, {}, []
    for scan_id in ids:
        for visit in settings.visits:
            key = scan_id, visit
            annotation = annotations.get(key)
            if annotation is None:
                raise ValueError(f"No Patella TXT annotation for scan {scan_id}, visit {visit}")
            time = settings.time_for(visit)
            knee = references.knee(scan_id, time) if references.tables else annotation.knee
            if knee != annotation.knee:
                raise ValueError(f"Excel/TXT knee mismatch for {scan_id}, {visit}")
            if (
                visit != settings.baseline_visit
                and knees[(scan_id, settings.baseline_visit)] != knee
            ):
                raise ValueError(f"Knee side changes between visits for {scan_id}")
            knees[key] = knee
            directory = visit_directory(settings.dicom_dir, scan_id, visit)
            series[key] = index_series(directory)
            selected = {}
            for coordinate in SLICE_COORDINATES:
                if settings.annotation_source == "excel" and visit == settings.baseline_visit:
                    compartment, _ = compartment_index(knee, coordinate)
                    row = references.row(compartment, scan_id, time)
                    start, end = float(row.Start_slice), float(row.End_slice)
                else:
                    start, end = annotation.start_slice, annotation.end_slice
                if not np.isfinite([start, end]).all() or start > end:
                    raise ValueError(f"Invalid slice range for {scan_id}, {visit}: {start}..{end}")
                if settings.annotation_source == "txt" or visit == settings.followup_visit:
                    annotated_slices = sorted(annotation.poi)
                    if len(annotated_slices) != len(SLICE_COORDINATES):
                        raise ValueError(f"{annotation.path.name}: expected six annotated slices")
                    selected[coordinate] = annotated_slices[SLICE_COORDINATES.index(coordinate)]
                else:
                    selected[coordinate] = ceil(start + (end - start) * coordinate / 200)
                if selected[coordinate] not in series[key]:
                    raise ValueError(
                        f"Missing DICOM slice {selected[coordinate]} for {scan_id}, {visit}"
                    )
            selections[key] = selected
            for coordinate, slice_id in selected.items():
                points = annotation.poi.get(slice_id, np.empty((0, 2)))
                if len(points) != 8:
                    raise ValueError(
                        f"{annotation.path.name}: expected 8 POI endpoints at slice {slice_id}, got {len(points)}"
                    )
                if len(annotation.bone.get(slice_id, [])) < 2:
                    raise ValueError(
                        f"{annotation.path.name}: missing bone outline at slice {slice_id}"
                    )
            inventory.append(
                {
                    "scan_id": scan_id,
                    "visit": visit,
                    "knee": knee,
                    "dicom_slices": len(series[key]),
                    "requested_slices": len(selected),
                    "annotation": annotation.path.name,
                }
            )
    return Inputs(
        ids, annotations, references, grades, series, selections, knees, pd.DataFrame(inventory)
    )


def prepare_volume_inputs(settings: Settings) -> Inputs:
    """Index full series without requiring manual annotations or patella Excel.

    Knee-qualified IDs or DICOM laterality identify the knee for optional KL
    matching. Unknown laterality remains unknown, rather than guessing a side.
    """
    ids = scan_ids(settings.dicom_dir, settings.scan_id)
    series, knees, inventory = {}, {}, []
    for scan_id in ids:
        for visit in settings.visits:
            key = scan_id, visit
            directory = visit_directory(settings.dicom_dir, scan_id, visit)
            series[key] = index_series(directory)
            suffix = scan_id.rpartition("_")[2] if "_" in scan_id else None
            header = pydicom.dcmread(next(iter(series[key].values())), stop_before_pixels=True)
            laterality = str(
                getattr(header, "ImageLaterality", "") or getattr(header, "Laterality", "")
            ).upper()
            knee = {"1": "right", "2": "left"}.get(
                suffix, {"R": "right", "L": "left"}.get(laterality, "unknown")
            )
            if visit == settings.followup_visit:
                baseline_knee = knees[(scan_id, settings.baseline_visit)]
                if "unknown" not in (baseline_knee, knee) and baseline_knee != knee:
                    raise ValueError(f"DICOM knee side changes between visits for {scan_id}")
            knees[key] = knee
            inventory.append(
                {"scan_id": scan_id, "visit": visit, "knee": knee, "dicom_files": len(series[key])}
            )
    return Inputs(
        ids,
        {},
        ExcelReferences(),
        load_kl_grades(settings.kl_excel),
        series,
        {},
        knees,
        pd.DataFrame(inventory),
    )


@dataclass
class AnalysisResult:
    points: pd.DataFrame
    bones: pd.DataFrame
    slices: list


def analyze(settings: Settings, inputs: Inputs, model, progress=print) -> AnalysisResult:
    """Infer once per unique requested slice; failed masks remain missing, never zero."""
    points, bones, slices = [], [], []
    for scan_id in inputs.scan_ids:
        for visit in settings.visits:
            key = scan_id, visit
            annotation = inputs.annotations[key]
            knee = inputs.knees[key]
            grade = kl_grade_for(inputs.grades, scan_id, knee)
            cache = {}
            for coordinate, slice_id in inputs.selections[key].items():
                if slice_id not in cache:
                    raw = read_slice(inputs.series[key][slice_id])
                    image, labels = predict(model, raw, settings.device)
                    mask = find_patella_mask(labels)
                    left, right = trace_edges(mask)
                    locations = find_cdi_locations(left, LINE_COORDINATES, mask)
                    thickness = compute_thickness_for_cdi_locations(locations, right)
                    cache[slice_id] = raw, image, labels, mask, left, right, locations, thickness
                raw, image, labels, mask, left, right, locations, thickness = cache[slice_id]
                found = bool(mask.any())
                common = {
                    "scan_id": scan_id,
                    "visit": visit,
                    "slice_coord": coordinate,
                    "slice_id": slice_id,
                    "knee": knee,
                    "kl_grade": grade,
                    "mask_found": found,
                }
                # Only the first scan is previewed; avoid retaining cohort-sized image arrays.
                if scan_id == inputs.scan_ids[0]:
                    slices.append(
                        dict(
                            common,
                            raw=raw,
                            image=image,
                            labels=labels,
                            mask=mask,
                            left=left,
                            right=right,
                            locations=locations,
                            thickness=thickness,
                            manual_bone=annotation.bone[slice_id],
                            manual_poi=annotation.poi[slice_id],
                        )
                    )
                calculated_length = arc_length_pixels(left) if found else np.nan
                manual_length = arc_length_pixels(annotation.bone[slice_id])
                excel_length = inputs.references.bone_length(
                    scan_id, settings.time_for(visit), knee, coordinate
                )
                bones.append(
                    dict(
                        common,
                        calculated_bone_px=calculated_length,
                        txt_bone_px=manual_length,
                        excel_bone_px=excel_length,
                        bone_error_px=calculated_length - excel_length,
                    )
                )
                for index, line_coordinate in enumerate(LINE_COORDINATES):
                    start, end = annotation.poi[slice_id][2 * index : 2 * index + 2]
                    measured = thickness.get(f"coord_{line_coordinate}", {})
                    calculated = measured.get("thickness", np.nan)
                    calculated = calculated if found and np.isfinite(calculated) else np.nan
                    manual = float(np.linalg.norm(end - start))
                    location = locations.get(f"coord_{line_coordinate}")
                    displacement = (
                        float(np.linalg.norm(np.asarray(location)[::-1] - start))
                        if location is not None
                        else np.nan
                    )
                    points.append(
                        dict(
                            common,
                            line_coord=line_coordinate,
                            calculated_px=calculated,
                            manual_px=manual,
                            error_px=calculated - manual,
                            displacement_px=displacement,
                        )
                    )
            progress(
                f"{scan_id} {visit}: {sum(bool(cache[s][3].any()) for s in cache)}/{len(cache)} slices with patellar masks"
            )
    return AnalysisResult(pd.DataFrame(points), pd.DataFrame(bones), slices)


def longitudinal(points, baseline, followup):
    """Pair the same scan, slice coordinate, and CDI location across visits."""
    keys = ["scan_id", "slice_coord", "line_coord"]
    values = ["calculated_px", "manual_px"]
    before = points.loc[points.visit == baseline, keys + values]
    after = points.loc[points.visit == followup, keys + values]
    paired = before.merge(
        after, on=keys, suffixes=("_baseline", "_followup"), how="outer", validate="one_to_one"
    )
    for method in ("calculated", "manual"):
        paired[f"{method}_delta_px"] = (
            paired[f"{method}_px_followup"] - paired[f"{method}_px_baseline"]
        )
    paired["delta_error_px"] = paired.calculated_delta_px - paired.manual_delta_px
    return paired


def scan_totals(points):
    """A total is valid only when all 24 locations are present for both methods.

    These sums of thicknesses (pixels) are not the area-valued Excel CDI column.
    """
    expected = len(SLICE_COORDINATES) * len(LINE_COORDINATES)
    keys = ["scan_id", "visit", "slice_coord", "line_coord"]
    if points.duplicated(keys).any():
        raise ValueError("Duplicate point measurements cannot form scan totals")
    rows = []
    for (scan_id, visit), group in points.groupby(["scan_id", "visit"]):
        complete = (
            len(group) == expected and group[["calculated_px", "manual_px"]].notna().all().all()
        )
        rows.append(
            {
                "scan_id": scan_id,
                "visit": visit,
                "complete": complete,
                "calculated_px": group.calculated_px.sum() if complete else np.nan,
                "manual_px": group.manual_px.sum() if complete else np.nan,
                "valid_points": int(group.calculated_px.notna().sum()),
                "kl_grade": group.kl_grade.iloc[0],
            }
        )
    return pd.DataFrame(rows)


def agreement(manual, calculated):
    """Paired, finite observations only; no spurious correlations for n < 2."""
    manual, calculated = np.asarray(manual, dtype=float), np.asarray(calculated, dtype=float)
    finite = np.isfinite(manual) & np.isfinite(calculated)
    manual, calculated = manual[finite], calculated[finite]
    difference = calculated - manual
    result = {
        "n": len(manual),
        "bias_px": np.nan,
        "mae_px": np.nan,
        "loa_lower_px": np.nan,
        "loa_upper_px": np.nan,
        "pearson_r": np.nan,
        "spearman_r": np.nan,
        "kendall_tau": np.nan,
    }
    if len(manual):
        result.update(bias_px=float(difference.mean()), mae_px=float(np.abs(difference).mean()))
    if len(manual) >= 2:
        spread = difference.std(ddof=1)
        result.update(
            loa_lower_px=float(difference.mean() - 1.96 * spread),
            loa_upper_px=float(difference.mean() + 1.96 * spread),
        )
        if np.ptp(manual) > 0 and np.ptp(calculated) > 0:
            result.update(
                pearson_r=float(stats.pearsonr(manual, calculated).statistic),
                spearman_r=float(stats.spearmanr(manual, calculated).statistic),
                kendall_tau=float(stats.kendalltau(manual, calculated).statistic),
            )
    return result


def agreement_table(points, changes):
    rows = []
    for visit, group in points.groupby("visit"):
        rows.append(dict(comparison=visit, **agreement(group.manual_px, group.calculated_px)))
    rows.append(
        dict(
            comparison="followup_minus_baseline",
            **agreement(changes.manual_delta_px, changes.calculated_delta_px),
        )
    )
    return pd.DataFrame(rows)


def scan_total_tables(totals, baseline, followup):
    """Pair complete scan totals and retain the original scan-level comparisons."""
    before = totals.loc[
        totals.visit == baseline, ["scan_id", "kl_grade", "calculated_px", "manual_px"]
    ]
    after = totals.loc[totals.visit == followup, ["scan_id", "calculated_px", "manual_px"]]
    changes = before.merge(
        after, on="scan_id", suffixes=("_baseline", "_followup"), how="outer", validate="one_to_one"
    )
    for method in ("calculated", "manual"):
        changes[f"{method}_delta_px"] = (
            changes[f"{method}_px_followup"] - changes[f"{method}_px_baseline"]
        )
    changes["delta_error_px"] = changes.calculated_delta_px - changes.manual_delta_px
    rows = []
    for visit, group in totals.groupby("visit"):
        rows.append(dict(comparison=visit, **agreement(group.manual_px, group.calculated_px)))
    rows.append(
        dict(
            comparison="followup_minus_baseline",
            **agreement(changes.manual_delta_px, changes.calculated_delta_px),
        )
    )
    return changes, pd.DataFrame(rows)


def export_results(settings, inputs, result):
    """Write all patient-derived outputs under the ignored output directory."""
    import json

    output = settings.output_dir
    output.mkdir(parents=True, exist_ok=True)
    changes = longitudinal(result.points, *settings.visits)
    tables = {
        "input_inventory": inputs.inventory,
        "point_measurements": result.points,
        "bone_validation": result.bones,
        "longitudinal_changes": changes,
        "scan_totals": scan_totals(result.points),
        "agreement": agreement_table(result.points, changes),
    }
    grouped = result.points.groupby(["visit", "kl_grade"], dropna=False)
    tables["validation_by_kl"] = grouped.agg(
        scans=("scan_id", "nunique"),
        valid_points=("calculated_px", "count"),
        mean_error_px=("error_px", "mean"),
        mean_displacement_px=("displacement_px", "mean"),
    ).reset_index()
    total_changes, total_agreement = scan_total_tables(tables["scan_totals"], *settings.visits)
    tables["scan_total_changes"] = total_changes
    tables["scan_total_agreement"] = total_agreement
    tables["scan_changes_by_kl"] = (
        total_changes.groupby("kl_grade", dropna=False)
        .agg(
            scans=("scan_id", "size"),
            valid_pairs=("calculated_delta_px", "count"),
            calculated_delta_mean_px=("calculated_delta_px", "mean"),
            manual_delta_mean_px=("manual_delta_px", "mean"),
            delta_error_mean_px=("delta_error_px", "mean"),
            delta_error_std_px=("delta_error_px", "std"),
        )
        .reset_index()
    )
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    with pd.ExcelWriter(output / "analysis.xlsx", engine="openpyxl") as workbook:
        for name, table in tables.items():
            table.to_excel(workbook, sheet_name=name, index=False)
    (output / "run_config.json").write_text(
        json.dumps(asdict(settings), indent=2, default=str) + "\n"
    )
    return tables
