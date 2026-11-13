"""Compact, reusable notebook figures. Saving is explicit and stays under results/."""

import matplotlib.pyplot as plt
import numpy as np

from .analysis import agreement
from .config import LINE_COORDINATES
from .geometry import zoom_region

COLORS = {1: (1, 0.2, 0.2), 2: (0.2, 1, 0.2), 3: (1, 0.8, 0.2), 4: (0.2, 0.4, 1)}


def plot_slices(result, scan_id, visit):
    """Show preprocessing, segmentation, edges, and manual CDI on the same slices."""
    selected = [s for s in result.slices if s["scan_id"] == str(scan_id) and s["visit"] == visit]
    if not selected:
        raise ValueError(f"No slices for {scan_id}, {visit}")
    figure, axes = plt.subplots(len(selected), 3, figsize=(12, 3 * len(selected)), squeeze=False)
    for row, item in zip(axes, selected):
        r0, r1, c0, c1 = zoom_region(item["image"].shape, item["mask"], size=180)
        row[0].imshow(item["raw"], cmap="gray")
        row[0].set_title(f"Slice {item['slice_id']} (coord {item['slice_coord']})")
        row[1].imshow(item["image"], cmap="gray")
        overlay = np.zeros((*item["labels"].shape, 4))
        for label, color in COLORS.items():
            overlay[item["labels"] == label] = (*color, 0.5)
        row[1].imshow(overlay)
        row[1].set_title("Enhanced image + segmentation")
        axis = row[2]
        axis.imshow(item["image"], cmap="gray")
        for name, color in (("left", "cyan"), ("right", "magenta")):
            edge = item[name]
            axis.plot(edge[:, 1], edge[:, 0], color=color)
        manual_bone = item["manual_bone"]
        axis.plot(manual_bone[:, 0], manual_bone[:, 1], color="lime", label="Manual bone")
        for index, line_coordinate in enumerate(LINE_COORDINATES):
            endpoints = item["manual_poi"][2 * index : 2 * index + 2]
            axis.plot(
                endpoints[:, 0],
                endpoints[:, 1],
                color="orange",
                linestyle=":",
                label="Manual thickness" if index == 0 else None,
            )
            path = item["thickness"].get(f"coord_{line_coordinate}", {}).get("path_coords")
            if path is not None and len(path):
                axis.plot(
                    path[:, 1],
                    path[:, 0],
                    color="cyan",
                    linestyle="--",
                    label="Calculated thickness" if index == 0 else None,
                )
        axis.set_xlim(c0, c1)
        axis.set_ylim(r1, r0)
        axis.set_title("CDI validation" if item["mask_found"] else "No patellar mask")
        axis.legend(fontsize=7, loc="upper right")
        for axis in row:
            axis.axis("off")
    figure.suptitle(f"{scan_id} · {visit}")
    figure.tight_layout()
    return figure


def plot_agreement(manual, calculated, title, unit="px", reference="Manual", estimate="Calculated"):
    """Correlation and Bland–Altman views with finite paired observations."""
    manual, calculated = np.asarray(manual, dtype=float), np.asarray(calculated, dtype=float)
    valid = np.isfinite(manual) & np.isfinite(calculated)
    manual, calculated = manual[valid], calculated[valid]
    summary = agreement(manual, calculated)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(manual, calculated, alpha=0.6, s=20)
    if len(manual):
        limits = [min(manual.min(), calculated.min()), max(manual.max(), calculated.max())]
        axes[0].plot(limits, limits, "k--", linewidth=1)
    axes[0].set(
        xlabel=f"{reference} ({unit})",
        ylabel=f"{estimate} ({unit})",
        title=f"{title}: n={summary['n']}, r={summary['pearson_r']:.3f}",
    )
    axes[1].scatter((manual + calculated) / 2, calculated - manual, alpha=0.6, s=20)
    for key, style in (("bias_px", "-"), ("loa_lower_px", "--"), ("loa_upper_px", "--")):
        if np.isfinite(summary[key]):
            axes[1].axhline(summary[key], color="gray", linestyle=style)
    axes[1].set(
        xlabel=f"Pair mean ({unit})",
        ylabel=f"{estimate} − {reference} ({unit})",
        title="Bland–Altman",
    )
    figure.tight_layout()
    return figure


def plot_kl_validation(points):
    """Show scan-level displacement means, preserving the unit of replication."""
    scan_means = (
        points.groupby(["scan_id", "visit", "kl_grade"]).displacement_px.mean().reset_index()
    )
    visits = sorted(points.visit.unique())
    figure, axes = plt.subplots(1, len(visits), figsize=(5 * len(visits), 4), squeeze=False)
    for axis, visit in zip(axes[0], visits):
        group = scan_means[scan_means.visit == visit]
        for grade in range(5):
            values = group.loc[group.kl_grade == grade, "displacement_px"].dropna()
            axis.scatter([grade] * len(values), values, alpha=0.6)
        axis.set(
            xlabel="Baseline KL grade",
            ylabel="Mean CDI displacement per scan (px)",
            title=visit,
            xticks=range(5),
        )
    figure.tight_layout()
    return figure


def plot_volume_changes(tables):
    """Whole-cartilage pairing, percentage changes and tissue volumes by KL."""
    changes = tables["volume_changes"]
    figures = {
        "whole_cartilage_agreement": plot_agreement(
            changes.whole_cartilage_baseline_cm3,
            changes.whole_cartilage_followup_cm3,
            "Whole cartilage",
            unit="cm³",
            reference="Baseline",
            estimate="Follow-up",
        )
    }
    figure, axis = plt.subplots(figsize=(7, 4))
    for grade, group in changes.groupby("kl_grade", dropna=False):
        values = group.whole_cartilage_pct_change.dropna()
        if len(values):
            axis.hist(
                values,
                bins=15,
                alpha=0.5,
                label=f"KL {grade:g}" if np.isfinite(grade) else "KL unknown",
            )
    axis.set(xlabel="Whole-cartilage change (%)", ylabel="Scans", title="Follow-up minus baseline")
    if axis.patches:
        axis.legend()
    figure.tight_layout()
    figures["whole_cartilage_percent_change"] = figure
    summary = tables["volume_by_kl"]
    figure, axis = plt.subplots(figsize=(10, 4))
    tissues = list(summary.tissue.unique())
    grades = summary.kl_grade.dropna().unique()
    width = 0.8 / max(1, len(grades))
    for index, grade in enumerate(sorted(grades)):
        group = summary.loc[summary.kl_grade == grade].set_index("tissue").reindex(tissues)
        axis.bar(
            np.arange(len(tissues)) + (index - (len(grades) - 1) / 2) * width,
            group.baseline_mean_cm3,
            width=width,
            label=f"KL {grade:g}",
        )
    axis.set(
        xticks=np.arange(len(tissues)),
        xticklabels=[t.replace("_", " ") for t in tissues],
        ylabel="Mean baseline volume (cm³)",
        title="Tissue volumes by baseline KL",
    )
    axis.tick_params(axis="x", labelrotation=15)
    if len(grades):
        axis.legend()
    figure.tight_layout()
    figures["tissue_volume_by_kl"] = figure
    return figures
