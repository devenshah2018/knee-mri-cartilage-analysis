"""Patellar geometry preserved from runner.ipynb.

Image coordinates are (row, column); TXT annotations use (x, y).
Thickness and arc length are measured in pixels, without physical rescaling.
The intersection routine retains the original nearest-edge fallback.
"""

import numpy as np
from scipy.signal import savgol_filter
from skimage import measure


def find_patella_mask(seg_pred):
    patella_mask = (seg_pred == 1).astype(np.uint8)
    if np.sum(patella_mask) == 0:
        return patella_mask
    labeled = measure.label(patella_mask.astype(np.uint8), connectivity=2)
    n_labels = int(labeled.max())
    if n_labels < 1:
        return patella_mask
    sizes = [(index, np.sum(labeled == index)) for index in range(1, n_labels + 1)]
    best_label = max(sizes, key=lambda x: x[1])[0]
    return (labeled == best_label).astype(np.uint8)


def smooth_edge_line(pts, method="savgol", w=5):
    if pts is None or len(pts) < 4:
        return pts
    pts = np.asarray(pts, dtype=float)
    w = min(w, len(pts) - 1) if len(pts) - 1 > 0 else 1
    if w < 2:
        return pts
    if method == "savgol":
        smooth_row_coordinates = savgol_filter(pts[:, 0], w, min(3, w - 1), mode="nearest")
        smooth_col_coordinates = savgol_filter(pts[:, 1], w, min(3, w - 1), mode="nearest")
        return np.column_stack([smooth_row_coordinates, smooth_col_coordinates])
    kernel = np.ones(min(w, len(pts))) / min(w, len(pts))
    return np.column_stack(
        [np.convolve(pts[:, 0], kernel, "same"), np.convolve(pts[:, 1], kernel, "same")]
    )


def trace_edges(patella_mask):
    raw = np.argwhere(patella_mask > 0)
    if len(raw) == 0:
        return (np.array([]).reshape(0, 2), np.array([]).reshape(0, 2))
    rows, cols = (raw[:, 0], raw[:, 1])
    min_row, max_row = (int(rows.min()), int(rows.max()))
    left_pts, right_pts = ([], [])
    for row in range(min_row, max_row + 1):
        col_in_row = cols[rows == row]
        if len(col_in_row) == 0:
            continue
        left_pts.append([float(row), float(col_in_row.min())])
        right_pts.append([float(row), float(col_in_row.max())])
    if not left_pts or not right_pts:
        median_col = float(np.median(cols))
        fallback = np.array([[min_row, median_col], [max_row, median_col]], dtype=float)
        return (fallback, fallback.copy())
    left = np.array(left_pts)
    right = np.array(right_pts)
    if len(left) >= 4:
        left = smooth_edge_line(left, method="savgol", w=5)
    if len(right) >= 4:
        right = smooth_edge_line(right, method="savgol", w=5)
    return (np.asarray(left, dtype=float), np.asarray(right, dtype=float))


def find_cdi_locations(line_coords, line_coordinates, patella_mask):
    if len(line_coords) == 0:
        return {}
    if len(line_coords) == 1:
        return {f"coord_{c}": tuple(line_coords[0]) for c in line_coordinates}
    validated_coordinates = sorted([c for c in line_coordinates if 0 <= c <= 500])
    if not validated_coordinates:
        return {}
    lengths_between_coordinates = np.sqrt(np.sum(np.diff(line_coords, axis=0) ** 2, axis=1))
    cumulative_lengths = np.concatenate([[0.0], np.cumsum(lengths_between_coordinates)])
    total_length = cumulative_lengths[-1]
    if total_length < 1e-09:
        return {f"coord_{c}": tuple(line_coords[0]) for c in validated_coordinates}
    result = {}
    for line_coordinate in validated_coordinates:
        normalized_value = np.clip(line_coordinate / 500, 0, 1)
        target_length = normalized_value * total_length
        target_index = max(
            0,
            min(
                np.searchsorted(cumulative_lengths, target_length, "right") - 1,
                len(line_coords) - 1,
            ),
        )
        if target_index + 1 >= len(line_coords):
            coordinate = line_coords[target_index].astype(float)
        else:
            a = np.clip(
                (target_length - cumulative_lengths[target_index])
                / (cumulative_lengths[target_index + 1] - cumulative_lengths[target_index] + 1e-09),
                0,
                1,
            )
            coordinate = (1 - a) * line_coords[target_index] + a * line_coords[target_index + 1]
        row_pixel_index, col_pixel_index = (int(round(coordinate[0])), int(round(coordinate[1])))
        if (
            0 <= row_pixel_index < patella_mask.shape[0]
            and 0 <= col_pixel_index < patella_mask.shape[1]
            and (patella_mask[row_pixel_index, col_pixel_index] > 0)
        ):
            pass
        else:
            d = np.sqrt(np.sum((line_coords - coordinate) ** 2, axis=1))
            coordinate = line_coords[int(np.argmin(d))].astype(float)
        result[f"coord_{line_coordinate}"] = (float(coordinate[0]), float(coordinate[1]))
    return result


def find_horizontal_intersection(bone_point, right_edge):
    """Find the point on the right edge that is directly horizontal (same row) to the bone line point.
    Interpolates across right_edge segments. Returns (distance, intersection_point) or (np.inf, None) if no intersection."""
    row_b, col_b = (float(bone_point[0]), float(bone_point[1]))
    if len(right_edge) == 0:
        return (np.inf, None)
    right_edge = np.asarray(right_edge, dtype=float)
    if len(right_edge) == 1:
        row, c = (right_edge[0, 0], right_edge[0, 1])
        if abs(row - row_b) < 1e-09:
            return (abs(c - col_b), (row, c))
        return (np.inf, None)
    best_col = np.inf
    best_point = None
    for index in range(len(right_edge) - 1):
        row1, column1 = (right_edge[index, 0], right_edge[index, 1])
        r2, c2 = (right_edge[index + 1, 0], right_edge[index + 1, 1])
        rlo, rhi = (min(row1, r2), max(row1, r2))
        if rlo <= row_b <= rhi and abs(rhi - rlo) > 1e-12:
            fraction = (row_b - row1) / (r2 - row1 + 1e-12)
            col_int = column1 + fraction * (c2 - column1)
            if col_int > col_b and col_int < best_col:
                best_col = col_int
                best_point = (row_b, col_int)
    if best_point is not None:
        return (float(best_col - col_b), best_point)
    row_dists = np.abs(right_edge[:, 0] - row_b)
    nearest_index = np.argmin(row_dists)
    row, c = (right_edge[nearest_index, 0], right_edge[nearest_index, 1])
    return (np.sqrt((row - row_b) ** 2 + (c - col_b) ** 2), (float(row), float(c)))


def compute_thickness_for_cdi_locations(key_locations, right_edge):
    result = {}
    if len(right_edge) == 0:
        return result
    for name, value in key_locations.items():
        minimum_distance, closest_right_point = find_horizontal_intersection(value, right_edge)
        if closest_right_point is not None:
            segment_to_right_point = np.array([value, closest_right_point])
        else:
            segment_to_right_point = np.array([]).reshape(0, 2)
        result[name] = {
            "thickness": minimum_distance,
            "cdi_location": value,
            "right_edge_point": closest_right_point,
            "path_coords": segment_to_right_point,
        }
    return result


def arc_length_pixels(pts):
    if pts is None or len(pts) < 2:
        return 0.0
    pts = np.asarray(pts, dtype=float)
    segs = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    return float(np.sum(segs))


def zoom_region(shape, patella_mask=None, size=120):
    H, W = shape[:2]
    if patella_mask is not None and np.sum(patella_mask) > 0:
        coords = np.argwhere(patella_mask > 0)
        cr, cc = (int(coords[:, 0].mean()), int(coords[:, 1].mean()))
    else:
        cr, cc = (H // 2, W // 2)
    half = size // 2
    row_start, row_end = (max(0, cr - half), min(H, cr + half))
    column_start, column_end = (max(0, cc - half), min(W, cc + half))
    return (row_start, row_end, column_start, column_end)
