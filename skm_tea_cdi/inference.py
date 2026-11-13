"""DICOM indexing and SKM-TEA inference, with one model load per notebook run."""

import re
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch


def scan_ids(directory: Path, selected: str | None):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {directory}")
    available = sorted(
        p.name for p in directory.iterdir() if p.is_dir() and re.fullmatch(r"\d+(?:_[12])?", p.name)
    )
    if selected is not None:
        if str(selected) not in available:
            raise ValueError(f"Scan {selected!r} not found in {directory}")
        return [str(selected)]
    if not available:
        raise ValueError(f"No scan directories in {directory}")
    return available


def visit_directory(root, scan_id, visit):
    parent = Path(root) / str(scan_id)
    exact = parent / visit
    if exact.is_dir():
        return exact
    alternatives = sorted(
        p
        for p in parent.glob(f"{visit}*")
        if p.is_dir() and re.fullmatch(re.escape(visit) + r"[a-z]+", p.name)
    )
    if len(alternatives) == 1:
        return alternatives[0]
    if not alternatives:
        raise FileNotFoundError(f"Missing DICOM visit: {exact}")
    raise ValueError(f"Ambiguous DICOM visit {visit} for {scan_id}: {alternatives}")


def index_series(directory: Path):
    """Map numeric slice filenames, or InstanceNumber for .dcm files, to paths.

    Numeric filenames retain the historical annotation indexing convention.
    Non-DICOM auxiliary files are ignored. Duplicate IDs are rejected.
    """
    series = {}
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if not (
            path.name.isdigit() or path.suffix.lower() == ".dcm" or pydicom.misc.is_dicom(path)
        ):
            continue
        header = pydicom.dcmread(path, stop_before_pixels=True)
        slice_id = int(path.name) if path.name.isdigit() else int(header.InstanceNumber)
        if slice_id in series:
            raise ValueError(f"Duplicate slice ID {slice_id} in {directory}")
        series[slice_id] = path
    if not series:
        raise ValueError(f"No DICOM slices found in {directory}")
    return dict(sorted(series.items()))


def read_slice(path):
    image = pydicom.dcmread(path).pixel_array.astype(np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale DICOM slice, got {image.shape}: {path}")
    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite DICOM pixels: {path}")
    return image


def enhance_slice(image):
    """Original notebook preprocessing: normalize, CLAHE, blur, then erode."""
    image = np.asarray(image, dtype=np.float32)
    image = 255 * (image - image.min()) / (np.ptp(image) + 1e-8)
    image = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(image.astype(np.uint8))
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return cv2.erode(image, np.ones((3, 3), np.uint8))


def resolve_device(device="auto"):
    return (
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else ("cpu" if device == "auto" else device)
    )


def load_model(config, weights, device="auto"):
    """Load the published SKM-TEA model without globally patching torch.load.

    Inputs may be local paths or the published model URLs. Checkpoints use the
    upstream pickle format and must come from a trusted source.
    """
    import skm_tea as st
    from skm_tea.utils.env import get_path_manager

    for label, value in (("configuration", config), ("weights", weights)):
        if not str(value).startswith(("https://", "http://")) and not Path(value).is_file():
            raise FileNotFoundError(f"Model {label} not found: {value}")
    model = st.get_model_from_zoo(str(config), weights_path="")
    checkpoint_path = get_path_manager().get_local_path(str(weights))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
    return model.eval().to(resolve_device(device))


def predict(model, image, device="auto", enhance=True):
    """Return enhanced/raw model input and categorical labels 0..4."""
    import meddlr.ops as ops

    processed = enhance_slice(image) if enhance else np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(processed).float()
    # torch.std's sample standard deviation preserves the original notebook math.
    standard_deviation = tensor.std(unbiased=enhance)
    if not torch.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("Cannot segment a constant or non-finite image")
    tensor = (tensor - tensor.mean()) / standard_deviation
    with torch.inference_mode():
        logits = model({"image": tensor[None, None].to(resolve_device(device))})["sem_seg_logits"]
    labels = ops.pred_to_categorical(logits, activation="sigmoid").squeeze(0)
    return processed, labels.cpu().numpy().astype(np.int8)
