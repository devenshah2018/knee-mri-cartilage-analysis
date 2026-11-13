"""Paths and analysis choices shared by the notebook and library."""

from dataclasses import dataclass
from pathlib import Path

MODEL_CONFIG_URL = "https://huggingface.co/arjundd/skm-tea-models/raw/main/neurips2021/segmentation-models/U-Net_E1/config.yaml"
MODEL_WEIGHTS_URL = "https://huggingface.co/arjundd/skm-tea-models/resolve/main/neurips2021/segmentation-models/U-Net_E1/model.ckpt"
SLICE_COORDINATES = (70, 80, 90, 110, 120, 130)
LINE_COORDINATES = (180, 230, 280, 330)


@dataclass(frozen=True)
class Settings:
    """Explicit local inputs; paths are relative to the notebook's working directory.

    Set scan_id=None to process every scan. In TXT mode the annotation records
    supply the actual slice IDs; Excel remains an independent reference.
    """

    dicom_dir: Path = Path("datasets/dess-phase-3")
    annotation_dir: Path = Path("datasets/u01-phase-3")
    medial_excel: Path | None = Path("datasets/clinical/Phase_3_Medial_Patella.xls")
    lateral_excel: Path | None = Path("datasets/clinical/Phase_3_Lateral_Patella.xls")
    kl_excel: Path | None = Path("datasets/clinical/KL for Ming May 12 2016.xls")
    output_dir: Path = Path("results")
    model_config: str = "models/config.yaml"
    model_weights: str = "models/model.ckpt"
    scan_id: str | None = "9493245"
    baseline_visit: str = "v00"
    followup_visit: str = "v03"
    baseline_time: int = 0
    followup_time: int = 24
    annotation_source: str = "txt"
    device: str = "auto"

    def __post_init__(self):
        if (self.medial_excel is None) != (self.lateral_excel is None):
            raise ValueError("Supply both patella workbooks, or set both to None")
        if self.annotation_source not in {"excel", "txt"}:
            raise ValueError("annotation_source must be 'excel' or 'txt'")
        if self.baseline_visit == self.followup_visit:
            raise ValueError("Baseline and follow-up visits must differ")
        if self.baseline_time == self.followup_time:
            raise ValueError("Baseline and follow-up Excel times must differ")
        if self.annotation_source == "excel" and (
            self.medial_excel is None or self.lateral_excel is None
        ):
            raise ValueError("Excel mode requires both medial_excel and lateral_excel")

    @property
    def visits(self):
        return (self.baseline_visit, self.followup_visit)

    def time_for(self, visit):
        return dict(zip(self.visits, (self.baseline_time, self.followup_time)))[visit]
