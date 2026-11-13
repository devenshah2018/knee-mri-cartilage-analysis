"""Read manual annotations and Excel references without silently guessing values."""

import re
from dataclasses import dataclass
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SLICE_COORDINATES


def normalize_visit(value: str) -> str:
    """Normalize annotation reissues such as v00b to their visit token."""
    match = re.fullmatch(r"(v\d+)[a-z]*", value.strip().lower())
    if match is None:
        raise ValueError(f"Invalid visit: {value!r}")
    return match[1]


@dataclass(frozen=True)
class Annotation:
    path: Path
    scan_id: str
    visit: str
    knee: str
    start_slice: int
    end_slice: int
    bone: dict
    poi: dict


def _section(lines, name):
    """Parse integer x.y tokens: the dot separates x and y, not decimals."""
    if name not in lines:
        return {}
    result = {}
    index = lines.index(name) + 1
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line or line.startswith("#"):
            continue
        if line in {"Patella", "POI"}:
            break
        if not line.isdigit():
            raise ValueError(f"Expected slice number in {name}, got {line!r}")
        slice_id = int(line)
        while index < len(lines) and not lines[index]:
            index += 1
        if index >= len(lines) or lines[index] != "{":
            raise ValueError(f"Missing opening brace for {name} slice {slice_id}")
        index += 1
        points = []
        while index < len(lines) and lines[index] != "}":
            content = lines[index].split("#", 1)[0]
            for token in content.replace(",", " ").split():
                match = re.fullmatch(r"(-?\d+)\.(-?\d+)", token)
                if match is None:
                    raise ValueError(f"Invalid x.y coordinate {token!r}")
                points.append(tuple(map(float, match.groups())))
            index += 1
        if index >= len(lines):
            raise ValueError(f"Missing closing brace for {name} slice {slice_id}")
        if slice_id in result:
            raise ValueError(f"Duplicate {name} slice {slice_id}")
        result[slice_id] = np.asarray(points, dtype=float).reshape(-1, 2)
        index += 1
    return result


def read_annotation(path: Path) -> Annotation:
    """Read Ming2016 or Cartilage annotations, including suffixed knee IDs."""
    path = Path(path)
    lines = [line.strip() for line in path.read_text().splitlines()]
    if len(lines) < 8:
        raise ValueError(f"{path.name}: incomplete seven-line annotation header")
    scan_id, visit, knee = lines[:3]
    if not re.fullmatch(r"\d+(?:_[12])?", scan_id):
        raise ValueError(f"{path.name}: invalid scan ID {scan_id!r}")
    # In the Cartilage set the knee suffix is authoritative, as in the old TXT runner.
    knee_match = re.search(r"_([12])_[vV]\d+", path.name)
    if knee_match:
        knee = {"1": "RIGHT", "2": "LEFT"}[knee_match[1]]
    knee = knee.lower()
    if knee not in {"left", "right"}:
        raise ValueError(f"{path.name}: knee must be LEFT or RIGHT")
    start, end = map(int, lines[5:7])
    if start > end:
        raise ValueError(f"{path.name}: reversed slice range {start}..{end}")
    return Annotation(
        path,
        scan_id,
        normalize_visit(visit),
        knee,
        start,
        end,
        _section(lines, "Patella"),
        _section(lines, "POI"),
    )


def annotation_index(directory: Path, scan_ids, visits):
    """Index selected records; prefer the latest mtime, then name, for reissues."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {directory}")
    candidates = {}
    selected = set(map(str, scan_ids))
    for path in sorted(directory.glob("*Patella*.txt")):
        # Inspect the identity first so unrelated experiments do not block a run.
        with path.open() as stream:
            identity = stream.readline().strip()
            visit = stream.readline().strip()
        if identity not in selected or normalize_visit(visit) not in visits:
            continue
        key = (identity, normalize_visit(visit))
        old = candidates.get(key)
        if old is None or (path.stat().st_mtime_ns, path.name) > (old.stat().st_mtime_ns, old.name):
            candidates[key] = path
    return {key: read_annotation(path) for key, path in candidates.items()}


def compartment_index(knee, coordinate):
    """Map ascending image coordinates to workbook compartment and column index.

    Right-knee Excel measurements run in reverse slice order. This convention
    matches the TXT-to-Excel experiment and corrects the original notebook's map.
    """
    index = SLICE_COORDINATES.index(coordinate)
    if knee == "left":
        return ("medial" if index < 3 else "lateral", index % 3)
    if knee == "right":
        return ("lateral" if index < 3 else "medial", 2 - index % 3)
    raise ValueError(f"Unknown knee side: {knee!r}")


class ExcelReferences:
    """Read each workbook once. Both legacy files use BoneLength_medial_* names."""

    def __init__(self, medial=None, lateral=None):
        self.tables = {}
        for compartment, path in (("medial", medial), ("lateral", lateral)):
            if path is None:
                continue
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(f"{compartment.title()} workbook not found: {path}")
            table = pd.read_excel(path, sheet_name="main", engine=_engine(path))
            required = {"id", "time", "knee", "Start_slice", "End_slice"}
            required |= {f"BoneLength_medial_{i}" for i in (1, 2, 3)}
            missing = required - set(table.columns)
            if missing:
                raise ValueError(f"{path.name}, sheet main: missing columns {sorted(missing)}")
            table["id"] = table["id"].astype(str).str.replace(r"\.0$", "", regex=True)
            if table.duplicated(["id", "time"]).any():
                raise ValueError(f"{path.name}: duplicate (id, time); use knee-qualified IDs")
            self.tables[compartment] = table.set_index(["id", "time"])

    def row(self, compartment, scan_id, visit_time):
        table = self.tables.get(compartment)
        if table is None or (str(scan_id), visit_time) not in table.index:
            raise ValueError(f"No {compartment} Excel row for {scan_id}, time={visit_time}")
        return table.loc[(str(scan_id), visit_time)]

    def knee(self, scan_id, visit_time):
        sides = {self.row(c, scan_id, visit_time)["knee"] for c in self.tables}
        if len(sides) != 1 or not sides <= {1, 2}:
            raise ValueError(f"Conflicting or invalid Excel knee sides for {scan_id}")
        return {1: "right", 2: "left"}[sides.pop()]

    def bone_length(self, scan_id, visit_time, knee, coordinate):
        if not self.tables:
            return np.nan
        compartment, index = compartment_index(knee, coordinate)
        row = self.row(compartment, scan_id, visit_time)
        return float(row[f"BoneLength_medial_{index + 1}"])


def _engine(path):
    return "xlrd" if Path(path).suffix.lower() == ".xls" else "openpyxl"


def _excel_number(value):
    """Decode legacy clinical integers, including times and '1: Right' labels.

    In combined.xlsx KL3 is actually stored as 03:03:00; the leading hour, not
    the minute field, is the grade. Side labels use the same leading-token rule.
    """
    if isinstance(value, time):
        return value.hour
    if isinstance(value, str):
        value = value.strip().split(":", 1)[0]
    return pd.to_numeric(value, errors="coerce")


def kl_grade_for(grades, scan_id, knee):
    """Resolve both bare Phase 3 identifiers and knee-qualified clinical IDs."""
    grade = grades.get(str(scan_id))
    if grade is None and "_" not in str(scan_id) and knee in {"left", "right"}:
        grade = grades.get(f"{scan_id}_{1 if knee == 'right' else 2}")
    return np.nan if grade is None else grade


def load_kl_grades(path: Path | None):
    """Support FORMING(id, side, V00XRKL) and combined.xlsx(ID, side, V00XRKL)."""
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"KL workbook not found: {path}; set kl_excel=None to skip")
    with pd.ExcelFile(path, engine=_engine(path)) as book:
        sheet = "FORMING" if "FORMING" in book.sheet_names else book.sheet_names[0]
        table = pd.read_excel(book, sheet_name=sheet)
    id_column = "id" if "id" in table.columns else "ID"
    required = {id_column, "V00XRKL"}
    if required - set(table.columns):
        raise ValueError(f"{path.name}: expected id/ID and V00XRKL columns")
    grades = {}
    for _, row in table.iterrows():
        grade = _excel_number(row["V00XRKL"])
        if pd.isna(grade) or grade not in range(5):
            continue
        scan_id = str(row[id_column]).removesuffix(".0")
        side = _excel_number(row.get("side", np.nan))
        key = f"{scan_id}_{int(side)}" if side in (1, 2) else scan_id
        if key in grades and grades[key] != int(grade):
            raise ValueError(f"Conflicting baseline KL grades for {key}")
        grades[key] = int(grade)
    return grades
