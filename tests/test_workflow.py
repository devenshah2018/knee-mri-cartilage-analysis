"""Synthetic tests: no copyrighted inputs or network/model downloads required."""

import tempfile
import unittest
from dataclasses import replace
from datetime import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from skm_tea_cdi.analysis import (
    agreement,
    analyze,
    export_results,
    longitudinal,
    prepare_inputs,
    prepare_volume_inputs,
    scan_totals,
)
from skm_tea_cdi.annotations import (
    ExcelReferences,
    annotation_index,
    compartment_index,
    load_kl_grades,
    read_annotation,
)
from skm_tea_cdi.config import LINE_COORDINATES, SLICE_COORDINATES, Settings
from skm_tea_cdi.geometry import (
    arc_length_pixels,
    compute_thickness_for_cdi_locations,
    find_cdi_locations,
    find_patella_mask,
    trace_edges,
    zoom_region,
)
from skm_tea_cdi.inference import index_series, predict, scan_ids, visit_directory
from skm_tea_cdi.volume import TISSUES, segment_volume, volume_changes


def write_dicom(path, instance=1, frames=1, spacing=True):
    metadata = FileMetaDataset()
    metadata.TransferSyntaxUID = ExplicitVRLittleEndian
    metadata.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    metadata.MediaStorageSOPInstanceUID = generate_uid()
    dicom = FileDataset(str(path), {}, file_meta=metadata, preamble=b"\0" * 128)
    dicom.is_little_endian = True
    dicom.is_implicit_VR = False
    dicom.SOPClassUID = metadata.MediaStorageSOPClassUID
    dicom.SOPInstanceUID = metadata.MediaStorageSOPInstanceUID
    dicom.InstanceNumber = instance
    dicom.Rows = dicom.Columns = 16
    dicom.SamplesPerPixel = 1
    dicom.PhotometricInterpretation = "MONOCHROME2"
    dicom.BitsAllocated = dicom.BitsStored = 16
    dicom.HighBit = 15
    dicom.PixelRepresentation = 0
    if frames > 1:
        dicom.NumberOfFrames = frames
    if spacing:
        dicom.PixelSpacing = [0.5, 0.5]
        dicom.SliceThickness = 2
    dicom.PixelData = np.arange(256 * frames, dtype=np.uint16).tobytes()
    dicom.save_as(path, write_like_original=False)


def write_annotation(path, visit="v00", scan_id="123", slices=SLICE_COORDINATES, knee="LEFT"):
    lines = [scan_id, visit, knee, "6", "4", "0", "200", "Patella"]
    for slice_id in slices:
        lines.extend([str(slice_id), "{", "3.2", "3.13", "}"])
    lines.append("POI")
    for slice_id in slices:
        lines.extend([str(slice_id), "{"])
        lines.extend(f"3.{y} 7.{y}" for y in (4, 6, 8, 10))
        lines.append("}")
    path.write_text("\r\n".join(lines))


def workbook_rows(knee=2):
    return [
        {
            "id": 123,
            "time": time,
            "knee": knee,
            "Start_slice": 0,
            "End_slice": 200,
            **{f"BoneLength_medial_{i}": i * 10 for i in (1, 2, 3)},
        }
        for time in (0, 24)
    ]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / "dicoms"
        self.annotations = self.root / "annotations"
        self.annotations.mkdir()
        self.medial, self.lateral = [self.root / f"{name}.xlsx" for name in ("medial", "lateral")]
        for path in (self.medial, self.lateral):
            pd.DataFrame(workbook_rows()).to_excel(path, sheet_name="main", index=False)
        for visit in ("v00", "v03"):
            directory = self.images / "123" / visit
            directory.mkdir(parents=True)
            for slice_id in SLICE_COORDINATES:
                write_dicom(directory / str(slice_id), instance=slice_id + 1)
            write_annotation(self.annotations / f"sample_123_{visit}_Patella.txt", visit=visit)
        self.settings = Settings(
            dicom_dir=self.images,
            annotation_dir=self.annotations,
            medial_excel=self.medial,
            lateral_excel=self.lateral,
            kl_excel=None,
            output_dir=self.root / "results",
            scan_id="123",
            annotation_source="excel",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_annotation_dot_is_coordinate_separator(self):
        record = read_annotation(self.annotations / "sample_123_v00_Patella.txt")
        np.testing.assert_array_equal(record.bone[70], [[3, 2], [3, 13]])
        np.testing.assert_array_equal(record.poi[70][:2], [[3, 4], [7, 4]])
        self.assertEqual(record.knee, "left")

    def test_annotation_knee_suffix_overrides_header(self):
        path = self.annotations / "Cartilage_123_1_v01_OldMethod_Patella.txt"
        write_annotation(path, visit="v01b", scan_id="123_1", knee="LEFT")
        record = read_annotation(path)
        self.assertEqual((record.scan_id, record.visit, record.knee), ("123_1", "v01", "right"))

    def test_superseded_malformed_annotation_does_not_block_winner(self):
        old = self.annotations / "sample_123_v00_Patella.txt"
        old.write_text(old.read_text().replace("}", "", 1))
        winner = self.annotations / "z_new_123_v00b_Patella.txt"
        write_annotation(winner, visit="v00b")
        records = annotation_index(self.annotations, ["123"], ("v00", "v03"))
        self.assertEqual(records[("123", "v00")].path, winner)
        winner.write_text(winner.read_text().replace("}", "", 1))
        with self.assertRaises(ValueError):
            annotation_index(self.annotations, ["123"], ("v00", "v03"))

    def test_annotation_missing_brace_fails(self):
        path = self.annotations / "broken_Patella.txt"
        write_annotation(path)
        path.write_text(path.read_text().replace("}", "", 1))
        with self.assertRaises(ValueError):
            read_annotation(path)

    def test_latest_annotation_reissue_is_deterministic(self):
        path = self.annotations / "z_123_v00b_Patella.txt"
        write_annotation(path, visit="v00b")
        records = annotation_index(self.annotations, ["123"], ("v00", "v03"))
        self.assertEqual(records[("123", "v00")].path, path)

    def test_right_knee_workbook_order_is_reversed(self):
        self.assertEqual(compartment_index("left", 70), ("medial", 0))
        self.assertEqual(compartment_index("right", 70), ("lateral", 2))
        self.assertEqual(compartment_index("right", 130), ("medial", 0))

    def test_missing_workbook_column_fails_before_inference(self):
        pd.DataFrame(workbook_rows()).drop(columns="End_slice").to_excel(
            self.medial, sheet_name="main", index=False
        )
        with self.assertRaisesRegex(ValueError, "missing columns.*End_slice"):
            prepare_inputs(self.settings)

    def test_duplicate_workbook_rows_fail(self):
        rows = workbook_rows() + workbook_rows()[:1]
        pd.DataFrame(rows).to_excel(self.medial, sheet_name="main", index=False)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ExcelReferences(self.medial, self.lateral)

    def test_missing_visit_row_is_not_replaced_with_baseline(self):
        pd.DataFrame(workbook_rows()[:1]).to_excel(self.medial, sheet_name="main", index=False)
        with self.assertRaisesRegex(ValueError, "time=24"):
            prepare_inputs(self.settings)

    def test_kl_combined_workbook_time_encoding_and_knee_identity(self):
        path = self.root / "combined.xlsx"
        pd.DataFrame(
            {"ID": [123, 123], "side": ["1: Right", "2: Left"], "V00XRKL": [time(3, 3), time(4, 4)]}
        ).to_excel(path, index=False)
        self.assertEqual(load_kl_grades(path), {"123_1": 3, "123_2": 4})

    def test_kl_time_string_preserves_legacy_hour_encoding(self):
        path = self.root / "kl.xlsx"
        pd.DataFrame({"id": [123], "side": ["2: Left"], "V00XRKL": ["03:03:00"]}).to_excel(
            path, sheet_name="FORMING", index=False
        )
        self.assertEqual(load_kl_grades(path), {"123_2": 3})

    def test_kl_forming_workbook(self):
        path = self.root / "kl.xlsx"
        pd.DataFrame({"id": [123], "side": [2], "V00XRKL": [0]}).to_excel(
            path, sheet_name="FORMING", index=False
        )
        self.assertEqual(load_kl_grades(path), {"123_2": 0})

    def test_numeric_filename_index_has_priority_over_instance_number(self):
        series = index_series(self.images / "123" / "v00")
        self.assertEqual(list(series), list(SLICE_COORDINATES))

    def test_dcm_extension_uses_instance_number(self):
        directory = self.root / "named"
        directory.mkdir()
        write_dicom(directory / "slice.dcm", instance=7)
        self.assertEqual(list(index_series(directory)), [7])

    def test_duplicate_dicom_slice_ids_fail(self):
        directory = self.images / "123" / "v00"
        write_dicom(directory / "extra.dcm", instance=70)
        with self.assertRaisesRegex(ValueError, "Duplicate slice"):
            index_series(directory)

    def test_visit_resolution_prefers_exact_and_rejects_ambiguity(self):
        parent = self.images / "123"
        (parent / "v01a").mkdir()
        (parent / "v01b").mkdir()
        self.assertEqual(visit_directory(self.images, "123", "v00"), parent / "v00")
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            visit_directory(self.images, "123", "v01")

    def test_missing_selected_scan_fails(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            scan_ids(self.images, "999")

    def test_followup_uses_recorded_slices_even_when_formula_differs(self):
        slices = (71, 81, 91, 111, 121, 131)
        directory = self.images / "123" / "v03"
        for path in directory.iterdir():
            path.unlink()
        for slice_id in slices:
            write_dicom(directory / str(slice_id))
        write_annotation(
            self.annotations / "sample_123_v03_Patella.txt", visit="v03", slices=slices
        )
        inputs = prepare_inputs(self.settings)
        self.assertEqual(list(inputs.selections[("123", "v03")].values()), list(slices))

    def test_txt_only_needs_no_spreadsheets(self):
        settings = replace(
            self.settings, annotation_source="txt", medial_excel=None, lateral_excel=None
        )
        inputs = prepare_inputs(settings)
        self.assertEqual(len(inputs.inventory), 2)
        self.assertTrue(np.isnan(inputs.references.bone_length("123", 0, "left", 70)))

    def test_volume_only_preflight_needs_no_manual_data(self):
        settings = replace(
            self.settings,
            annotation_dir=self.root / "does-not-exist",
            annotation_source="txt",
            medial_excel=None,
            lateral_excel=None,
        )
        inputs = prepare_volume_inputs(settings)
        self.assertEqual(len(inputs.series), 2)
        self.assertFalse(inputs.annotations)
        self.assertEqual(set(inputs.knees.values()), {"unknown"})

    def test_volume_only_preflight_uses_knee_qualified_ids(self):
        (self.images / "123").rename(self.images / "123_1")
        inputs = prepare_volume_inputs(replace(self.settings, scan_id="123_1"))
        self.assertEqual(set(inputs.knees.values()), {"right"})

    def test_missing_manual_pair_fails_preflight(self):
        path = self.annotations / "sample_123_v00_Patella.txt"
        path.write_text(path.read_text().replace("3.4 7.4", "3.4", 1))
        with self.assertRaisesRegex(ValueError, "expected 8 POI"):
            prepare_inputs(self.settings)

    def test_empty_masks_remain_missing_and_inference_is_not_repeated(self):
        inputs = prepare_inputs(self.settings)
        with patch(
            "skm_tea_cdi.analysis.predict",
            return_value=(np.zeros((16, 16)), np.zeros((16, 16), dtype=np.int8)),
        ) as predictor:
            result = analyze(self.settings, inputs, object(), progress=lambda _: None)
        self.assertEqual(predictor.call_count, 12)
        self.assertEqual(len(result.points), 48)
        self.assertTrue(result.points.calculated_px.isna().all())
        self.assertTrue((result.points.manual_px == 4).all())
        self.assertFalse(scan_totals(result.points).complete.any())
        self.assertTrue(longitudinal(result.points, "v00", "v03").calculated_delta_px.isna().all())

    def test_analysis_export_and_pairing_preserve_point_identity(self):
        inputs = prepare_inputs(self.settings)
        labels = np.zeros((16, 16), dtype=np.int8)
        labels[2:14, 3:8] = 1
        with patch("skm_tea_cdi.analysis.predict", return_value=(labels, labels)):
            result = analyze(self.settings, inputs, object(), progress=lambda _: None)
        tables = export_results(self.settings, inputs, result)
        self.assertEqual(len(tables["longitudinal_changes"]), 24)
        self.assertEqual(tables["scan_total_changes"].calculated_delta_px.iloc[0], 0)
        self.assertEqual(len(tables["scan_total_agreement"]), 3)
        np.testing.assert_allclose(tables["longitudinal_changes"].calculated_delta_px, 0)
        self.assertTrue(tables["scan_totals"].complete.all())
        self.assertTrue((self.settings.output_dir / "analysis.xlsx").is_file())
        self.assertEqual(
            len(pd.ExcelFile(self.settings.output_dir / "analysis.xlsx").sheet_names), len(tables)
        )
        altered = result.points.drop(index=0)
        self.assertFalse(scan_totals(altered).set_index("visit").loc["v00", "complete"])

    def test_constant_image_rejected_before_model_call(self):
        with self.assertRaisesRegex(ValueError, "constant"):
            predict(object(), np.ones((16, 16), dtype=np.float32))

    def test_multiframe_volume_and_spacing(self):
        path = self.root / "multiframe.dcm"
        write_dicom(path, frames=3)
        labels = np.ones((16, 16), dtype=np.int8)
        with patch("skm_tea_cdi.volume.predict", return_value=(labels, labels)) as predictor:
            volume, sizes = segment_volume(object(), [path])
        self.assertEqual(volume.shape, (3, 16, 16))
        self.assertEqual(sizes, [0.5, 0.5, 0.5])
        self.assertEqual(predictor.call_count, 3)

    def test_volume_missing_spacing_does_not_fabricate_mm3(self):
        path = self.root / "no-spacing.dcm"
        write_dicom(path, spacing=False)
        labels = np.ones((16, 16), dtype=np.int8)
        with patch("skm_tea_cdi.volume.predict", return_value=(labels, labels)):
            _, sizes = segment_volume(object(), [path])
        self.assertIsNone(sizes)


class GeometryAndStatisticsTests(unittest.TestCase):
    def test_zoom_region_has_nonempty_clipped_bounds(self):
        self.assertEqual(zoom_region((256, 256), size=180), (38, 218, 38, 218))
        mask = np.zeros((20, 20))
        mask[:3, :3] = 1
        self.assertEqual(zoom_region(mask.shape, mask, size=10), (0, 6, 0, 6))

    def test_largest_patellar_component_only(self):
        labels = np.zeros((20, 20), dtype=np.int8)
        labels[2:10, 3:8] = 1
        labels[15, 15] = 1
        labels[10:13, 10:13] = 2
        mask = find_patella_mask(labels)
        self.assertEqual(int(mask.sum()), 40)
        self.assertEqual(mask[15, 15], 0)

    def test_rectangular_mask_thickness_and_empty_mask(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:18, 3:9] = 1
        left, right = trace_edges(mask)
        locations = find_cdi_locations(left, LINE_COORDINATES, mask)
        thicknesses = compute_thickness_for_cdi_locations(locations, right)
        np.testing.assert_allclose([v["thickness"] for v in thicknesses.values()], 5, atol=1e-8)
        self.assertEqual(trace_edges(np.zeros_like(mask))[0].shape, (0, 2))
        self.assertEqual(find_cdi_locations(np.empty((0, 2)), LINE_COORDINATES, mask), {})
        self.assertEqual(arc_length_pixels([[0, 0], [3, 4]]), 5)

    def test_volume_changes_keep_tissues_and_missingness(self):
        rows = []
        for scan_id in ("123", "456"):
            for visit, value in (("v00", 1000), ("v03", 800)):
                rows.append(
                    {
                        "scan_id": scan_id,
                        "visit": visit,
                        "kl_grade": 2,
                        **{
                            f"{name}_mm3": value if scan_id == "123" else np.nan
                            for name in [*TISSUES.values(), "whole_cartilage"]
                        },
                    }
                )
        changes, by_kl = volume_changes(pd.DataFrame(rows), "v00", "v03")
        self.assertAlmostEqual(changes.whole_cartilage_delta_cm3.iloc[0], -0.2)
        self.assertAlmostEqual(changes.whole_cartilage_pct_change.iloc[0], -20)
        self.assertTrue(np.isnan(changes.whole_cartilage_pct_change.iloc[1]))
        self.assertEqual(len(by_kl), 5)
        self.assertTrue((by_kl.valid_pairs == 1).all())

    def test_statistics_filter_pairs_and_use_sample_limits(self):
        result = agreement([1, 2, np.nan, 9], [2, 4, 5, np.inf])
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["bias_px"], 1.5)
        self.assertAlmostEqual(result["loa_upper_px"], 1.5 + 1.96 * np.sqrt(0.5))
        self.assertTrue(np.isnan(agreement([1], [2])["pearson_r"]))
        self.assertTrue(np.isnan(agreement([1, 1], [2, 3])["pearson_r"]))


if __name__ == "__main__":
    unittest.main()
