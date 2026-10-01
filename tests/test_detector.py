"""Lightweight unit tests that don't need torch/ultralytics or model weights."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detector import DETECTION_COLUMNS, YOLODetector, find_output_video  # noqa: E402


class _Tensor:
    """Minimal stand-in for a torch tensor row."""

    def __init__(self, values):
        self._v = list(values)

    def __getitem__(self, i):
        return self._v[i]

    def tolist(self):
        return list(self._v)


class _Box:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [_Tensor(xyxy)]


class _Result:
    def __init__(self, boxes, image=None):
        self.boxes = boxes
        self._image = image

    def plot(self):
        return self._image


def _detector():
    # Bypass __init__ so no model/weights are needed.
    det = YOLODetector.__new__(YOLODetector)
    det.class_names = {0: "person", 1: "car"}
    det.confidence = 0.25
    return det


def test_get_detections_df_empty():
    df = _detector().get_detections_df([_Result([])])
    assert list(df.columns) == DETECTION_COLUMNS
    assert df.empty


def test_get_detections_df_rows():
    boxes = [_Box(0, 0.9123, [1.04, 2.0, 30.0, 40.0]), _Box(1, 0.5, [0, 0, 5, 5])]
    df = _detector().get_detections_df([_Result(boxes)])
    assert list(df["class"]) == ["person", "car"]
    assert df.loc[0, "confidence"] == 0.912
    assert df.loc[0, "x1"] == 1.0


def test_annotate_rgb_swaps_channels():
    bgr = np.zeros((1, 1, 3), dtype=np.uint8)
    bgr[0, 0] = [255, 0, 0]  # blue in BGR
    rgb = _detector().annotate_rgb([_Result([], image=bgr)])
    assert tuple(rgb[0, 0]) == (0, 0, 255)


def test_find_output_video_prefers_mp4(tmp_path):
    (tmp_path / "a.avi").write_bytes(b"x")
    assert find_output_video(tmp_path).suffix == ".avi"
    (tmp_path / "b.mp4").write_bytes(b"x")
    assert find_output_video(tmp_path).suffix == ".mp4"


def test_find_output_video_missing_or_empty(tmp_path):
    assert find_output_video(tmp_path / "nope") is None
    assert find_output_video(tmp_path) is None
