"""
Core YOLO detection utilities.
Separates model loading and inference from the Streamlit UI for cleaner code.
"""

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image

DETECTION_COLUMNS = ["class", "confidence", "x1", "y1", "x2", "y2"]


class YOLODetector:
    """Wrapper around Ultralytics YOLOv8 for easy inference."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.25):
        """
        Initialize the detector.

        Args:
            model_name: Path to .pt model or official name (yolov8n.pt, yolov8s.pt, etc.)
            confidence: Default confidence threshold
        """
        # Imported lazily so the helpers in this module (and the unit tests)
        # don't require torch/ultralytics to be installed.
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.confidence = confidence
        self.class_names = self.model.names

    def predict(
        self,
        source: Union[str, np.ndarray, Image.Image],
        conf: Optional[float] = None,
        classes: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Run inference on an image, video path, or numpy array.

        Extra keyword arguments (e.g. ``save``, ``project``, ``name``) are
        passed straight through to ``YOLO.predict``.

        Returns the Ultralytics Results object.
        """
        conf = conf if conf is not None else self.confidence
        kwargs.setdefault("verbose", False)
        results = self.model.predict(
            source=source,
            conf=conf,
            classes=classes,
            **kwargs,
        )
        return results

    def annotate(self, results) -> np.ndarray:
        """Return the annotated image (BGR) from the first result."""
        return results[0].plot()

    def annotate_rgb(self, results) -> np.ndarray:
        """Return the annotated image (RGB) from the first result."""
        return self.annotate(results)[:, :, ::-1].copy()

    def get_detections_df(self, results) -> pd.DataFrame:
        """Convert detections to a clean pandas DataFrame."""
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return pd.DataFrame(columns=DETECTION_COLUMNS)

        data = []
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            data.append({
                "class": self.class_names[cls_id],
                "confidence": round(conf, 3),
                "x1": round(xyxy[0], 1),
                "y1": round(xyxy[1], 1),
                "x2": round(xyxy[2], 1),
                "y2": round(xyxy[3], 1),
            })
        return pd.DataFrame(data, columns=DETECTION_COLUMNS)

    def get_available_classes(self) -> dict:
        """Return the class name mapping."""
        return self.class_names


def find_output_video(output_dir: Union[str, Path]) -> Optional[Path]:
    """Return the processed video in ``output_dir``, preferring ``.mp4`` over ``.avi``.

    Browsers can generally play mp4 but not avi, so mp4 is preferred when both exist.
    Returns None if the directory doesn't exist or contains no video.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None
    for suffix in (".mp4", ".avi"):
        matches = sorted(output_dir.glob(f"*{suffix}"))
        if matches:
            return matches[0]
    return None


def convert_to_mp4(video_path: Union[str, Path]) -> Optional[Path]:
    """Best-effort conversion of a video to browser-friendly H.264 mp4 using ffmpeg.

    Returns the new path, or None if ffmpeg is unavailable or the conversion fails.
    """
    video_path = Path(video_path)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    target = video_path.with_name(video_path.stem + "_h264.mp4")
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(video_path), "-vcodec", "libx264",
             "-pix_fmt", "yuv420p", str(target)],
            check=True, capture_output=True, timeout=600,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return target if target.exists() else None
