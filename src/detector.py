"""
Core YOLO detection utilities.
Separates model loading and inference from the Streamlit UI for cleaner code.
"""

from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple, List, Optional
import pandas as pd


class YOLODetector:
    """Wrapper around Ultralytics YOLOv8 for easy inference."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.25):
        """
        Initialize the detector.

        Args:
            model_name: Path to .pt model or official name (yolov8n.pt, yolov8s.pt, etc.)
            confidence: Default confidence threshold
        """
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.class_names = self.model.names

    def predict(
        self,
        source: Union[str, np.ndarray, Image.Image],
        conf: Optional[float] = None,
        classes: Optional[List[int]] = None,
    ):
        """
        Run inference on an image, video path, or numpy array.

        Returns the Ultralytics Results object.
        """
        conf = conf if conf is not None else self.confidence
        results = self.model.predict(
            source=source,
            conf=conf,
            classes=classes,
            verbose=False,
        )
        return results

    def annotate(self, results) -> np.ndarray:
        """Return the annotated image (BGR) from the first result."""
        return results[0].plot()

    def get_detections_df(self, results) -> pd.DataFrame:
        """Convert detections to a clean pandas DataFrame."""
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return pd.DataFrame(columns=["class", "confidence", "x1", "y1", "x2", "y2"])

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
        return pd.DataFrame(data)

    def get_available_classes(self) -> dict:
        """Return the class name mapping."""
        return self.class_names
