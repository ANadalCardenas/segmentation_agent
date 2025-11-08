
from typing import List, Dict, Any, Set, Tuple

import torch
import numpy as np


class ObjectDetector:
    def __init__(self, model_name: str = "yolov5s"):
        # Downloads weights on first run; uses CUDA if available.
        self.model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=True)
        self.model.eval()

        self.class_names = self.model.names  # dict: {class_id: class_name}

    def get_class_names(self) -> List[str]:
        """
        Returns YOLOv5 class names in a stable order (id order).
        """
        return [self.class_names[i] for i in range(len(self.class_names))]

    def detect(
        self,
        frame_bgr: np.ndarray,
        allowed_classes: Set[str] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Run detection on a single BGR frame.

        Returns a list of detections: dicts with keys
        { 'class_id', 'class_name', 'conf', 'bbox' = (x1, y1, x2, y2) }.
        If allowed_classes is not None, only those class names are kept.
        """
        results = self.model(frame_bgr)  # inference
        detections: List[Dict[str, Any]] = []

        # results.xyxy[0]: [N, 6] tensor (x1, y1, x2, y2, conf, cls)
        for *xyxy, conf, cls_id in results.xyxy[0].tolist():
            cls_id = int(cls_id)
            class_name = self.class_names[cls_id].lower()
            if allowed_classes is not None and class_name not in allowed_classes:
                continue

            detections.append({
                "class_id": cls_id,
                "class_name": class_name,
                "conf": float(conf),
                "bbox": tuple(xyxy),  # (x1, y1, x2, y2)
            })

        return detections
