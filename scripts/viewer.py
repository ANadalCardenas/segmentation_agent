# app/viewer.py
from typing import List, Dict, Tuple
import random

import cv2
import numpy as np


class Viewer:
    """
    Handles visualization:
    - draw detections on video frame
    - combine webcam and video frames side by side
    """

    def __init__(self, window_name: str = "Object Video Detector"):
        self.window_name = window_name
        self._class_colors: dict[str, Tuple[int, int, int]] = {}

    def _get_color_for_class(self, class_name: str) -> Tuple[int, int, int]:
        if class_name not in self._class_colors:
            # Random BGR color
            self._class_colors[class_name] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
        return self._class_colors[class_name]

    def draw_detections(
        self,
        frame_bgr: np.ndarray,
        detections: List[Dict],
    ) -> np.ndarray:
        output = frame_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            class_name = det["class_name"]
            color = self._get_color_for_class(class_name)

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}"
            cv2.putText(
                output, label, (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        return output

    def combine_frames(
        self,
        webcam_frame: np.ndarray,
        video_frame: np.ndarray,
    ) -> np.ndarray:
        # Resize webcam frame to match video_frame height
        h_v, w_v, _ = video_frame.shape
        h_w, w_w, _ = webcam_frame.shape

        scale = h_v / h_w
        new_w = int(w_w * scale)
        resized_webcam = cv2.resize(webcam_frame, (new_w, h_v))

        combined = np.hstack([resized_webcam, video_frame])
        return combined

    def show(self, frame: np.ndarray):
        cv2.imshow(self.window_name, frame)
