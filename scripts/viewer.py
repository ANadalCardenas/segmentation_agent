# app/viewer.py
from typing import List, Dict, Tuple, Callable, Optional
import random

import cv2
import numpy as np


class Viewer:
    """
    Handles visualization:
    - draw detections on video frame
    - combine webcam and video frames side by side
    - draw UI (push-to-talk button)
    """

    def __init__(self, window_name: str = "Object Video Detector"):
        self.window_name = window_name
        self._class_colors: dict[str, Tuple[int, int, int]] = {}

        # UI state
        self._button_pressed: bool = False
        self._button_callback: Optional[Callable[[bool], None]] = None
        self._button_rect: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)

        # Create window and register mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 450)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    # --------- Public API for main.py ---------

    def set_button_callback(self, callback: Callable[[bool], None]):
        """
        Called with True when button is pressed, False when released.
        """
        self._button_callback = callback

    # --------- Detection drawing ---------

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
                output,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
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

    # --------- UI drawing (button) ---------

    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """
        Adds a bottom bar with a 'Press while talking' button.
        Returns a new frame.
        """
        h, w = frame.shape[:2]
        ui_height = 60

        # New canvas: original frame + UI bar at bottom
        canvas = np.zeros((h + ui_height, w, 3), dtype=frame.dtype)
        canvas[:h, :, :] = frame

        # Button geometry
        btn_w, btn_h = 220, 40
        x1 = (w - btn_w) // 2
        y1 = h + (ui_height - btn_h) // 2
        x2 = x1 + btn_w
        y2 = y1 + btn_h
        self._button_rect = (x1, y1, x2, y2)

        # Button color: green (idle) or red (recording)
        if self._button_pressed:
            color = (0, 0, 255)  # BGR: red
        else:
            color = (0, 255, 0)  # BGR: green

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=-1)

        # Button label
        label = "Press while talking"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = x1 + (btn_w - text_size[0]) // 2
        text_y = y1 + (btn_h + text_size[1]) // 2
        cv2.putText(canvas, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        return canvas

    def show(self, frame: np.ndarray):
        cv2.imshow(self.window_name, frame)

    # --------- Mouse handling ---------

    def _on_mouse(self, event, x, y, flags, param):
        if self._button_rect is None:
            return

        x1, y1, x2, y2 = self._button_rect
        inside = (x1 <= x <= x2) and (y1 <= y <= y2)

        if event == cv2.EVENT_LBUTTONDOWN and inside:
            if not self._button_pressed:
                self._button_pressed = True
                if self._button_callback is not None:
                    self._button_callback(True)

        elif event == cv2.EVENT_LBUTTONUP:
            if self._button_pressed:
                self._button_pressed = False
                if self._button_callback is not None:
                    self._button_callback(False)
