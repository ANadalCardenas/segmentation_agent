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
    - draw UI (push-to-talk button + side panel)
    """

    def __init__(self, window_name: str = "Object Video Detector"):
        self.window_name = window_name
        self._class_colors: dict[str, Tuple[int, int, int]] = {}

        # UI state
        self._button_pressed: bool = False
        self._button_callback: Optional[Callable[[bool], None]] = None
        self._button_rect: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)

        # Create window + mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1600, 900)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    # --------- Public API for main.py ---------

    def set_button_callback(self, callback: Callable[[bool], None]):
        """
        Called with True when button is toggled ON, False when toggled OFF.
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

    # --------- Side panel drawing ---------

    def add_side_panel(
        self,
        frame: np.ndarray,
        active_classes: List[str],
        unsupported_classes: List[str],
    ) -> np.ndarray:
        """
        Adds a thin panel on the right showing:
          - Active detected classes
          - Unsupported requested objects
        """
        h, w, _ = frame.shape
        panel_w = max(220, w // 6)

        panel = np.zeros((h, panel_w, 3), dtype=frame.dtype)
        panel[:] = (40, 40, 40)  # dark grey background

        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_scale = 0.6
        item_scale = 0.5
        title_thickness = 2
        item_thickness = 1

        x_margin = 10
        y = 25

        # Helper to draw a block (title + list)
        def draw_block(title: str, items: List[str], y_start: int, color_title, color_item) -> int:
            y_cur = y_start
            cv2.putText(panel, title, (x_margin, y_cur), font, title_scale, color_title, title_thickness)
            y_cur += 20
            if not items:
                cv2.putText(panel, "-", (x_margin, y_cur), font, item_scale, color_item, item_thickness)
                y_cur += 18
            else:
                for item in sorted(items):
                    cv2.putText(panel, f"- {item}", (x_margin, y_cur), font, item_scale, color_item, item_thickness)
                    y_cur += 18
            y_cur += 10
            return y_cur

        # Detecting block
        y = draw_block(
            "Detecting:",
            list(active_classes),
            y,
            color_title=(0, 255, 0),        # green
            color_item=(200, 255, 200),
        )

        # Unsupported block
        y = draw_block(
            "Unsupported:",
            list(unsupported_classes),
            y,
            color_title=(0, 0, 255),        # red
            color_item=(200, 200, 255),
        )

        # Combine original frame + panel
        combined = np.hstack([frame, panel])
        return combined

    # --------- Bottom button UI ---------

    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """
        Adds a bottom bar with a 'Press while talking' button.
        Returns a new frame.
        """
        h, w = frame.shape[:2]
        ui_height = 60

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
            color = (0, 0, 255)  # red
        else:
            color = (0, 255, 0)  # green

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

    # --------- Mouse handling (toggle button) ---------

    def _on_mouse(self, event, x, y, flags, param):
        if self._button_rect is None:
            return

        x1, y1, x2, y2 = self._button_rect
        inside = (x1 <= x <= x2) and (y1 <= y <= y2)

        # Toggle on click inside the button
        if event == cv2.EVENT_LBUTTONDOWN and inside:
            self._button_pressed = not self._button_pressed
            if self._button_callback is not None:
                self._button_callback(self._button_pressed)
