# app/main.py
import argparse
from typing import Set

import cv2

from detector import ObjectDetector
from viewer import Viewer
from commands import CommandParser
from speech_to_text import ConversationManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="/workspace/segmentation_agent/media/video.mp4")
    parser.add_argument("--model", default="yolov5s")
    parser.add_argument("--mic", action="store_true", help="Enable microphone-based commands")
    args = parser.parse_args()

    # 1) Init detector & parser
    detector = ObjectDetector(model_name=args.model)
    class_names = detector.get_class_names()
    cmd_parser = CommandParser(class_names)

    # 2) Conversation manager (background thread)
    conv_manager = ConversationManager(parser=cmd_parser)
    if args.mic:
        conv_manager.start()

    # 3) Video streams
    video_cap = cv2.VideoCapture(args.video)
    if not video_cap.isOpened():
        raise RuntimeError(f"Cannot open video {args.video}")

    webcam_cap = cv2.VideoCapture(0)
    if not webcam_cap.isOpened():
        print("[WARN] Cannot open webcam; using black frame instead.")
        webcam_cap = None

    viewer = Viewer()

    # State
    active_detect: Set[str] = set()
    ignored_classes: Set[str] = set()

    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = video_cap.read()
        if not ret:
            print("[INFO] Video ended.")
            break

        # Read webcam frame
        if webcam_cap is not None:
            ret_w, webcam_frame = webcam_cap.read()
            if not ret_w:
                print("[WARN] Webcam frame error; disabling webcam.")
                webcam_cap.release()
                webcam_cap = None
                # Use black frame to keep layout consistent
                webcam_frame = None
        else:
            webcam_frame = None

        if webcam_frame is None:
            webcam_frame = frame.copy() * 0  # black placeholder

        # 1.4.3.1 – manage new conversation
        if args.mic:
            detect_list, undetect_list = conv_manager.get_current_filters()
            if detect_list or undetect_list:
                active_detect.update(detect_list)
                active_detect.difference_update(undetect_list)
                ignored_classes.update(undetect_list)

        # 1.4.2 – if no conversation (no active classes) -> show original video
        if not active_detect:
            processed = frame
        else:
            # 1.4.3.2 – show video with latest detections
            detections = detector.detect(frame, allowed_classes=active_detect)
            processed = viewer.draw_detections(frame, detections)

        combined = viewer.combine_frames(webcam_frame, processed)
        viewer.show(combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    video_cap.release()
    if webcam_cap is not None:
        webcam_cap.release()
    cv2.destroyAllWindows()

    if args.mic:
        conv_manager.stop()


if __name__ == "__main__":
    main()
