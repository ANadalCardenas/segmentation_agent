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
    parser.add_argument(
        "--video",
        default="/workspace/segmentation_agent/media/video.mp4",
        help="Path to video file for the right view",
    )
    parser.add_argument("--model", default="yolov5s")
    parser.add_argument(
        "--mic", action="store_true", help="Enable microphone-based commands"
    )
    args = parser.parse_args()

    # 1) Init detector & parser
    detector = ObjectDetector(model_name=args.model)
    class_names = detector.get_class_names()
    cmd_parser = CommandParser(class_names)

    # 2) Conversation manager (background thread) – only if mic is enabled
    conv_manager = None
    if args.mic:
        conv_manager = ConversationManager(parser=cmd_parser)
        conv_manager.start()

    # 3) Video sources
    webcam_cap = cv2.VideoCapture(0)
    if not webcam_cap.isOpened():
        print("[WARN] Cannot open webcam; using black frame instead.")
        webcam_cap = None

    video_cap = cv2.VideoCapture(args.video)
    if not video_cap.isOpened():
        print(f"[WARN] Cannot open video {args.video}; using black frame instead.")
        video_cap = None

    viewer = Viewer()

    # If mic is enabled, connect button to mic control
    if conv_manager is not None:
        viewer.set_button_callback(lambda pressed: conv_manager.set_recording(pressed))


    # State
    active_detect: Set[str] = set()
    ignored_classes: Set[str] = set()

    print("[INFO] Press 'q' to quit or click the window 'X'.")

    while True:
        # Read right video (file)
        if video_cap is not None:
            ret_v, frame_video = video_cap.read()
            if not ret_v:
                print("[INFO] Video ended.")
                break
        else:
            frame_video = None

        # Read left video (webcam)
        if webcam_cap is not None:
            ret_w, frame_webcam = webcam_cap.read()
            if not ret_w:
                print("[WARN] Webcam frame error; disabling webcam.")
                webcam_cap.release()
                webcam_cap = None
                frame_webcam = None
        else:
            frame_webcam = None

        # Handle missing sources (black placeholders)
        if frame_webcam is None and frame_video is None:
            print("[ERROR] No video sources available.")
            break

        if frame_webcam is None and frame_video is not None:
            frame_webcam = frame_video.copy() * 0
        elif frame_video is None and frame_webcam is not None:
            frame_video = frame_webcam.copy() * 0

        # Ensure same height for concatenation
        h = min(frame_webcam.shape[0], frame_video.shape[0])
        frame_webcam = cv2.resize(
            frame_webcam,
            (int(frame_webcam.shape[1] * h / frame_webcam.shape[0]), h),
        )
        frame_video = cv2.resize(
            frame_video,
            (int(frame_video.shape[1] * h / frame_video.shape[0]), h),
        )

        # Apply detection only on right (video file)
        if conv_manager is not None:
            detect_list, undetect_list = conv_manager.get_current_filters()
            if detect_list or undetect_list:
                active_detect.update(detect_list)
                active_detect.difference_update(undetect_list)
                ignored_classes.update(undetect_list)

        if not active_detect:
            processed = frame_video
        else:
            detections = detector.detect(frame_video, allowed_classes=active_detect)
            processed = viewer.draw_detections(frame_video, detections)

        # Combine frames: webcam (left) + processed video (right)
        combined = cv2.hconcat([frame_webcam, processed])

        # Add bottom UI (push-to-talk button)
        frame_with_ui = viewer.draw_ui(combined)

        # Show frame in a single persistent window
        viewer.show(frame_with_ui)

        # Process GUI events (keyboard + window "X")
        key = cv2.waitKey(400) & 0xFF
        if key == ord("q"):
            print("[INFO] 'q' pressed. Exiting.")
            break

        # Check if user closed the window with "X"
        try:
            prop = cv2.getWindowProperty(viewer.window_name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            # Window already destroyed
            print("[INFO] Window destroyed. Exiting.")
            break

        if prop < 1:
            print("[INFO] Window closed by user. Exiting.")
            break

    # Cleanup
    if video_cap is not None:
        video_cap.release()
    if webcam_cap is not None:
        webcam_cap.release()
    cv2.destroyAllWindows()

    if conv_manager is not None:
        conv_manager.stop()


if __name__ == "__main__":
    main()
