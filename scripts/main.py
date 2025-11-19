# app/main.py
import argparse
from typing import Set
import cv2

from speech_to_text import SpeechToText




def main():
    
    stt = SpeechToText(model_name="small")
    text = stt.transcribe_from_mic(duration_sec=5.0)
    print("You said:", text)
    
if __name__ == "__main__":
    main()
