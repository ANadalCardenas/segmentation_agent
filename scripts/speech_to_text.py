# app/speech_to_text.py
import threading
import tempfile
import time
from typing import Set, Tuple, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper


class SpeechToText:
    """
    Thin wrapper around Whisper ASR.
    """
    def __init__(self, model_name: str = "tiny"):
        # Use a lighter model to avoid freezing
        self.model = whisper.load_model(model_name)

    def transcribe_wav(self, filename: str) -> str:
        result = self.model.transcribe(filename, language="en")
        return result["text"].strip()


class ConversationManager:
    def __init__(self, parser, sample_rate: int | None = None, chunk_seconds: float = 2.0):
        self.parser = parser
        self.chunk_seconds = chunk_seconds

        # Auto-detect valid input sample rate if not provided
        if sample_rate is None:
            dev_info = sd.query_devices(None, 'input')  # default input device
            self.sample_rate = int(dev_info['default_samplerate'])
            print(f"[ConversationManager] Using input sample rate: {self.sample_rate}")
        else:
            self.sample_rate = sample_rate

        self._stt = SpeechToText()
        self._thread = None
        self._stop_event = threading.Event()

        # Recording state (controlled by UI button)
        self._recording = False

        # Shared state
        self._current_detect: Set[str] = set()
        self._current_undetect: Set[str] = set()
        self._current_unsupported: Set[str] = set()
        self._lock = threading.Lock()

    # --------- Public control API ---------

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def set_recording(self, recording: bool):
        """
        Called from UI (button) to start/stop mic capture.
        """
        self._recording = recording
        if recording:
            print("[ConversationManager] Recording started.")
        else:
            print("[ConversationManager] Recording stopped.")

    # --------- Internal loop ---------

    def _run_loop(self):
        while not self._stop_event.is_set():
            # Idle until recording is requested
            if not self._recording:
                time.sleep(0.05)
                continue

            # Record one chunk while recording is True
            audio = self._record_chunk()
            if audio is None:
                continue

            # If user already released the button, skip STT
            if not self._recording:
                continue

            self._process_audio(audio)

    def _record_chunk(self) -> Optional[np.ndarray]:
        try:
            frames = int(self.chunk_seconds * self.sample_rate)
            audio = sd.rec(frames, samplerate=self.sample_rate, channels=1, dtype="float32")
            sd.wait()
            return np.squeeze(audio, axis=-1)
        except Exception as e:
            print(f"[ConversationManager] Audio error at {self.sample_rate} Hz: {e}")
            return None

    def _process_audio(self, audio: np.ndarray):
        # Write to temp WAV and call Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, audio, self.sample_rate)

            try:
                text = self._stt.transcribe_wav(tmp.name)
            except Exception as e:
                print(f"[SpeechToText] Transcription error: {e}")
                return

        if not text:
            return

        print(f"[YOU SAID]: {text}")

        # Parse commands → now returns 3 sets
        try:
            detect, undetect, unsupported = self.parser.parse(text)
        except Exception as e:
            print(f"[ConversationManager] Command parsing error: {e}")
            return

        # Update filters
        with self._lock:
            self._current_detect.update(detect)
            self._current_detect.difference_update(undetect)
            self._current_undetect.update(undetect)
            self._current_unsupported.update(unsupported)

    def get_current_filters(self) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Returns (objects_to_detect, objects_to_undetect, unsupported_objects).
        Thread-safe snapshot.
        """
        with self._lock:
            return (
                set(self._current_detect),
                set(self._current_undetect),
                set(self._current_unsupported),
            )
