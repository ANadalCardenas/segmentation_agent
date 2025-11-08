# app/speech_to_text.py
import threading
import queue
import tempfile
from typing import List, Set, Tuple, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper


class SpeechToText:
    """
    Thin wrapper around Whisper ASR.
    """
    def __init__(self, model_name: str = "small"):
        # For GPU, Whisper will automatically use CUDA if available.
        self.model = whisper.load_model(model_name)

    def transcribe_wav(self, filename: str) -> str:
        result = self.model.transcribe(filename, language="en")
        return result["text"].strip()


class ConversationManager:
    """
    Runs a background thread that:
    - records short audio chunks
    - transcribes them
    - passes transcripts into a parser to get detect/undetect lists
    """
    def __init__(self, parser, sample_rate: int = 16000, chunk_seconds: int = 5):
        self.parser = parser
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds

        self._stt = SpeechToText()
        self._thread = None
        self._stop_event = threading.Event()

        # Shared state
        self._current_detect: Set[str] = set()
        self._current_undetect: Set[str] = set()
        self._lock = threading.Lock()

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run_loop(self):
        while not self._stop_event.is_set():
            audio = self._record_chunk()
            if audio is None:
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, audio, self.sample_rate)
                text = self._stt.transcribe_wav(tmp.name)

            if text:
                detect, undetect = self.parser.parse(text)
                with self._lock:
                    self._current_detect.update(detect)
                    self._current_detect.difference_update(undetect)
                    self._current_undetect.update(undetect)

    def _record_chunk(self) -> Optional[np.ndarray]:
        try:
            frames = int(self.chunk_seconds * self.sample_rate)
            audio = sd.rec(frames, samplerate=self.sample_rate, channels=1, dtype="float32")
            sd.wait()
            return np.squeeze(audio, axis=-1)
        except Exception as e:
            print(f"[ConversationManager] Audio error: {e}")
            return None

    def get_current_filters(self) -> Tuple[Set[str], Set[str]]:
        """
        Returns (objects_to_detect, objects_to_undetect).
        Thread-safe snapshot.
        """
        with self._lock:
            return set(self._current_detect), set(self._current_undetect)
