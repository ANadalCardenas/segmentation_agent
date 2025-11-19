# speech_to_text.py

"""from typing import Optional

import numpy as np
import sounddevice as sd
import whisper


class SpeechToText:
    
    def __init__(self, model_name: str = "base", sample_rate: int = 16000):
        
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.model = whisper.load_model(model_name)

    # ---------- MICROPHONE RECORDING ----------

    def _record_from_mic(self, duration_sec: float) -> np.ndarray:
        
        num_samples = int(duration_sec * self.sample_rate)
        audio = sd.rec(
            num_samples,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        print("[SpeechToText] Recording finished.")
        # Flatten to 1D array
        return audio.flatten()

    # ---------- TRANSCRIPTION METHODS ----------

    def transcribe_from_mic(self, duration_sec: float, language: str = "en") -> str:
        
        audio_buffer = self._record_from_mic(duration_sec)
        return self.transcribe_buffer(audio_buffer, language=language)

    def transcribe_buffer(self, audio_buffer, language: str = "en") -> str:
        
        result = self.model.transcribe(audio_buffer, language=language)
        return result["text"].strip()
"""

# speech_to_text.py

from typing import Optional

import numpy as np
import sounddevice as sd
import whisper
from sounddevice import PortAudioError


class SpeechToText:
    """
    Wrapper around OpenAI Whisper that can record audio from the microphone
    and transcribe it to text.
    """

    def __init__(self, model_name: str = "large", sample_rate: Optional[int] = None):
        """
        Args:
            model_name: tiny, base, small, medium, large
        """
        self.model_name = model_name
        # --- choose device sample rate ---
        if sample_rate is None:
            try:
                dev_info = sd.query_devices(sd.default.device[0], "input")
            except Exception:
                dev_info = sd.query_devices(kind="input")
            self.device_sample_rate = int(dev_info["default_samplerate"])
        else:
            try:
                sd.check_input_settings(samplerate=sample_rate)
                self.device_sample_rate = sample_rate
            except PortAudioError:
                print(
                    f"[SpeechToText] Sample rate {sample_rate} not supported, "
                    "falling back to default input device sample rate."
                )
                try:
                    dev_info = sd.query_devices(sd.default.device[0], "input")
                except Exception:
                    dev_info = sd.query_devices(kind="input")
                self.device_sample_rate = int(dev_info["default_samplerate"])

        print(f"[SpeechToText] Device sample rate: {self.device_sample_rate} Hz")


        # 2) Load Whisper model
        self.model = whisper.load_model(model_name)

    # ---------- MICROPHONE RECORDING ----------

    def _record_from_mic(self, duration_sec: float) -> np.ndarray:
        """
        Record audio from the default microphone.

        Returns 1D float32 NumPy array at device_sample_rate.
        """
        num_samples = int(duration_sec * self.device_sample_rate)
        print(f"[SpeechToText] Recording {duration_sec:.1f}s of audio...")
        audio = sd.rec(
            num_samples,
            samplerate=self.device_sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        print("[SpeechToText] Recording finished.")
        return audio.flatten()


    # ---------- TRANSCRIPTION METHODS ----------

    def transcribe_from_mic(
        self,
        duration_sec: float,
        language: Optional[str] = "en",
    ) -> str:
        audio_buffer = self._record_from_mic(duration_sec)
        return self.transcribe_buffer(audio_buffer, language=language)

    def transcribe_buffer(
        self,
        audio_buffer,
        language: Optional[str] = "en",
    ) -> str:
        
        result = self.model.transcribe(audio_buffer, language=language)
        return result["text"].strip()
