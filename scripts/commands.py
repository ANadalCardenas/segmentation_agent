# app/commands.py
from typing import Iterable, Set, Tuple, Dict
import re


class CommandParser:
    """
    Extracts:
      - objects to detect
      - objects to undetect
      - objects requested but not in YOLO classes (unsupported)
    from natural language transcripts.
    """

    def __init__(self, yolo_class_names: Iterable[str]):
        self.yolo_classes = set(name.lower() for name in yolo_class_names)
        # Simple synonym mapping (add more as needed)
        self.synonyms: Dict[str, str] = {
            "people": "person",
            "cars": "car",
            "bottles": "bottle",
        }

    def _normalize_token(self, token: str) -> str:
        token = token.lower()
        if token in self.synonyms:
            return self.synonyms[token]
        return token

    def parse(self, transcript: str) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Returns (detect, undetect, unsupported).
        - detect / undetect: class names that exist in YOLO.
        - unsupported: tokens that look like requested objects but are not YOLO classes.
        """
        text = transcript.lower()
        tokens = re.findall(r"[a-zA-Z]+", text)

        detect: Set[str] = set()
        undetect: Set[str] = set()
        unsupported: Set[str] = set()

        for i, token in enumerate(tokens):
            norm = self._normalize_token(token)

            prev_tokens = tokens[max(0, i - 3):i]
            prev_text = " ".join(prev_tokens)

            # Known YOLO class
            if norm in self.yolo_classes:
                # Negative patterns
                if re.search(r"\b(no|not|don't|stop|remove|hide|ignore)\b", prev_text):
                    undetect.add(norm)
                else:
                    detect.add(norm)
            else:
                # Unknown class → mark as unsupported if preceded by a "detect/show" verb
                if re.search(r"\b(detect|show|track|find|highlight|mark|see|watch)\b", prev_text):
                    unsupported.add(token.lower())

        return detect, undetect, unsupported
