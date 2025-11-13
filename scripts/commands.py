
from typing import Iterable, Set, Tuple, Dict
import re


class CommandParser:
    """
    Extracts 'objects to detect' and 'objects to undetect' from
    natural language transcripts, given known YOLO class names.

    Very simple heuristic example; you can make this smarter later.
    """

    def __init__(self, yolo_class_names: Iterable[str]):
        self.yolo_classes = set(name.lower() for name in yolo_class_names)
        # Simple synonym mapping (add more as needed)
        self.synonyms: Dict[str, str] = {}
        self.synonyms = {
        "people": "person",
        "cars": "car",
        "bottles": "bottle",
        }
    

    def _normalize_token(self, token: str) -> str:
        token = token.lower()
        if token in self.synonyms:
            return self.synonyms[token]
        return token

    def parse(self, transcript: str) -> Tuple[Set[str], Set[str]]:
        text = transcript.lower()
        tokens = re.findall(r"[a-zA-Z]+", text)

        detect: Set[str] = set()
        undetect: Set[str] = set()

        # Heuristics: look for patterns like:
        #  "show people", "detect bottles", "start detecting cars"
        #  "stop showing people", "ignore cars", "don't detect bottles"
        for i, token in enumerate(tokens):
            norm = self._normalize_token(token)

            if norm in self.yolo_classes:
                prev_tokens = tokens[max(0, i - 3):i]
                prev_text = " ".join(prev_tokens)

                # Negative patterns
                if re.search(r"\b(no|not|don't|stop|remove|hide|ignore)\b", prev_text):
                    undetect.add(norm)
                else:
                    detect.add(norm)

        return detect, undetect
