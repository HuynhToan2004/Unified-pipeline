from __future__ import annotations

import re
from typing import Dict, List, Optional

try:
    import spacy  # type: ignore
except Exception:
    spacy = None


class SpacySentenceExtractor:
    def __init__(self) -> None:
        self._nlp: Optional[object] = None
        if spacy is not None:
            try:
                self._nlp = spacy.blank("en")
                if "sentencizer" not in self._nlp.pipe_names:
                    self._nlp.add_pipe("sentencizer")
            except Exception:
                self._nlp = None

    def extract(self, text: str) -> List[Dict[str, str]]:
        text = (text or "").strip()
        if not text:
            return []

        if self._nlp is not None:
            doc = self._nlp(text)
            out: List[Dict[str, str]] = []
            for idx, sent in enumerate(doc.sents, start=1):
                s = sent.text.strip()
                if s:
                    out.append({"sent_id": f"s{idx}", "text": s})
            return out

        parts = re.split(r"(?<=[.!?])\s+", text)
        out = []
        for idx, part in enumerate(parts, start=1):
            p = part.strip()
            if p:
                out.append({"sent_id": f"s{idx}", "text": p})
        return out
