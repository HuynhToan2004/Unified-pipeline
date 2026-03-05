from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    from dotenv import find_dotenv, load_dotenv  # type: ignore

    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass


class UnifiedLLMClient:
    def __init__(self, model: Optional[str] = None, temperature: float = 0.1):
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL") or "models/gemini-2.5-flash-lite"
        self.temperature = temperature
        self._mode = "heuristic"
        self._client = None

        if self.api_key:
            try:
                from google import genai  # type: ignore

                self._client = genai.Client(api_key=self.api_key)
                self._mode = "gemini"
            except Exception:
                self._mode = "heuristic"

    @property
    def mode(self) -> str:
        return self._mode

    def generate_json(self, prompt: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
        if self._mode != "gemini":
            return fallback

        try:
            from google import genai  # type: ignore

            resp = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=4096,
                ),
            )
            text = (resp.text or "").strip()
            text = _strip_code_fences(text)
            text = _extract_first_json(text)
            return json.loads(text)
        except Exception:
            return fallback


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_first_json(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    depth = 0
    in_str = False
    esc = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    raise ValueError("Unbalanced JSON object")


def basic_atomize(text: str) -> List[str]:
    lines = [ln.strip("-•* \t") for ln in text.splitlines() if ln.strip()]
    out: List[str] = []
    for line in lines:
        parts = re.split(r"(?<=[.!?])\s+", line)
        for part in parts:
            p = part.strip()
            if p:
                out.append(p)
    return out
