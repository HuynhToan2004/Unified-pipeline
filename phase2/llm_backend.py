from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

try:
    from dotenv import find_dotenv, load_dotenv  # type: ignore

    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass

try:
    from phase1.azure_chat_client import connect_to_azure_chat_model, parse_model_response
except Exception:
    connect_to_azure_chat_model = None
    parse_model_response = None


class UnifiedLLMBackend:
    def __init__(self, provider: str = "gemini", model: Optional[str] = None, temperature: float = 0.0) -> None:
        self.provider = (provider or os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
        self.temperature = temperature
        self.model = model
        self._mode = "heuristic"
        self._client = None

        if self.provider == "azure":
            self.model = self.model or os.getenv("AZURE_CHAT_DEPLOYMENT")
            if connect_to_azure_chat_model is not None:
                self._client = connect_to_azure_chat_model(json_mode=True)
                if self._client is not None:
                    self._mode = "azure"
            return

        if self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            self.model = self.model or os.getenv("GEMINI_MODEL") or "models/gemini-2.5-flash-lite"
            if api_key:
                try:
                    from google import genai  # type: ignore

                    self._client = genai.Client(api_key=api_key)
                    self._mode = "gemini"
                except Exception:
                    self._mode = "heuristic"
            return

    @property
    def mode(self) -> str:
        return self._mode

    def generate_json(self, prompt: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
        if self._mode == "heuristic":
            return fallback

        if self._mode == "azure":
            try:
                ai_answer = self._client.invoke(prompt)
                parsed = parse_model_response(ai_answer) if parse_model_response else {}
                if isinstance(parsed, dict) and parsed:
                    if "raw" in parsed:
                        raw = _extract_first_json(_strip_code_fences(str(parsed.get("raw", ""))))
                        return json.loads(raw)
                    return parsed
            except Exception:
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
            raw = _strip_code_fences((resp.text or "").strip())
            raw = _extract_first_json(raw)
            return json.loads(raw)
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
    if start < 0:
        raise ValueError("No JSON object found")

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
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
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("Unbalanced JSON braces")
