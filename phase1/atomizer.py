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

try:
    from .azure_chat_client import connect_to_azure_chat_model, parse_model_response
except Exception:
    try:
        from azure_chat_client import connect_to_azure_chat_model, parse_model_response
    except Exception:
        connect_to_azure_chat_model = None
        parse_model_response = None


class LLMAtomizer:
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> None:
        self.provider = (provider or os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model = model
        self.temperature = temperature
        self._mode = "heuristic"
        self._client = None

        if self.provider == "azure":
            self.model = self.model or os.getenv("AZURE_CHAT_DEPLOYMENT")
            if connect_to_azure_chat_model is not None:
                self._client = connect_to_azure_chat_model(json_mode=True)
                if self._client is not None:
                    self._mode = "azure"
            return

        if self.provider == "gemini" and self.api_key:
            self.model = self.model or os.getenv("GEMINI_MODEL") or "models/gemini-2.5-flash-lite"
            try:
                from google import genai  # type: ignore

                self._client = genai.Client(api_key=self.api_key)
                self._mode = "gemini"
            except Exception:
                self._mode = "heuristic"
            return

        self._mode = "heuristic"

    @property
    def mode(self) -> str:
        return self._mode

    def atomize_section(self, section_name: str, section_text: str) -> List[Dict[str, str]]:
        fallback_texts = heuristic_atomize(section_text)
        fallback = {"atomic_arguments": fallback_texts}

        if self._mode == "heuristic":
            return [{"arg_id": f"a{i}", "text": t} for i, t in enumerate(fallback_texts, start=1)]

        prompt = f"""
You are an expert peer-review analyst.
Task: split the section into independent atomic arguments.
Return STRICT JSON only:
{{"atomic_arguments": ["arg1", "arg2", "arg3"]}}

Rules:
- Keep each argument independent and minimally sufficient.
- Preserve original meaning, do not hallucinate new facts.
- Keep concise and specific.

Section: {section_name}
Text:
{section_text}
""".strip()

        parsed = self._generate_json(prompt, fallback)
        values = parsed.get("atomic_arguments") if isinstance(parsed, dict) else None
        if not isinstance(values, list):
            values = fallback_texts

        out: List[Dict[str, str]] = []
        for idx, value in enumerate(values, start=1):
            t = str(value).strip()
            if t:
                out.append({"arg_id": f"a{idx}", "text": t})
        return out

    def _generate_json(self, prompt: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
        if self._mode == "azure":
            try:
                ai_answer = self._client.invoke(prompt)
                parsed = parse_model_response(ai_answer) if parse_model_response else {}
                if isinstance(parsed, dict) and "atomic_arguments" in parsed:
                    return parsed

                raw = parsed.get("raw") if isinstance(parsed, dict) else str(ai_answer)
                raw = _strip_code_fences(str(raw))
                raw = _extract_first_json(raw)
                return json.loads(raw)
            except Exception:
                return fallback

        try:
            from google import genai  # type: ignore

            resp = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=2048,
                ),
            )
            raw = (resp.text or "").strip()
            raw = _strip_code_fences(raw)
            raw = _extract_first_json(raw)
            return json.loads(raw)
        except Exception:
            return fallback


class GeminiAtomizer(LLMAtomizer):
    def __init__(self, model: Optional[str] = None, temperature: float = 0.1) -> None:
        super().__init__(provider="gemini", model=model, temperature=temperature)


def heuristic_atomize(text: str) -> List[str]:
    if not text or not text.strip():
        return []

    lines = [line.strip("-*• \t") for line in text.splitlines() if line.strip()]
    out: List[str] = []
    for line in lines:
        pieces = re.split(r"(?<=[.!?])\s+", line)
        for piece in pieces:
            p = piece.strip()
            if p:
                out.append(p)
    return out


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
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
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
