from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=False)


class NoveltyLLMBackend:
    def __init__(self, provider: str = "heuristic", model: Optional[str] = None, temperature: float = 0.0):
        self.provider = (provider or os.getenv("LLM_PROVIDER") or "heuristic").strip().lower()
        self.model = model
        self.temperature = temperature
        self._mode = "heuristic"
        self._client = None

        if self.provider in {"openai", "vllm"}:
            try:
                from openai import OpenAI  # type: ignore

                base_url = os.getenv("OPENAI_BASE_URL") if self.provider == "vllm" else os.getenv("LLM_API_ENDPOINT")
                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or "dummy"
                self.model = self.model or os.getenv("LLM_MODEL_NAME") or "gpt-4o-mini"
                self._client = OpenAI(api_key=api_key, base_url=base_url)
                self._mode = self.provider
            except Exception:
                self._mode = "heuristic"
            return

        if self.provider == "azure":
            try:
                from phase1.azure_chat_client import connect_to_azure_chat_model, parse_model_response

                self._client = connect_to_azure_chat_model(json_mode=True)
                self._parse_azure = parse_model_response
                self._mode = "azure" if self._client is not None else "heuristic"
            except Exception:
                self._mode = "heuristic"
            return

        if self.provider == "gemini":
            try:
                from google import genai  # type: ignore

                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                self.model = self.model or os.getenv("GEMINI_MODEL") or "models/gemini-2.5-flash-lite"
                if api_key:
                    self._client = genai.Client(api_key=api_key)
                    self._mode = "gemini"
            except Exception:
                self._mode = "heuristic"

    @property
    def mode(self) -> str:
        return self._mode

    def generate_json(self, prompt: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
        if self._mode == "heuristic":
            return fallback

        try:
            if self._mode in {"openai", "vllm"}:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content or "{}"
                return json.loads(raw)

            if self._mode == "azure":
                ai_answer = self._client.invoke(prompt)
                parsed = self._parse_azure(ai_answer)
                if isinstance(parsed, dict) and parsed:
                    if "raw" in parsed:
                        raw = _extract_first_json(_strip_code_fences(str(parsed.get("raw", ""))))
                        return json.loads(raw)
                    return parsed

            if self._mode == "gemini":
                from google import genai  # type: ignore

                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(temperature=self.temperature, max_output_tokens=4096),
                )
                raw = _extract_first_json(_strip_code_fences((resp.text or "").strip()))
                return json.loads(raw)
        except Exception:
            return fallback

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
        raise ValueError("No JSON found")
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
    raise ValueError("Unbalanced JSON")
