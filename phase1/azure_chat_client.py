from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

try:
    from .auth_azure import get_azure_token_provider
except Exception:
    from auth_azure import get_azure_token_provider


def connect_to_azure_chat_model(json_mode: bool = True):
    """Connect to Azure OpenAI chat model using AAD token provider."""
    try:
        from langchain_openai import AzureChatOpenAI  # type: ignore

        token_provider = get_azure_token_provider()

        model_kwargs = {}
        if json_mode:
            model_kwargs["response_format"] = {"type": "json_object"}

        model = AzureChatOpenAI(
            api_version=os.getenv("AZURE_API_VERSION", "2024-10-21"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
            azure_ad_token_provider=token_provider,
            temperature=0,
            model_kwargs=model_kwargs,
        )
        return model
    except Exception:
        return None


def parse_model_response(ai_answer: Any) -> Dict[str, Any]:
    """Safely parse model response into a dictionary."""
    if ai_answer is None:
        return {}

    content = getattr(ai_answer, "content", None)

    if isinstance(content, dict):
        return content

    raw = content if content else str(ai_answer)
    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw}
