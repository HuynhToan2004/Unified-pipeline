from __future__ import annotations

import os
from typing import Callable


def get_azure_token_provider() -> Callable[[], str]:
    """Return Azure AD bearer token provider for Azure OpenAI."""
    tenant_id = os.getenv("AZURE_TENANT_ID") or os.getenv("TENANT_ID")
    client_id = os.getenv("AZURE_CLIENT_ID") or os.getenv("CLIENT_ID")
    client_secret = os.getenv("AZURE_CLIENT_SECRET") or os.getenv("CLIENT_SECRET")

    if not tenant_id or not client_id or not client_secret:
        raise RuntimeError(
            "Missing Azure credentials. Set AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET"
        )

    from azure.identity import ClientSecretCredential, get_bearer_token_provider  # type: ignore

    credential = ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
    )

    token_provider = get_bearer_token_provider(
        credential,
        "https://cognitiveservices.azure.com/.default",
    )
    return token_provider
