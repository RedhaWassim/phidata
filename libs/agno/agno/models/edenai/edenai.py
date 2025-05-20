from dataclasses import dataclass
from os import getenv
from typing import Any, Dict, Optional

from agno.exceptions import ModelProviderError
from agno.models.openai.like import OpenAILike


@dataclass
class Edenai(OpenAILike):
    """
    A class for interacting with Edenai models.

    For more information, see: https://docs.edenai.co/reference/llm_root_create
    """

    id: str = "edenai-chat"
    name: str = "Edenai"
    provider: str = "Edenai"

    api_key: Optional[str] = getenv("EDENAI_API_KEY", None)
    base_url: str = "https://api.edenai.run/v2/llm"
