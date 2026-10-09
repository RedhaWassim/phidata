from dataclasses import dataclass, field
from os import getenv
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from agno.exceptions import ModelAuthenticationError
from agno.models.openai.like import OpenAILike
from agno.run.agent import RunOutput
from agno.run.team import TeamRunOutput


@dataclass
class EdenAI(OpenAILike):
    """
    A class for using LLMs through the Eden AI V3 OpenAI-compatible Chat Completions endpoint.

    Eden AI routes requests to many underlying providers behind a single API key. Models are
    addressed as "<provider>/<model>" (e.g. "openai/gpt-5.4-mini", "anthropic/claude-sonnet-latest").
    A bare model name (e.g. "gpt-5.6-sol") lets Eden AI pick the provider.
    See https://www.edenai.co/docs/v3/llms/chat-completions

    Attributes:
        id (str): The model id. Defaults to "openai/gpt-5.4-mini".
        name (str): The model name. Defaults to "EdenAI".
        provider (str): The provider name. Defaults to "EdenAI".
        api_key (Optional[str]): The API key. Defaults to the EDENAI_API_KEY environment variable.
        base_url (str): The base URL. Defaults to "https://api.edenai.run/v3".
            Use "https://api.eu.edenai.run/v3" for EU data residency.
        fallbacks (Optional[List[str]]): Up to 3 model ids Eden AI tries in order if the primary model fails.
            Example: ["anthropic/claude-sonnet-latest", "mistral/mistral-large-latest"]
        routing (Optional[Dict[str, Any]]): Provider routing preferences for models served by several
            providers, e.g. {"sort": "latency", "allowed_providers": ["openai", "azure"]}.
            See https://www.edenai.co/docs/v3/llms/provider-routing
    """

    id: str = "openai/gpt-5.4-mini"
    name: str = "EdenAI"
    provider: str = "EdenAI"

    api_key: Optional[str] = field(default_factory=lambda: getenv("EDENAI_API_KEY"))
    base_url: str = "https://api.edenai.run/v3"

    fallbacks: Optional[List[str]] = None
    routing: Optional[Dict[str, Any]] = None

    def _get_client_params(self) -> Dict[str, Any]:
        """
        Returns client parameters for API requests, checking for EDENAI_API_KEY.

        Returns:
            Dict[str, Any]: A dictionary of client parameters for API requests.
        """
        if not self.api_key:
            raise ModelAuthenticationError(
                message="EDENAI_API_KEY not set. Please set the EDENAI_API_KEY environment variable.",
                model_name=self.name,
            )
        return super()._get_client_params()

    def get_request_params(
        self,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_response: Optional[Union[RunOutput, TeamRunOutput]] = None,
    ) -> Dict[str, Any]:
        """
        Returns keyword arguments for API requests, adding the Eden AI fallbacks and routing preferences.

        Returns:
            Dict[str, Any]: A dictionary of keyword arguments for API requests.
        """
        request_params = super().get_request_params(
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            run_response=run_response,
        )

        eden_params: Dict[str, Any] = {}
        if self.fallbacks:
            eden_params["fallbacks"] = self.fallbacks
        if self.routing:
            eden_params["routing"] = self.routing

        if eden_params:
            # Copy so the model's own extra_body is never mutated between requests
            extra_body = dict(request_params.get("extra_body") or {})
            extra_body.update(eden_params)
            request_params["extra_body"] = extra_body

        return request_params

    def to_dict(self) -> Dict[str, Any]:
        model_dict = super().to_dict()
        if self.fallbacks:
            model_dict["fallbacks"] = self.fallbacks
        if self.routing:
            model_dict["routing"] = self.routing
        return model_dict
