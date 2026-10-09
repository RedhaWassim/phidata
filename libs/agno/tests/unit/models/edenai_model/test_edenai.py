"""Unit tests for the EdenAI model class.

Eden AI exposes an OpenAI-compatible Chat Completions endpoint, so the class is an OpenAILike
subclass that adds Eden AI's `fallbacks` and `routing` request fields. These tests pin the
defaults, the missing-key behavior and the request body without any network access.
(The to_dict/from_dict round-trip is covered generically by ``test_provider_resolution.py``.)
"""

import pytest

from agno.exceptions import ModelAuthenticationError
from agno.models.edenai import EdenAI
from agno.models.openai.like import OpenAILike
from agno.models.utils import get_model


def test_defaults():
    """Defaults match the Eden AI V3 OpenAI-compatible endpoint."""
    model = EdenAI(api_key="test-key")
    assert isinstance(model, OpenAILike)
    assert model.id == "openai/gpt-5.4-mini"
    assert model.name == "EdenAI"
    assert model.provider == "EdenAI"
    assert model.base_url == "https://api.edenai.run/v3"


def test_api_key_from_env(monkeypatch):
    """The API key is read from EDENAI_API_KEY when not passed explicitly."""
    monkeypatch.setenv("EDENAI_API_KEY", "env-key")
    assert EdenAI().api_key == "env-key"


def test_client_params_include_base_url():
    """Client params carry the configured key and base URL through to the SDK."""
    params = EdenAI(api_key="test-key")._get_client_params()
    assert params["api_key"] == "test-key"
    assert params["base_url"] == "https://api.edenai.run/v3"


def test_eu_base_url():
    """The EU data residency endpoint can be selected through base_url."""
    params = EdenAI(api_key="test-key", base_url="https://api.eu.edenai.run/v3")._get_client_params()
    assert params["base_url"] == "https://api.eu.edenai.run/v3"


def test_missing_api_key_raises(monkeypatch):
    """A missing API key raises ModelAuthenticationError rather than a client error."""
    monkeypatch.delenv("EDENAI_API_KEY", raising=False)
    with pytest.raises(ModelAuthenticationError):
        EdenAI(api_key=None)._get_client_params()


def test_request_params_without_eden_options():
    """No Eden AI specific fields are sent unless configured."""
    params = EdenAI(api_key="test-key").get_request_params()
    assert "extra_body" not in params


def test_request_params_include_fallbacks_and_routing():
    """fallbacks and routing are sent in the request body."""
    model = EdenAI(
        api_key="test-key",
        fallbacks=["anthropic/claude-sonnet-latest"],
        routing={"sort": "latency", "allowed_providers": ["openai"]},
    )
    params = model.get_request_params()
    assert params["extra_body"] == {
        "fallbacks": ["anthropic/claude-sonnet-latest"],
        "routing": {"sort": "latency", "allowed_providers": ["openai"]},
    }


def test_request_params_merge_with_extra_body_without_mutating_it():
    """User supplied extra_body is kept and is not mutated across requests."""
    extra_body = {"tags": {"project": "agno"}}
    model = EdenAI(api_key="test-key", extra_body=extra_body, fallbacks=["mistral/mistral-large-latest"])
    params = model.get_request_params()
    assert params["extra_body"] == {"tags": {"project": "agno"}, "fallbacks": ["mistral/mistral-large-latest"]}
    assert extra_body == {"tags": {"project": "agno"}}


def test_to_dict_includes_eden_options():
    """fallbacks and routing survive serialization."""
    model = EdenAI(api_key="test-key", fallbacks=["openai/gpt-5.4"], routing={"sort": "cost"})
    data = model.to_dict()
    assert data["provider"] == "EdenAI"
    assert data["fallbacks"] == ["openai/gpt-5.4"]
    assert data["routing"] == {"sort": "cost"}


def test_string_syntax(monkeypatch):
    """'edenai:<provider>/<model>' resolves to the EdenAI class."""
    monkeypatch.setenv("EDENAI_API_KEY", "env-key")
    model = get_model("edenai:anthropic/claude-sonnet-latest")
    assert isinstance(model, EdenAI)
    assert model.id == "anthropic/claude-sonnet-latest"
