"""
Eden AI Fallbacks And Routing
=============================

Cookbook example for `edenai/fallbacks_and_routing.py`.

- `fallbacks`: models Eden AI tries in order if the primary model fails (max 3).
- `routing`: how Eden AI picks the provider of a model sold by several providers.
  A model id without a provider prefix (e.g. "gpt-5.6-sol") lets Eden AI choose.
- The EU endpoint only routes to providers cleared for EU data residency.
"""

from agno.agent import Agent
from agno.models.edenai import EdenAI

# ---------------------------------------------------------------------------
# Create Agents
# ---------------------------------------------------------------------------

# Falls back to Claude, then Mistral, if the primary model is unavailable
fallback_agent = Agent(
    model=EdenAI(
        id="openai/gpt-5.4-mini",
        fallbacks=["anthropic/claude-sonnet-latest", "mistral/mistral-large-latest"],
    ),
    markdown=True,
)

# Lets Eden AI pick the lowest-latency provider serving the model
routed_agent = Agent(
    model=EdenAI(
        id="gpt-5.6-sol",
        routing={"sort": "latency", "allowed_providers": ["openai", "azure"]},
    ),
    markdown=True,
)

# Keeps requests in the EU
eu_agent = Agent(
    model=EdenAI(
        id="mistral/mistral-large-latest",
        base_url="https://api.eu.edenai.run/v3",
    ),
    markdown=True,
)

# ---------------------------------------------------------------------------
# Run Agents
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fallback_agent.print_response("Explain model fallbacks in one sentence.")
    routed_agent.print_response("Explain provider routing in one sentence.")
    eu_agent.print_response("Explain data residency in one sentence.")
