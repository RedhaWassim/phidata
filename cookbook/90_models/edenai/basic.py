"""
Eden AI Basic
=============

Cookbook example for `edenai/basic.py`.
"""

import asyncio

from agno.agent import Agent
from agno.models.edenai import EdenAI

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------

agent = Agent(
    model=EdenAI(id="openai/gpt-5.4-mini"),
    markdown=True,
)

# The same kind of agent, using the "provider:model" string syntax
string_agent = Agent(
    model="edenai:anthropic/claude-sonnet-latest",
    markdown=True,
)

# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Sync ---
    agent.print_response("Share a 2 sentence horror story")

    # --- Sync + Streaming ---
    string_agent.print_response("Share a 2 sentence horror story", stream=True)

    # --- Async ---
    asyncio.run(agent.aprint_response("Share a 2 sentence horror story"))

    # --- Async + Streaming ---
    asyncio.run(
        string_agent.aprint_response("Share a 2 sentence horror story", stream=True)
    )
