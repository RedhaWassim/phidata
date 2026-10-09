"""
Eden AI Image Agent
===================

Cookbook example for `edenai/image_agent.py`.
"""

from agno.agent import Agent
from agno.media import Image
from agno.models.edenai import EdenAI

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------

agent = Agent(
    model=EdenAI(id="google/gemini-flash-latest"),
    markdown=True,
)

# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent.print_response(
        "Tell me about this image.",
        images=[
            Image(
                url="https://upload.wikimedia.org/wikipedia/commons/0/0c/GoldenGateBridge-001.jpg"
            )
        ],
        stream=True,
    )
