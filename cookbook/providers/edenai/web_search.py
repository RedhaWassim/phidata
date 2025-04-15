"""Run `pip install duckduckgo-search` to install dependencies."""

from phi.agent import Agent
from phi.model.edenai import EdenAIChat
from phi.tools.duckduckgo import DuckDuckGo

agent = Agent(
    model=EdenAIChat(name="openai/gpt-4o",
                     api_key = ".."
),
    tools=[DuckDuckGo()],
    show_tool_calls=True,
    markdown=True,
)

agent.print_response("Whats happening in France?", stream=False)
