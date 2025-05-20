"""Run `pip install duckduckgo-search` to install dependencies."""

from agno.agent import Agent
from agno.models.edenai import Edenai
from agno.tools.duckduckgo import DuckDuckGoTools
import dotenv
dotenv.load_dotenv()
agent = Agent(
    model=Edenai(id="deepseek/deepseek-chat"),

    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
)
agent.print_response("Whats happening in France?")
