"""Run `pip install yfinance` to install dependencies."""

from phi.agent import Agent, RunResponse  # noqa
from phi.model.edenai import EdenAI
from phi.tools.yfinance import YFinanceTools
agent = Agent(
    model=EdenAI(id="openai/gpt-4",markdown=True)
)

# Get the response in a variable
# run: RunResponse = agent.run("What is the stock price of NVDA and TSLA")
# print(run.content)

# Print the response on the terminal
print("test")
agent.print_response("Give me in-depth analysis of NVDA and TSLA")