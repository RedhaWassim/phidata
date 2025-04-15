from typing import Iterator  # noqa
from phi.agent import Agent, RunResponse  # noqaclear
from phi.model.edenai import EdenAIChat

agent = Agent(model=EdenAIChat(id="openai/gpt-4o",    api_key = ".."
                               ,stream=True), markdown=True)

# Get the response in a variable
# run_response: Iterator[RunResponse] = agent.run("Share a 2 sentence horror story", stream=True)
# for chunk in run_response:
#     print(chunk.content)

# Print the response in the terminal
agent.print_response("Share a 2 sentence love story", stream=True, verbose=True)
