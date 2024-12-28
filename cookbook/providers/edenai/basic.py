from phi.agent import Agent, RunResponse  # noqa
from phi.model.edenai import EdenAIChat

agent = Agent(model=EdenAIChat(name="openai/gpt-4o"), markdown=True)

# Get the response in a variable
# run: RunResponse = agent.run("Share a 2 sentence horror story")
# print(run.content)

# Print the response on the terminal
agent.print_response("Share a 2 sentence horror story")
