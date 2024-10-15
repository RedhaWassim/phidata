from phi.agent import Agent
from phi.model.openai import OpenAIChat

task = "9.11 and 9.9 -- which is bigger?"

reasoning_agent = Agent(
    model=OpenAIChat(id="gpt-4o-2024-08-06"), reasoning=True, markdown=True, structured_outputs=True
)

reasoning_agent.print_response(task, stream=True, show_full_reasoning=True)