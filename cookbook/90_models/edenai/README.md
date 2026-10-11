# Eden AI

Cookbook examples for `cookbook/90_models/edenai`.

[Eden AI](https://www.edenai.co) is an EU-based AI gateway. Its V3
[Chat Completions endpoint](https://www.edenai.co/docs/v3/llms/chat-completions) is
OpenAI-compatible and routes to 500+ models from many providers behind a single API key.
Models are addressed as `<provider>/<model>`, e.g. `openai/gpt-5.4-mini`,
`anthropic/claude-sonnet-latest` or `mistral/mistral-large-latest`. The full list is
available at `https://api.edenai.run/v3/models`.

Set your API key first:

```bash
export EDENAI_API_KEY=***
```

Use the model class or the string syntax:

```python
from agno.agent import Agent
from agno.models.edenai import EdenAI

agent = Agent(model=EdenAI(id="anthropic/claude-sonnet-latest"))
agent = Agent(model="edenai:anthropic/claude-sonnet-latest")
```

Eden AI specific options:

- `fallbacks`: up to 3 models tried in order if the primary model fails.
- `routing`: provider routing preferences (`sort`, `allowed_providers`, `allow_fallbacks`) for
  models served by several providers.
- `base_url="https://api.eu.edenai.run/v3"`: EU data residency endpoint.

For Eden AI expert models (OCR, speech, translation, image generation, moderation...), see
`EdenAITools` in `cookbook/91_tools/edenai_tools.py`.

Run examples with:

```bash
.venvs/demo/bin/python cookbook/90_models/edenai/<example>.py
```

| Example | Description |
| --- | --- |
| `basic.py` | Sync, streaming and async runs, with the class and string syntax |
| `tool_use.py` | Function calling with a toolkit |
| `structured_output.py` | Pydantic output schema |
| `image_agent.py` | Vision input |
| `fallbacks_and_routing.py` | Model fallbacks, provider routing and the EU endpoint |
