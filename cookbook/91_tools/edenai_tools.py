"""
Eden AI Tools
=============================

Demonstrates Eden AI expert model tools (OCR, translation, moderation, speech, image generation)
running on an Eden AI LLM, so a single EDENAI_API_KEY powers the whole agent.
"""

from agno.agent import Agent
from agno.models.edenai import EdenAI
from agno.tools.edenai import EdenAITools

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------

edenai_agent = Agent(
    name="Eden AI Expert Models Agent",
    model=EdenAI(id="openai/gpt-5.4-mini"),
    tools=[
        EdenAITools(
            ocr_model="ocr/ocr/mistral",
            translation_model="translation/automatic_translation/deepl",
            image_generation_model="image/generation/openai/gpt-image-1-mini",
            # Try Google if DeepL fails
            fallbacks={"translation/automatic_translation/deepl": ["google"]},
        )
    ],
    description="You are an assistant that reads documents, translates text and creates images and audio.",
    instructions=[
        "Use the Eden AI tools to process files, text and media.",
        "Return media URLs as raw text.",
    ],
    markdown=True,
)

# Any expert model (e.g. named entity recognition) can be called with call_expert_model
expert_agent = Agent(
    name="Eden AI Expert Model Caller",
    model=EdenAI(id="openai/gpt-5.4-mini"),
    tools=[EdenAITools(enable_call_expert_model=True)],
    instructions=[
        "Call Eden AI expert models with call_expert_model.",
        "Model strings use the format feature/subfeature/provider, e.g. text/named_entity_recognition/amazon.",
        "Pass the language code of the text in the input, e.g. {'text': '...', 'language': 'en'}.",
    ],
    markdown=True,
)

# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    edenai_agent.print_response(
        "Extract the text from "
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/ReceiptSwiss.jpg/500px-ReceiptSwiss.jpg "
        "and translate it to English."
    )
    edenai_agent.print_response("Generate an image of a lighthouse on a stormy coast")
    expert_agent.print_response(
        "List the named entities in: 'Eden AI is based in Montpellier, France.'"
    )
