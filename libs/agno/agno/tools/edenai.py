"""Eden AI expert model tools: OCR, translation, moderation, speech and image generation.

Uses the Eden AI V3 Universal AI endpoint (https://www.edenai.co/docs/v3/expert-models).
Models are addressed as "feature/subfeature/provider[/model]", e.g. "ocr/ocr/mistral".
For LLMs, use `agno.models.edenai.EdenAI` instead.
"""

import asyncio
import json
import time
from os import getenv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import httpx

from agno.media import Audio, Image
from agno.tools import Toolkit
from agno.tools.function import ToolResult
from agno.utils.log import log_debug, log_error

EDENAI_BASE_URL = "https://api.edenai.run/v3"

# Minimum budget required to start another poll request. Polling stops instead of
# issuing a request that cannot complete before max_wait_time elapses.
MIN_POLL_REQUEST_SECONDS = 1.0


class EdenAIError(Exception):
    """Raised when Eden AI reports a failed expert model call."""


class EdenAITools(Toolkit):
    """Tools for Eden AI expert models.

    Set EDENAI_API_KEY or pass api_key. Each tool uses a configurable model string
    ("feature/subfeature/provider[/model]"); see https://api.edenai.run/v3/info for the catalog.
    Files are passed as a public URL or as a file ID returned by `upload_file`.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = EDENAI_BASE_URL,
        ocr_model: str = "ocr/ocr/mistral",
        translation_model: str = "translation/automatic_translation/google",
        moderation_model: str = "text/moderation/openai",
        text_to_speech_model: str = "audio/tts/openai/gpt-4o-mini-tts",
        image_generation_model: str = "image/generation/openai/gpt-image-1-mini",
        speech_to_text_model: str = "audio/speech_to_text_async/deepgram/nova-3",
        fallbacks: Optional[Dict[str, List[str]]] = None,
        timeout: float = 60,
        poll_interval: float = 2,
        max_wait_time: float = 600,
        enable_extract_text: bool = True,
        enable_translate_text: bool = True,
        enable_moderate_text: bool = True,
        enable_text_to_speech: bool = True,
        enable_generate_image: bool = True,
        enable_transcribe_audio: bool = True,
        enable_call_expert_model: bool = False,
        all: bool = False,
        **kwargs,
    ):
        """
        Args:
            api_key: Eden AI API key. Defaults to the EDENAI_API_KEY environment variable.
            base_url: API base URL. Use "https://api.eu.edenai.run/v3" for EU data residency.
            ocr_model: Model used by extract_text.
            translation_model: Model used by translate_text.
            moderation_model: Model used by moderate_text.
            text_to_speech_model: Model used by text_to_speech.
            image_generation_model: Model used by generate_image.
            speech_to_text_model: Async model used by transcribe_audio.
            fallbacks: Optional fallback providers per model string, e.g.
                {"ocr/ocr/mistral": ["ocr/ocr/google"]}. At most 3 per model.
            timeout: Timeout in seconds for each HTTP request.
            poll_interval: Seconds between polls of an async job.
            max_wait_time: Maximum seconds to wait for an async job.
            enable_call_expert_model: Register call_expert_model, which lets the agent call any expert model.
        """
        self.api_key = api_key or getenv("EDENAI_API_KEY")
        if not self.api_key:
            log_error("EDENAI_API_KEY not set. Please set the EDENAI_API_KEY environment variable.")

        self.base_url = base_url.rstrip("/")
        self.ocr_model = ocr_model
        self.translation_model = translation_model
        self.moderation_model = moderation_model
        self.text_to_speech_model = text_to_speech_model
        self.image_generation_model = image_generation_model
        self.speech_to_text_model = speech_to_text_model
        self.fallbacks = fallbacks or {}
        self.request_timeout = timeout
        self.poll_interval = poll_interval
        self.max_wait_time = max_wait_time

        tools: List[Any] = []
        async_tools: List[Any] = []
        if all or enable_extract_text:
            tools.append(self.extract_text)
            async_tools.append((self.aextract_text, "extract_text"))
        if all or enable_translate_text:
            tools.append(self.translate_text)
            async_tools.append((self.atranslate_text, "translate_text"))
        if all or enable_moderate_text:
            tools.append(self.moderate_text)
            async_tools.append((self.amoderate_text, "moderate_text"))
        if all or enable_text_to_speech:
            tools.append(self.text_to_speech)
            async_tools.append((self.atext_to_speech, "text_to_speech"))
        if all or enable_generate_image:
            tools.append(self.generate_image)
            async_tools.append((self.agenerate_image, "generate_image"))
        if all or enable_transcribe_audio:
            tools.append(self.transcribe_audio)
            async_tools.append((self.atranscribe_audio, "transcribe_audio"))
        if all or enable_call_expert_model:
            tools.append(self.call_expert_model)
            async_tools.append((self.acall_expert_model, "call_expert_model"))

        super().__init__(name="edenai_tools", tools=tools, async_tools=async_tools, **kwargs)

    # ---------------------------------------------------------------------------
    # HTTP helpers
    # ---------------------------------------------------------------------------

    @property
    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _build_body(self, model: str, input: Dict[str, Any]) -> Dict[str, Any]:
        # "compact" drops base64 media and dense position data, keeping the *_resource_url links
        body: Dict[str, Any] = {
            "model": model,
            "input": {k: v for k, v in input.items() if v is not None},
            "output_format": "compact",
        }
        if self.fallbacks.get(model):
            body["fallbacks"] = self.fallbacks[model]
        return body

    @staticmethod
    def _is_async_model(model: str) -> bool:
        parts = model.split("/")
        return len(parts) > 1 and parts[1].endswith("_async")

    @staticmethod
    def _get_output(data: Dict[str, Any]) -> Any:
        if data.get("status") == "fail":
            error = data.get("error") or {}
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise EdenAIError(message or "Eden AI request failed")
        log_debug(f"Eden AI {data.get('feature')}/{data.get('subfeature')} served by {data.get('provider')}")
        return data.get("output")

    @staticmethod
    def _poll_url(base_url: str, job: Dict[str, Any]) -> str:
        job_id = job.get("public_id")
        if not job_id:
            raise EdenAIError("Eden AI did not return a job ID")
        return f"{base_url}/universal-ai/async/{job_id}"

    def _run(self, model: str, input: Dict[str, Any]) -> Any:
        if self._is_async_model(model):
            return self._run_job(model, input)
        response = httpx.post(
            f"{self.base_url}/universal-ai",
            headers=self._headers,
            json=self._build_body(model, input),
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        return self._get_output(response.json())

    def _run_job(self, model: str, input: Dict[str, Any]) -> Any:
        response = httpx.post(
            f"{self.base_url}/universal-ai/async",
            headers=self._headers,
            json=self._build_body(model, input),
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        job = response.json()
        poll_url = self._poll_url(self.base_url, job)

        deadline = time.monotonic() + self.max_wait_time
        while job.get("status") == "processing":
            remaining = deadline - time.monotonic()
            if remaining < MIN_POLL_REQUEST_SECONDS:
                raise EdenAIError(f"Eden AI job timed out after {self.max_wait_time} seconds")
            time.sleep(min(self.poll_interval, remaining))
            poll = httpx.get(
                poll_url,
                headers=self._headers,
                params={"output_format": "compact"},
                timeout=min(self.request_timeout, max(deadline - time.monotonic(), MIN_POLL_REQUEST_SECONDS)),
            )
            poll.raise_for_status()
            job = poll.json()
        return self._get_output(job)

    async def _arun(self, model: str, input: Dict[str, Any]) -> Any:
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            if not self._is_async_model(model):
                response = await client.post(
                    f"{self.base_url}/universal-ai", headers=self._headers, json=self._build_body(model, input)
                )
                response.raise_for_status()
                return self._get_output(response.json())

            response = await client.post(
                f"{self.base_url}/universal-ai/async", headers=self._headers, json=self._build_body(model, input)
            )
            response.raise_for_status()
            job = response.json()
            poll_url = self._poll_url(self.base_url, job)

            deadline = time.monotonic() + self.max_wait_time
            while job.get("status") == "processing":
                remaining = deadline - time.monotonic()
                if remaining < MIN_POLL_REQUEST_SECONDS:
                    raise EdenAIError(f"Eden AI job timed out after {self.max_wait_time} seconds")
                await asyncio.sleep(min(self.poll_interval, remaining))
                poll = await client.get(
                    poll_url,
                    headers=self._headers,
                    params={"output_format": "compact"},
                    timeout=min(self.request_timeout, max(deadline - time.monotonic(), MIN_POLL_REQUEST_SECONDS)),
                )
                poll.raise_for_status()
                job = poll.json()
            return self._get_output(job)

    def _error(self, action: str, error: Exception) -> str:
        if isinstance(error, httpx.HTTPStatusError):
            detail = f"HTTP {error.response.status_code}: {error.response.text[:500]}"
        else:
            detail = str(error)
        log_error(f"Eden AI {action} failed: {detail}")
        return f"Error: Eden AI {action} failed: {detail}"

    def upload_file(self, path: Union[str, Path], purpose: Optional[str] = None) -> str:
        """Upload a local file to Eden AI and return its file ID.

        The ID can be passed to the tools in place of a URL. Uploaded files expire after 7 days.
        This is a helper for application code and is not registered as an agent tool.

        Args:
            path: Path of the local file to upload.
            purpose: Optional intent tag, e.g. "ocr" or "speech".

        Returns:
            The Eden AI file ID.
        """
        file_path = Path(path)
        with file_path.open("rb") as f:
            response = httpx.post(
                f"{self.base_url}/upload",
                headers=self._headers,
                files={"file": (file_path.name, f)},
                data={"purpose": purpose} if purpose else None,
                timeout=self.request_timeout,
            )
        response.raise_for_status()
        return response.json()["file_id"]

    async def aupload_file(self, path: Union[str, Path], purpose: Optional[str] = None) -> str:
        """Upload a local file to Eden AI and return its file ID.

        The ID can be passed to the tools in place of a URL. Uploaded files expire after 7 days.
        This is a helper for application code and is not registered as an agent tool.

        Args:
            path: Path of the local file to upload.
            purpose: Optional intent tag, e.g. "ocr" or "speech".

        Returns:
            The Eden AI file ID.
        """
        file_path = Path(path)
        content = await asyncio.to_thread(file_path.read_bytes)
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            response = await client.post(
                f"{self.base_url}/upload",
                headers=self._headers,
                files={"file": (file_path.name, content)},
                data={"purpose": purpose} if purpose else None,
            )
        response.raise_for_status()
        return response.json()["file_id"]

    # ---------------------------------------------------------------------------
    # Result formatting
    # ---------------------------------------------------------------------------

    @staticmethod
    def _text(output: Any) -> str:
        if isinstance(output, dict) and isinstance(output.get("text"), str):
            return output["text"]
        return json.dumps(output)

    @staticmethod
    def _audio_result(output: Any) -> ToolResult:
        url = output.get("audio_resource_url") if isinstance(output, dict) else None
        if not url:
            return ToolResult(content="Error: Eden AI returned no audio")
        return ToolResult(content=f"Audio generated successfully: {url}", audios=[Audio(id=str(uuid4()), url=url)])

    @staticmethod
    def _image_result(output: Any, prompt: str) -> ToolResult:
        items = output.get("items", [output]) if isinstance(output, dict) else []
        urls = [
            item["image_resource_url"] for item in items if isinstance(item, dict) and item.get("image_resource_url")
        ]
        if not urls:
            return ToolResult(content="Error: Eden AI returned no image")
        images = [Image(id=str(uuid4()), url=url, original_prompt=prompt) for url in urls]
        return ToolResult(content=f"Generated {len(images)} image(s) successfully: {', '.join(urls)}", images=images)

    # ---------------------------------------------------------------------------
    # Tools
    # ---------------------------------------------------------------------------

    def extract_text(self, file: str, language: Optional[str] = None) -> str:
        """Use this function to extract the text from an image or a PDF document (OCR).

        Args:
            file (str): Public URL of the image or PDF, or an Eden AI file ID.
            language (str, optional): Language code of the document, e.g. "en". Most providers detect it.

        Returns:
            str: The extracted text, or an error message.
        """
        try:
            return self._text(self._run(self.ocr_model, {"file": file, "language": language}))
        except Exception as e:
            return self._error("text extraction", e)

    async def aextract_text(self, file: str, language: Optional[str] = None) -> str:
        """Use this function to extract the text from an image or a PDF document (OCR).

        Args:
            file (str): Public URL of the image or PDF, or an Eden AI file ID.
            language (str, optional): Language code of the document, e.g. "en". Most providers detect it.

        Returns:
            str: The extracted text, or an error message.
        """
        try:
            return self._text(await self._arun(self.ocr_model, {"file": file, "language": language}))
        except Exception as e:
            return self._error("text extraction", e)

    def translate_text(self, text: str, target_language: str, source_language: Optional[str] = None) -> str:
        """Use this function to translate text into another language.

        Args:
            text (str): The text to translate.
            target_language (str): Target language code, e.g. "fr".
            source_language (str, optional): Source language code, e.g. "en". Detected when omitted.

        Returns:
            str: The translated text, or an error message.
        """
        input = {"text": text, "target_language": target_language, "source_language": source_language}
        try:
            return self._text(self._run(self.translation_model, input))
        except Exception as e:
            return self._error("translation", e)

    async def atranslate_text(self, text: str, target_language: str, source_language: Optional[str] = None) -> str:
        """Use this function to translate text into another language.

        Args:
            text (str): The text to translate.
            target_language (str): Target language code, e.g. "fr".
            source_language (str, optional): Source language code, e.g. "en". Detected when omitted.

        Returns:
            str: The translated text, or an error message.
        """
        input = {"text": text, "target_language": target_language, "source_language": source_language}
        try:
            return self._text(await self._arun(self.translation_model, input))
        except Exception as e:
            return self._error("translation", e)

    def moderate_text(self, text: str) -> str:
        """Use this function to check text for offensive, sexual, violent, hateful or self-harm content.

        Args:
            text (str): The text to moderate.

        Returns:
            str: JSON with the detected categories and their likelihood, or an error message.
        """
        try:
            return json.dumps(self._run(self.moderation_model, {"text": text}))
        except Exception as e:
            return self._error("moderation", e)

    async def amoderate_text(self, text: str) -> str:
        """Use this function to check text for offensive, sexual, violent, hateful or self-harm content.

        Args:
            text (str): The text to moderate.

        Returns:
            str: JSON with the detected categories and their likelihood, or an error message.
        """
        try:
            return json.dumps(await self._arun(self.moderation_model, {"text": text}))
        except Exception as e:
            return self._error("moderation", e)

    def text_to_speech(self, text: str, voice: Optional[str] = None) -> ToolResult:
        """Use this function to convert text to spoken audio.

        Args:
            text (str): The text to speak.
            voice (str, optional): Provider voice name or ID. The provider default is used when omitted.

        Returns:
            ToolResult: The generated audio, or an error message.
        """
        try:
            return self._audio_result(self._run(self.text_to_speech_model, {"text": text, "voice": voice}))
        except Exception as e:
            return ToolResult(content=self._error("text to speech", e))

    async def atext_to_speech(self, text: str, voice: Optional[str] = None) -> ToolResult:
        """Use this function to convert text to spoken audio.

        Args:
            text (str): The text to speak.
            voice (str, optional): Provider voice name or ID. The provider default is used when omitted.

        Returns:
            ToolResult: The generated audio, or an error message.
        """
        try:
            return self._audio_result(await self._arun(self.text_to_speech_model, {"text": text, "voice": voice}))
        except Exception as e:
            return ToolResult(content=self._error("text to speech", e))

    def generate_image(self, prompt: str, resolution: Optional[str] = None) -> ToolResult:
        """Use this function to generate an image from a text prompt.

        Args:
            prompt (str): A text description of the image to generate.
            resolution (str, optional): Image size such as "1024x1024". The provider default is used when omitted.

        Returns:
            ToolResult: The generated image, or an error message.
        """
        try:
            output = self._run(self.image_generation_model, {"text": prompt, "resolution": resolution})
            return self._image_result(output, prompt)
        except Exception as e:
            return ToolResult(content=self._error("image generation", e))

    async def agenerate_image(self, prompt: str, resolution: Optional[str] = None) -> ToolResult:
        """Use this function to generate an image from a text prompt.

        Args:
            prompt (str): A text description of the image to generate.
            resolution (str, optional): Image size such as "1024x1024". The provider default is used when omitted.

        Returns:
            ToolResult: The generated image, or an error message.
        """
        try:
            output = await self._arun(self.image_generation_model, {"text": prompt, "resolution": resolution})
            return self._image_result(output, prompt)
        except Exception as e:
            return ToolResult(content=self._error("image generation", e))

    def transcribe_audio(self, file: str, language: Optional[str] = None) -> str:
        """Use this function to transcribe speech from an audio or video file to text.

        Args:
            file (str): Public URL of the audio file, or an Eden AI file ID.
            language (str, optional): Language code of the speech, e.g. "en".

        Returns:
            str: The transcription, or an error message.
        """
        try:
            return self._text(self._run(self.speech_to_text_model, {"file": file, "language": language}))
        except Exception as e:
            return self._error("transcription", e)

    async def atranscribe_audio(self, file: str, language: Optional[str] = None) -> str:
        """Use this function to transcribe speech from an audio or video file to text.

        Args:
            file (str): Public URL of the audio file, or an Eden AI file ID.
            language (str, optional): Language code of the speech, e.g. "en".

        Returns:
            str: The transcription, or an error message.
        """
        try:
            return self._text(await self._arun(self.speech_to_text_model, {"file": file, "language": language}))
        except Exception as e:
            return self._error("transcription", e)

    def call_expert_model(self, model: str, input: Dict[str, Any]) -> str:
        """Use this function to call any Eden AI expert model, such as named entity recognition,
        object detection, invoice parsing or web search. Async models (subfeature ending in "_async") are polled.

        Args:
            model (str): Model string "feature/subfeature/provider[/model]", e.g. "text/named_entity_recognition/amazon".
            input (dict): Feature specific input, e.g. {"text": "...", "language": "en"} or
                {"file": "<url or file ID>"}. Some providers require a language code.

        Returns:
            str: JSON output of the model, or an error message.
        """
        try:
            return json.dumps(self._run(model, input))
        except Exception as e:
            return self._error(model, e)

    async def acall_expert_model(self, model: str, input: Dict[str, Any]) -> str:
        """Use this function to call any Eden AI expert model, such as named entity recognition,
        object detection, invoice parsing or web search. Async models (subfeature ending in "_async") are polled.

        Args:
            model (str): Model string "feature/subfeature/provider[/model]", e.g. "text/named_entity_recognition/amazon".
            input (dict): Feature specific input, e.g. {"text": "...", "language": "en"} or
                {"file": "<url or file ID>"}. Some providers require a language code.

        Returns:
            str: JSON output of the model, or an error message.
        """
        try:
            return json.dumps(await self._arun(model, input))
        except Exception as e:
            return self._error(model, e)
