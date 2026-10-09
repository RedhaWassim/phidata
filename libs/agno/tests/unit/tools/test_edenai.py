"""Unit tests for EdenAITools, the Eden AI V3 Universal AI (expert models) toolkit."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from agno.tools.edenai import EDENAI_BASE_URL, EdenAITools


def _response(data, method="POST", status_code=200):
    request = httpx.Request(method, "https://api.edenai.run/v3/universal-ai")
    return httpx.Response(status_code, json=data, request=request)


def _success(output, feature="ocr", subfeature="ocr"):
    return {
        "status": "success",
        "cost": "0.001",
        "provider": "mistral",
        "feature": feature,
        "subfeature": subfeature,
        "output": output,
    }


def _async_client(client):
    factory = MagicMock()
    factory.return_value.__aenter__.return_value = client
    factory.return_value.__aexit__.return_value = False
    return factory


def test_default_tools_registered():
    tools = EdenAITools(api_key="test-key")

    assert tools.base_url == EDENAI_BASE_URL
    for name in (
        "extract_text",
        "translate_text",
        "moderate_text",
        "text_to_speech",
        "generate_image",
        "transcribe_audio",
    ):
        assert name in tools.functions
        assert name in tools.async_functions
    assert "call_expert_model" not in tools.functions


def test_flags_select_tools():
    tools = EdenAITools(
        api_key="test-key",
        enable_extract_text=False,
        enable_translate_text=False,
        enable_moderate_text=False,
        enable_text_to_speech=False,
        enable_generate_image=False,
        enable_transcribe_audio=False,
        enable_call_expert_model=True,
    )

    assert list(tools.functions) == ["call_expert_model"]


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("EDENAI_API_KEY", "env-key")

    assert EdenAITools().api_key == "env-key"


def test_extract_text_sends_universal_ai_request():
    tools = EdenAITools(api_key="test-key")

    with patch("agno.tools.edenai.httpx.post", return_value=_response(_success({"text": "Invoice #42"}))) as post:
        result = tools.extract_text("https://example.com/invoice.png")

    assert result == "Invoice #42"
    args, kwargs = post.call_args
    assert args[0] == "https://api.edenai.run/v3/universal-ai"
    assert kwargs["headers"] == {"Authorization": "Bearer test-key"}
    assert kwargs["json"] == {
        "model": "ocr/ocr/mistral",
        "input": {"file": "https://example.com/invoice.png"},
        "output_format": "compact",
    }


def test_translate_text_and_fallbacks():
    tools = EdenAITools(
        api_key="test-key",
        translation_model="translation/automatic_translation/deepl",
        fallbacks={"translation/automatic_translation/deepl": ["google"]},
    )
    output = _success({"text": "Bonjour"}, feature="translation", subfeature="automatic_translation")

    with patch("agno.tools.edenai.httpx.post", return_value=_response(output)) as post:
        result = tools.translate_text("Hello", target_language="fr", source_language="en")

    assert result == "Bonjour"
    assert post.call_args.kwargs["json"] == {
        "model": "translation/automatic_translation/deepl",
        "input": {"text": "Hello", "target_language": "fr", "source_language": "en"},
        "output_format": "compact",
        "fallbacks": ["google"],
    }


def test_moderate_text_returns_json():
    tools = EdenAITools(api_key="test-key")
    output = {"nsfw_likelihood": 1, "items": [{"label": "Toxic", "likelihood": 1}]}

    with patch("agno.tools.edenai.httpx.post", return_value=_response(_success(output, "text", "moderation"))):
        result = tools.moderate_text("some text")

    assert json.loads(result) == output


def test_failed_status_returns_error():
    tools = EdenAITools(api_key="test-key")
    failed = {
        "status": "fail",
        "cost": "0",
        "provider": "mistral",
        "feature": "ocr",
        "subfeature": "ocr",
        "output": None,
        "error": {"message": "Unsupported file type"},
    }

    with patch("agno.tools.edenai.httpx.post", return_value=_response(failed)):
        result = tools.extract_text("https://example.com/file.xyz")

    assert result == "Error: Eden AI text extraction failed: Unsupported file type"


def test_http_error_returns_error():
    tools = EdenAITools(api_key="test-key")

    with patch("agno.tools.edenai.httpx.post", return_value=_response({"detail": "Invalid token"}, status_code=401)):
        result = tools.moderate_text("some text")

    assert result.startswith("Error: Eden AI moderation failed: HTTP 401")
    assert "test-key" not in result


def test_text_to_speech_returns_audio_artifact():
    tools = EdenAITools(api_key="test-key")
    output = _success({"audio_resource_url": "https://cdn.edenai.run/audio.mp3"}, "audio", "tts")

    with patch("agno.tools.edenai.httpx.post", return_value=_response(output)) as post:
        result = tools.text_to_speech("Hello there", voice="alloy")

    assert result.audios is not None
    assert result.audios[0].url == "https://cdn.edenai.run/audio.mp3"
    assert post.call_args.kwargs["json"]["input"] == {"text": "Hello there", "voice": "alloy"}


def test_generate_image_returns_image_artifact():
    tools = EdenAITools(api_key="test-key")
    output = _success({"image_resource_url": "https://cdn.edenai.run/image.png"}, "image", "generation")

    with patch("agno.tools.edenai.httpx.post", return_value=_response(output)):
        result = tools.generate_image("A lighthouse at dusk", resolution="1024x1024")

    assert result.images is not None
    assert result.images[0].url == "https://cdn.edenai.run/image.png"
    assert result.images[0].original_prompt == "A lighthouse at dusk"


def test_generate_image_without_url_returns_error():
    tools = EdenAITools(api_key="test-key")

    with patch("agno.tools.edenai.httpx.post", return_value=_response(_success({}, "image", "generation"))):
        result = tools.generate_image("A lighthouse at dusk")

    assert result.content == "Error: Eden AI returned no image"
    assert result.images is None


def test_transcribe_audio_polls_async_job():
    tools = EdenAITools(api_key="test-key", poll_interval=0, max_wait_time=10)
    job = {"status": "processing", "cost": "0", "provider": "deepgram", "feature": "audio"}
    job.update({"subfeature": "speech_to_text_async", "public_id": "job-123", "created_at": "2026-10-09T00:00:00Z"})
    done = dict(job, status="success", output={"text": "Hello world", "diarization": {"total_speakers": 1}})

    with (
        patch("agno.tools.edenai.httpx.post", return_value=_response(job)) as post,
        patch("agno.tools.edenai.httpx.get", side_effect=[_response(job, "GET"), _response(done, "GET")]) as get,
        patch("agno.tools.edenai.time.sleep"),
    ):
        result = tools.transcribe_audio("https://example.com/audio.mp3", language="en")

    assert result == "Hello world"
    assert post.call_args.args[0] == "https://api.edenai.run/v3/universal-ai/async"
    assert post.call_args.kwargs["json"]["model"] == "audio/speech_to_text_async/deepgram/nova-3"
    assert get.call_count == 2
    assert get.call_args.args[0] == "https://api.edenai.run/v3/universal-ai/async/job-123"


def test_async_job_times_out():
    tools = EdenAITools(api_key="test-key", poll_interval=5, max_wait_time=10)
    job = {"status": "processing", "public_id": "job-123"}
    clock = {"now": 100.0}

    def fake_sleep(seconds):
        clock["now"] += seconds

    def fake_get(*args, **kwargs):
        clock["now"] += 4.5
        return _response(job, "GET")

    with (
        patch("agno.tools.edenai.httpx.post", return_value=_response(job)),
        patch("agno.tools.edenai.httpx.get", side_effect=fake_get),
        patch("agno.tools.edenai.time.monotonic", side_effect=lambda: clock["now"]),
        patch("agno.tools.edenai.time.sleep", side_effect=fake_sleep),
    ):
        result = tools.transcribe_audio("https://example.com/audio.mp3")

    assert result == "Error: Eden AI transcription failed: Eden AI job timed out after 10 seconds"


def test_call_expert_model_routes_async_models_to_jobs():
    tools = EdenAITools(api_key="test-key", enable_call_expert_model=True)
    done = {"status": "success", "public_id": "job-1", "output": {"raw_text": "page 1"}}

    with patch("agno.tools.edenai.httpx.post", return_value=_response(done)) as post:
        result = tools.call_expert_model("ocr/ocr_async/amazon", {"file": "https://example.com/doc.pdf"})

    assert json.loads(result) == {"raw_text": "page 1"}
    assert post.call_args.args[0] == "https://api.edenai.run/v3/universal-ai/async"


def test_eu_base_url():
    tools = EdenAITools(api_key="test-key", base_url="https://api.eu.edenai.run/v3/")

    with patch("agno.tools.edenai.httpx.post", return_value=_response(_success({"text": "ok"}))) as post:
        tools.extract_text("https://example.com/a.png")

    assert post.call_args.args[0] == "https://api.eu.edenai.run/v3/universal-ai"


def test_upload_file(tmp_path):
    tools = EdenAITools(api_key="test-key")
    document = tmp_path / "invoice.pdf"
    document.write_bytes(b"%PDF-1.4")

    with patch("agno.tools.edenai.httpx.post", return_value=_response({"file_id": "file-123"})) as post:
        file_id = tools.upload_file(document, purpose="ocr")

    assert file_id == "file-123"
    assert post.call_args.args[0] == "https://api.edenai.run/v3/upload"
    assert post.call_args.kwargs["data"] == {"purpose": "ocr"}


async def test_aupload_file(tmp_path):
    tools = EdenAITools(api_key="test-key")
    document = tmp_path / "audio.mp3"
    document.write_bytes(b"ID3")
    client = MagicMock()
    client.post = AsyncMock(return_value=_response({"file_id": "file-456"}))

    with patch("agno.tools.edenai.httpx.AsyncClient", _async_client(client)):
        file_id = await tools.aupload_file(document)

    assert file_id == "file-456"
    assert client.post.await_args.args[0] == "https://api.edenai.run/v3/upload"
    assert client.post.await_args.kwargs["files"] == {"file": ("audio.mp3", b"ID3")}


async def test_aextract_text():
    tools = EdenAITools(api_key="test-key")
    client = MagicMock()
    client.post = AsyncMock(return_value=_response(_success({"text": "Invoice #42"})))

    with patch("agno.tools.edenai.httpx.AsyncClient", _async_client(client)):
        result = await tools.aextract_text("https://example.com/invoice.png")

    assert result == "Invoice #42"
    assert client.post.await_args.args[0] == "https://api.edenai.run/v3/universal-ai"


async def test_agenerate_image():
    tools = EdenAITools(api_key="test-key")
    client = MagicMock()
    output = _success({"image_resource_url": "https://cdn.edenai.run/image.png"}, "image", "generation")
    client.post = AsyncMock(return_value=_response(output))

    with patch("agno.tools.edenai.httpx.AsyncClient", _async_client(client)):
        result = await tools.agenerate_image("A lighthouse at dusk")

    assert result.images is not None
    assert result.images[0].url == "https://cdn.edenai.run/image.png"


async def test_atranscribe_audio_polls_async_job():
    tools = EdenAITools(api_key="test-key", poll_interval=0, max_wait_time=10)
    job = {"status": "processing", "public_id": "job-123"}
    done = dict(job, status="success", output={"text": "Hello world"})
    client = MagicMock()
    client.post = AsyncMock(return_value=_response(job))
    client.get = AsyncMock(side_effect=[_response(job, "GET"), _response(done, "GET")])

    with (
        patch("agno.tools.edenai.httpx.AsyncClient", _async_client(client)),
        patch("agno.tools.edenai.asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await tools.atranscribe_audio("https://example.com/audio.mp3")

    assert result == "Hello world"
    assert client.post.await_args.args[0] == "https://api.edenai.run/v3/universal-ai/async"
    assert client.get.await_count == 2
