# TEST_LOG

Run live against `https://api.edenai.run/v3` on 2026-10-09.

### basic.py

**Status:** PASS

**Description:** Sync, sync + streaming, async and async + streaming runs, with the `EdenAI` class (`openai/gpt-5.4-mini`) and the `edenai:anthropic/claude-sonnet-latest` string syntax.

**Result:** All four runs returned a response.

---

### tool_use.py

**Status:** PASS

**Description:** Function calling with `WebSearchTools`, sync and async streaming.

**Result:** The model called `search_news` and answered from the results.

---

### structured_output.py

**Status:** PASS

**Description:** `output_schema=MovieScript` through the `json_schema` response format.

**Result:** A valid `MovieScript` object was returned.

---

### image_agent.py

**Status:** PASS

**Description:** Vision input (image URL) with `google/gemini-flash-latest`.

**Result:** The image was described correctly.

---

### fallbacks_and_routing.py

**Status:** PASS

**Description:** `fallbacks`, `routing` (`sort=latency`, `allowed_providers`) with a provider-less model id, and the EU endpoint `https://api.eu.edenai.run/v3`.

**Result:** All three agents returned a response.

---
