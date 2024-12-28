# edenai Cookbook

> Note: Fork and clone this repository if needed

### 1. Create and activate a virtual environment

```shell
python3 -m venv ~/.venvs/aienv
source ~/.venvs/aienv/bin/activate
```

### 2. Export your `EDENAI_API_KEY`

```shell
export EDENAI_API_KEY=***
```

### 3. Install libraries

```shell
pip install -U openai duckduckgo-search duckdb yfinance phidata
```

### 4. Run Agent without Tools

- Streaming on

```shell
python cookbook/providers/edenai/basic_stream.py
```

- Streaming off

```shell
python cookbook/providers/edenai/basic.py
```

### 5. Run Agent with Tools

- DuckDuckGo Search with streaming on

```shell
python cookbook/providers/edenai/agent_stream.py
```

- DuckDuckGo Search without streaming

```shell
python cookbook/providers/edenai/agent.py
```

- Finance Agent

```shell
python cookbook/providers/edenai/finance_agent.py
```

- Data Analyst

```shell
python cookbook/providers/edenai/data_analyst.py
```

- Web Search

```shell
python cookbook/providers/edenai/web_search.py
```

### 6. Run Agent that returns structured output

```shell
python cookbook/providers/edenai/structured_output.py
```

