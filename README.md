# Simple Rag Writer

Early scaffold for a slim writing assistant that:
- Uses multiple LLM providers (OpenAI, OpenRouter, Gemini) via `litellm`.
- Uses MCP servers as its only retrieval/knowledge layer.
- Supports interactive planning (`srw -c config.yaml plan`).
- Supports automated task execution from YAML (`srw -c config.yaml run tasks/*.yaml`).
- Supports replaying planning prompts from logs (`srw -c config.yaml replay --log ... --turn ...`).

This is an incomplete scaffold intended for test-driven development.
Many components are stubs with TODOs.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
srw --help
srw -c config.yaml plan
```

The CLI requires a configuration YAML file that declares providers, models,
and MCP settings. Use `srw -c config.yaml run path/to/tasks/*.yaml` to execute
tasks once a config is available, or `srw replay` to inspect planning logs.

## Example Config

```yaml
default_model: "openai:gpt-4.1-mini"
providers:
  openai:
    type: "openai"
    api_key_env: "OPENAI_API_KEY"
model_defaults:
  temperature: 0.3
models:
  - id: "openai:gpt-4.1-mini"
    provider: "openai"
    model_name: "gpt-4.1-mini"
mcp_prompt_policy:
  default_mode: "raw_capped"
logging:
  planning:
    enabled: true
```
