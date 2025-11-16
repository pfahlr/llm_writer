# Simple Rag Writer

Early scaffold for a slim writing assistant that:
- Uses multiple LLM providers (OpenAI, OpenRouter, Gemini) via `litellm`.
- Uses MCP servers as its only retrieval/knowledge layer.
- Supports interactive planning (`srw -c config.yaml plan`).
- Supports automated task execution from YAML (`srw -c config.yaml run tasks/*.yaml`).
- Supports replaying planning prompts from logs (`srw -c config.yaml replay --log ... --turn ...`).

This is an incomplete scaffold intended for test-driven development.
Many components are stubs with TODOs.
