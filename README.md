# Simple Rag Writer

Simple Rag Writer is a test-driven skeleton for a writing assistant that:

  - Orchestrates multiple LLM providers (OpenAI, OpenRouter, Gemini, etc.) via
    `litellm`.
  - Uses MCP servers as the document/reference retrieval layer.
  - Offers an interactive planning REPL (`srw -c config.yaml plan`) with
    slash commands for model switching, knowledge browsing, and context
    injection.
  - Executes declarative writing tasks from YAML for batch generation
    (`srw -c config.yaml run path/to/tasks/*.yaml`).
  - Replays past planning turns with their injected MCP context (`srw -c
    config.yaml replay --log <file> --turn <n>`).

The project is intentionally minimal: every feature is covered by tests, MCP
interactions are stubbed by fixtures, and real network calls are avoided unless
explicitly requested.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs `srw`, which is just a thin wrapper around the `simple_rag_writer`
package. From the repository root you can run `./srw ...` or rely on the shim to
add `src/` to `PYTHONPATH`.

## Configuration overview

The CLI requires a configuration YAML file that defines models, providers, MCP
servers, and logging/prompt policies. At minimum you must supply a default model
and the provider/model entries you plan on using.

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
mcp_servers:
  - id: "notes"
    command: ["mcp-notes-server"]
mcp_prompt_policy:
  default_mode: "raw_capped"
  raw_capped:
    max_items_per_reference: 3
    max_chars_per_item: 800
    max_total_chars: 8000
  summary:
    summarizer_model: "openai:gpt-4.1-mini"
    max_items_per_reference: 5
    summary_max_tokens: 512
logging:
  planning:
    enabled: true
    dir: "logs"
    include_mcp_events: true
    mcp_inline: true
```

Key configuration sections:

  - `providers`: map provider names to `type`, `api_key`, and optional overrides
    (`base_url`, `model_prefix`, etc.).
  - `models`: list of `ModelConfig` entries (id, provider, model_name, optional
    `params`, `label`, `tags`, and `max_context_tokens`).
  - `model_defaults`: shared generation parameters merged before each request.
  - `mcp_servers`: each server needs `id`, `command` (executable + args), and
    optional `auto_start`. The MCP client shells out to the command, performs a
    JSON-RPC handshake, and caches discovered tools via `tools/list`.
  - `mcp_prompt_policy`: controls how retrieved `NormalizedItem`s are transformed
    before injection. `raw_capped` trims and truncates, while `summary` uses a
    summarizer model and per-type prompts.
  - `logging.planning`: enables planning transcripts, inline MCP sections, and
    output location.

### Adding an MCP server

1. Choose an `id` that you will reference from `/use <id> ...` in the REPL and
   from `references` entries in task YAML.
2. Determine the exact command that starts your MCP server and split it into an
   argv-style list. The config **must** list every argument inside `command`.
3. Add the server under `mcp_servers`, optionally flipping `auto_start` to
   `false` if you prefer to start the server manually ahead of time.

Example (`uv tool run arxiv-mcp-server --storage-path ...`):

```yaml
mcp_servers:
  - id: "arxiv"
    command:
      - "uv"
      - "tool"
      - "run"
      - "arxiv-mcp-server"
      - "--storage-path"
      - "/home/you/.cache/arxiv-papers"
    auto_start: true
```

After saving the config you can run `srw -c config.yaml plan`, use `/sources` to
verify that the server is listed, and then call `/use arxiv <tool> "query" <n>`
to fetch data. The same `arxiv` id can be referenced from automated task YAML
via `references` entries.

## Usage

`srw` exposes three high-level commands.

### Planning mode (`srw -c config.yaml plan`)

Interactive planning mode spins up a REPL that records history, logs turns, and
builds prompts with optional MCP context. Slash commands include:

  - `/models`: list configured models, marking the active one.
  - `/model <id>`: switch the current model.
  - `/sources`: show configured MCP servers plus discovered tool names and
    descriptions.
  - `/use <server> <tool> "query" [limit]`: call an MCP tool and display results.
    You can also provide key/value args like `query:"text" paper_id:"1234"` to
    pass multiple parameters to the tool.
  - Planning and task prompts now include a `call_mcp_tool` function hint so the
    model can request additional MCP data on its own. The tool expects JSON with
    `server`, `tool`, and optional `params`, and the REPL/runner feeds the result
    back into the conversation automatically.
  - `/inject <indices>`: inject selected MCP items into the internal context
    buffer.
  - `/context`: preview the accumulated MCP context that will be prepended to
    the next prompt.
  - `/url <url> [label]`: fetch a URL, normalize it, and make it available for
    injection.
  - `/quit`/`/q`: exit planning mode.

The repl maintains a history window (default 5 turns) and logs MCP injections
and completions via `PlanningLogWriter`. `/context` shows the combined chunks so
you can review what will be prepended.

### Automated tasks (`srw -c config.yaml run tasks/*.yaml`)

Execute YAML-defined tasks that describe what to write and what references to
resolve. Each task can declare:

  - `id`, `title`, `description`: metadata used in prompts and logging.
  - `context`: optional outline references (TODO: outline loader).
  - `model`: override `config.default_model`.
  - `model_params`: per-task parameter overrides.
  - `references`: list of `McpReference` or `UrlReference` entries that specify
    servers/tools or URLs to fetch. MCP references can set `item_type` hints.
  - `mcp_error_mode`: `"skip_with_warning"` (default) or `"fail_task"`.
  - `output`: path for the generated Markdown draft.

`run_tasks_for_paths`:

  1. Expands glob patterns provided on the CLI.
  2. Loads each task and resolves MCP/URL references, normalizes payloads, and
     applies the configured prompt policy.
  3. Builds the final prompt with references and calls the ModelRegistry.
  4. Writes Markdown to the `output` path and reports progress via Rich.

### Replay mode (`srw -c config.yaml replay --log <file> --turn <n>`)

Replay rehydrates a planning turn by:

  - Parsing the Markdown log header for config/model metadata.
  - Reconstructing history (up to `HISTORY_WINDOW`) and the MCP `mcp-yaml`
    block.
  - Rebuilding the prompt exactly as it was sent originally (including inserted
    context) so you can inspect or rerun it (`--show-prompt`, `--run-model`).

## MCP integration

MCP servers are located with `AppConfig.mcp_servers`. Each entry must match the
`id` referenced in `/sources`, `/use`, and task references. The client:

  1. Spawns the configured command via `anyio`/`mcp.client.stdio`.
  2. Performs `initialize`, `tools/list`, and `tools/call` JSON-RPC
    negotiations.
  3. Extracts text/structured payloads from `CallToolResult` and normalizes
    them via `normalize_payload` before the prompt policy.

Tool discovery populates the `/sources` table with tool descriptions and titles,
and the `/use` command automatically injects the fetched items for later use.
MCP errors are surfaced to the REPL and respect `mcp_error_mode` during automated
runs.

You can also expose the local LLM as an MCP tool by describing it via
`llm_tool` and wiring that entry to a process that runs
`python -m simple_rag_writer.mcp.llm_tool --config config.yaml`. The `llm_tool`
section maps skill keywords to a single configured model (`skills: {reason:
or-qwen3-235b-a22b}`) and limits the models the tool may call. A matching MCP
server entry can be as simple as:

```yaml
llm_tool:
  id: "llm"
  tool_name: "llm-complete"
  title: "LLM skill completions"
  skills:
    reason: "or-qwen3-235b-a22b"
    summarize: "or-kimi-k2"
  default_skill: "reason"
  max_tokens_limit: 2048

mcp_servers:
  - id: "llm"
    command:
      - "python"
      - "srw_llm_tool.py"
      - "--config"
      - "config.yaml"
```

The server will advertise `skill` as an enum in `tools/list`, and calls to
`tool` in the MCP protocol may request `skill: "summarize"` to ensure only the
allowed model receives that prompt.

## Development & Testing

This repo follows strict TDD. For every change:

  - Start with an acceptance test under `tests/` (no new functionality without
    a test).
  - Mock/fixture MCP servers; `tests/mcp_fixtures/notes_server.py` demonstrates
    a lightweight JSON-RPC server that exposes `/statuses` and `/call`.
  - Run `pytest` (the suite avoids real network activity by mocking `litellm`
    completions and HTTP requests).
  - Update `codex/TASKS/*.yaml` only when explicitly instructed—these describe
    the master spec.

Commands:

```bash
pytest  # runs everything
```

Contributions should respect the project structure (`src/simple_rag_writer/` for
logic, `tests/` for behavior) and keep formatting consistent (2-space indentation
for Python/YAML).
