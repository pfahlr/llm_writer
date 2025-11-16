# AGENTS.md — Codex Execution Policy for simple-rag-writer

## Role

You are the **CODEX Agent** working on the `simple-rag-writer` project.
Your job is to complete each task in `codex/TASKS/*.yaml` **end-to-end**
under strict **Test-Driven Development (TDD)** and **Spec-as-Source-of-Truth**
rules.

The YAML task files in `codex/TASKS/` and this AGENTS.md are your primary
specifications. When in doubt, favor these documents over comments elsewhere.

---

## Global Rules

1. **TDD First (No Untested Code)**
   - For each task:
     - Start by reading the corresponding `codex/TASKS/<id>_*.yaml` file
       carefully.
     - Write or update **failing tests first** in `tests/` that express the
       required behavior.
     - Only then implement or modify the production code to make those tests
       pass.
   - If tests already exist but are `pytest.skip(...)`, you may:
     - Remove the skip and update the test as needed, or
     - Add new tests that better capture the behavior described in the task.
   - Never leave new functionality without tests.

2. **Scope Discipline**
   - Implement **only** what is required to satisfy the current task’s
     description, goals, and tests.
   - Do not anticipate future tasks or add speculative features.
   - If you discover missing or ambiguous requirements, document them in code
     comments or docstrings and then make the smallest reasonable assumption.

3. **Use Existing Structure and Style**
   - Respect the existing project structure and naming:
     - Code lives under `src/simple_rag_writer/`.
     - Tests live under `tests/`.
     - Codex tasks live under `codex/TASKS/`.
   - Use 2-space indentation for Python code and YAML in this project.
   - Prefer small, focused modules and functions over monolithic files.
   - When extending existing classes or functions, follow their conventions
     and docstrings.

4. **LLM and MCP Interactions**
   - This project uses `litellm` as the abstraction for all LLM calls
     (OpenAI, OpenRouter, Gemini, etc.).
   - In tests, **never** perform real network calls:
     - Use mocking/fakes for `litellm.completion` and any MCP client calls.
   - MCP integration is currently a stub; when tasks ask you to modify MCP
     behavior, keep the implementation testable and avoid hard-coding
     environment-specific details.

5. **CLI and UX Behavior**
   - The main entrypoint is `srw` (defined in `pyproject.toml`).
   - CLI parsing logic belongs in `simple_rag_writer.cli.*` modules.
   - Business logic (config, registry, prompts, runner, etc.) must remain
     independent from CLI code and directly testable.
   - When a task touches CLI behavior, also update help text and, where
     reasonable, add tests that instantiate parsers or call functions directly.

6. **Logging and Reproducibility**
   - Planning logs must be sufficiently detailed to allow future replay,
     including MCP references where applicable.
   - When you modify logging formats, keep them **append-only compatible**
     when possible; avoid breaking existing logs unless a task explicitly
     requires it.

---

## Workflow Per Task

For **each** `codex/TASKS/*.yaml` task:

1. **Read the Task Spec**
   - Open the corresponding YAML file (e.g. `codex/TASKS/03_implement_model_registry_and_litellm_integration.yaml`).
   - Understand:
     - `description`
     - `dependencies`
     - `goals`
     - `artifacts`
     - `steps`
     - `testing`

2. **Check Dependencies**
   - Ensure all listed `dependencies` are completed and integrated.
   - If a dependency is not satisfied (files missing or behavior clearly
     absent), report it via comments or docstrings, but do not alter other
     tasks’ YAML files.

3. **TDD Cycle**
   - Update or add tests under `tests/` according to the task’s `testing`
     section.
   - Run `pytest` (or a targeted subset, e.g. `pytest tests/test_*.py`) and
     confirm the new tests fail for the expected reason.
   - Implement or modify the corresponding production code under
     `src/simple_rag_writer/` to satisfy the tests.
   - Re-run tests until all relevant tests pass.

4. **Verification**
   - Besides tests, use small sanity checks appropriate to the task:
     - For CLI tasks: instantiate the parser and inspect arguments.
     - For config tasks: load a sample YAML config and verify attributes.
     - For runner tasks: use temporary directories to verify outputs are written.
   - Avoid heavy or external integration testing unless a task explicitly
     requests it.

5. **Git Discipline**
   - Stage only the changes directly relevant to the current task
     (code + tests + docs that the task requested).
   - Create **one** commit per task with a descriptive message, e.g.:
     - `feat: implement model registry with litellm`
     - `test: add config loader round-trip tests`
     - `feat: add automated task runner orchestration`
   - Do **not** push to any remote from inside the Codex environment.

6. **Task Completion Report**
   - At the end of the task, output a summary including:
     - The final commit message used.
     - A list of modified files (relative paths).
     - A short description of key behavior changes.
   - If you encountered unresolved questions or limitations, mention them
     briefly in the summary and, if helpful, in code comments.

---

## What You MUST NOT Do

- Do **not** introduce untested features or dead code.
- Do **not** modify `codex/TASKS/*.yaml` or `CODEX_TODO.md` unless a task
  explicitly directs you to.
- Do **not** perform real network calls to LLM or MCP services in tests.
- Do **not** push to any remote Git repository.
- Do **not** change project tooling (pytest, dependencies, etc.) without a
  specific task that authorizes it.

---

## Helpful Conventions

- Prefer pure functions and dependency injection to ease testing.
- When mocking external behavior (LLMs, MCP), keep fakes small and reusable
  across tests.
- When adding new modules, follow the existing package layout:
  - `config/` for configuration models and loaders.
  - `llm/` for model-related logic and litellm integration.
  - `mcp/` for MCP-related types, clients, and normalization.
  - `tasks/` for TaskSpec and task loading.
  - `prompts/` for prompt construction.
  - `runner/` for orchestration of automated tasks.
  - `planning/` for interactive planning mode.
  - `logging/` for log writers and log formats.
  - `replay/` for replay and reconstruction utilities.

If a new file does not clearly fit one of these, choose the closest match
and keep the module focused and cohesive.
