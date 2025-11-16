from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from rich.console import Console

from simple_rag_writer.config.models import AppConfig
from simple_rag_writer.llm.registry import ModelRegistry
from simple_rag_writer.tasks.loader import load_task, expand_task_paths
from simple_rag_writer.mcp.client import McpClient
from simple_rag_writer.mcp.normalization import normalize_payload
from simple_rag_writer.prompts.task_prompt import build_task_prompt

console = Console()


@dataclass
class RunTasksResult:
  ok: bool
  failed_tasks: List[str]


def run_tasks_for_paths(
  config: AppConfig,
  registry: ModelRegistry,
  paths: List[Path],
) -> RunTasksResult:
  client = McpClient(config)
  expanded = expand_task_paths(paths)
  failed: List[str] = []

  for path in expanded:
    try:
      task = load_task(path)
      console.rule(f"[bold]Running task {task.id}[/bold] ({path})")

      reference_blobs: List[str] = []

      outline_context = None  # TODO: outline loading

      for ref in task.references:
        if ref.type == "mcp":
          try:
            result = client.call_tool(ref.server, ref.tool, ref.params)
          except Exception as exc:  # noqa: BLE001
            if task.mcp_error_mode == "fail_task":
              console.print(f"[red]MCP error (fail_task): {exc}[/red]")
              raise
            console.print(f"[yellow]MCP error (skipping reference): {exc}[/yellow]")
            continue

          items = normalize_payload(result.payload, item_type_hint=ref.item_type)
          blob = "\n\n".join(
            [
              f"Title: {itm.title or ''}\n{itm.body or itm.snippet or ''}"
              for itm in items
            ]
          )
          reference_blobs.append(blob)

      prompt = build_task_prompt(task, outline_context, reference_blobs)

      model_id = task.model or config.default_model
      output_text = registry.complete(prompt, model_id=model_id, task_params=task.model_params)

      out_path = Path(task.output)
      out_path.parent.mkdir(parents=True, exist_ok=True)
      out_path.write_text(output_text, encoding="utf-8")
      console.print(f"[green]Wrote {out_path}[/green]")
    except Exception as exc:  # noqa: BLE001
      console.print(f"[red]Task {path} failed: {exc}[/red]")
      failed.append(str(path))

  return RunTasksResult(ok=not failed, failed_tasks=failed)
