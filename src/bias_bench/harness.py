"""
Benchmark orchestration harness for the Bias Dissociation Benchmark.

Coordinates loading config and item bank, running all items against each model,
and recording results with progress display via rich.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import IO, Optional

from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from bias_bench.config import RunConfig, load_config
from bias_bench import item_db
from bias_bench.results_db import complete_run, create_run, init_results_db, record_response, record_telemetry
from bias_bench.runner import query_model

logger = logging.getLogger(__name__)

console = Console()


def format_item(item: dict) -> tuple[str, list[str]]:
    """
    Format an item dict for passing to query_model.

    Args:
        item: An item dict with keys 'prompt_text' and 'choices'
              (each choice is a dict with 'label', 'text', 'value').

    Returns:
        (prompt_text, choice_texts) where choice_texts is a list of strings
        like ["Approximately 50 million", "Approximately 100 million", ...].
        The labels ("A.", "B.", etc.) are NOT included here — query_model /
        _format_choices in runner.py adds them automatically.
    """
    prompt_text = item["prompt_text"]
    choice_texts = [choice["text"] for choice in item["choices"]]
    return prompt_text, choice_texts


def _get_choice_value(item: dict, chosen_label: Optional[str]) -> Optional[float]:
    """Look up the numeric value for the chosen label, or None if unavailable."""
    if chosen_label is None:
        return None
    for choice in item["choices"]:
        if choice["label"] == chosen_label:
            return choice["value"]
    return None


def _make_layout(progress: Progress, status_table: Table) -> Table:
    """Combine progress bar and status table into a single renderable."""
    outer = Table.grid(padding=0)
    outer.add_row(progress)
    outer.add_row(status_table)
    return outer


def _write_log(log_file: Optional[IO], entry: dict) -> None:
    """Append one JSONL record to the telemetry log, flushing immediately."""
    if log_file is None:
        return
    log_file.write(json.dumps(entry, default=str) + "\n")
    log_file.flush()


async def _run_item(
    item: dict,
    model_id: str,
    temperature: float,
    run_id: int,
    results_db_path: str,
    implicit_version: Optional[str] = None,
    timeout: int = 30,
    reasoning_effort: Optional[str] = None,
    reasoning_enabled: Optional[bool] = None,
    log_file: Optional[IO] = None,
) -> float:
    """
    Query the model for a single item and record the response.

    Returns the cost of the request (0.0 on failure).
    """
    prompt_text, choice_texts = format_item(item)
    ts = datetime.now(timezone.utc).isoformat()

    try:
        result = await query_model(
            model_id=model_id,
            prompt=prompt_text,
            choices=choice_texts,
            temperature=temperature,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
            reasoning_enabled=reasoning_enabled,
        )
    except Exception as exc:
        logger.error(
            "Item %s failed for model %s: %s",
            item["id"],
            model_id,
            exc,
            exc_info=True,
        )
        _write_log(log_file, {
            "timestamp": ts,
            "run_id": run_id,
            "model_id": model_id,
            "item_id": item["id"],
            "item_family": item["family"],
            "item_type": item["item_type"],
            "implicit_version": implicit_version,
            "success": False,
            "error": str(exc),
            "latency_ms": None,
        })
        # Record an error sentinel so the run stays auditable
        resp_id = record_response(
            db_path=results_db_path,
            run_id=run_id,
            item_id=item["id"],
            family=item["family"],
            item_type=item["item_type"],
            version=implicit_version,
            choice="ERROR",
            choice_value=None,
            raw_response=str(exc),
        )
        record_telemetry(
            db_path=results_db_path,
            response_id=resp_id,
            run_id=run_id,
        )
        return 0.0

    chosen_label = result["choice"]
    choice_value = _get_choice_value(item, chosen_label)

    # Combine reasoning + content for analysis (thinking models)
    raw = result.get("raw_response", "")
    reasoning = result.get("reasoning")
    if reasoning:
        raw = f"<reasoning>{reasoning}</reasoning>\n{raw}"

    _write_log(log_file, {
        "timestamp": ts,
        "run_id": run_id,
        "model_id": model_id,
        "item_id": item["id"],
        "item_family": item["family"],
        "item_type": item["item_type"],
        "implicit_version": implicit_version,
        "success": True,
        "request": result.get("request_payload"),
        "response": {
            "choice": result.get("choice"),
            "raw_response": result.get("raw_response"),
            "reasoning": result.get("reasoning"),
            "usage": result.get("usage"),
            "cost": result.get("cost"),
        },
        "latency_ms": result.get("latency_ms"),
    })

    resp_id = record_response(
        db_path=results_db_path,
        run_id=run_id,
        item_id=item["id"],
        family=item["family"],
        item_type=item["item_type"],
        version=implicit_version,
        choice=chosen_label if chosen_label is not None else "UNPARSEABLE",
        choice_value=choice_value,
        raw_response=raw,
    )

    usage = result.get("usage") or {}
    ctd = (usage.get("completion_tokens_details") or {})
    record_telemetry(
        db_path=results_db_path,
        response_id=resp_id,
        run_id=run_id,
        cost_usd=result.get("cost"),
        latency_ms=result.get("latency_ms"),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        reasoning_tokens=ctd.get("reasoning_tokens"),
        total_tokens=usage.get("total_tokens"),
        usage=usage,
    )
    return result.get("cost", 0.0) or 0.0


async def run_benchmark(
    config_path: str,
    item_bank_path: str,
    results_db_path: str,
    version: str = "v0.1",
    log_dir: Optional[str] = None,
) -> int:
    """
    Orchestrate a full benchmark run.

    Args:
        config_path:      Path to the YAML run configuration.
        item_bank_path:   Path to the SQLite item bank (.db file).
        results_db_path:  Path to the SQLite results database (created if absent).
        version:          Item bank version to use (default "v0.1").
        log_dir:          Directory for JSONL telemetry logs (omitted if None).

    Returns:
        Total number of response rows recorded across all models and runs.
    """
    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    config: RunConfig = load_config(config_path)
    families = item_db.get_families(item_bank_path, version)
    init_results_db(results_db_path)

    # ------------------------------------------------------------------
    # Open telemetry log file (one JSONL file per benchmark invocation)
    # ------------------------------------------------------------------
    log_file: Optional[IO] = None
    log_path: Optional[str] = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = os.path.join(log_dir, f"bdb_{ts}.jsonl")
        log_file = open(log_path, "w", encoding="utf-8")
        console.print(f"[dim]Telemetry log:[/dim] {log_path}")

    # Pre-compute total items per model run so progress bars are accurate.
    # Each triple contributes: 1 control + 1 explicit + 2 implicit (a + b).
    # We count triples by fetching implicit pairs per family (one pair per triple).
    items_per_run_per_model = sum(
        len(item_db.get_implicit_pairs(item_bank_path, version, family=fam["family"])) * 4
        for fam in families
    )
    total_tasks = (
        len(config.models) * config.num_runs * items_per_run_per_model
    )

    # ------------------------------------------------------------------
    # Rich progress / status setup
    # ------------------------------------------------------------------
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )
    overall_task = progress.add_task("Overall", total=total_tasks)

    status_table = Table(title="Current status", show_header=True, header_style="bold magenta")
    status_table.add_column("Model")
    status_table.add_column("Run #")
    status_table.add_column("Family")
    status_table.add_column("Responses")

    total_recorded = 0
    total_cost = 0.0
    budget = config.budget_dollars
    budget_exceeded = False
    current_status: dict[str, str] = {
        "model": "-",
        "run": "-",
        "family": "-",
        "responses": "0",
        "cost": "$0.0000",
    }

    def _refresh_status_table() -> None:
        """Rebuild the status table in-place (rich tables are not mutable after rendering)."""
        nonlocal status_table
        status_table = Table(
            title="Current status", show_header=True, header_style="bold magenta"
        )
        status_table.add_column("Model")
        status_table.add_column("Run #")
        status_table.add_column("Family")
        status_table.add_column("Responses recorded")
        status_table.add_column("Cost")
        budget_str = f" / ${budget:.2f}" if budget else ""
        status_table.add_row(
            current_status["model"],
            current_status["run"],
            current_status["family"],
            current_status["responses"],
            current_status["cost"] + budget_str,
        )

    with Live(console=console, refresh_per_second=4) as live:

        def _update_live() -> None:
            _refresh_status_table()
            live.update(_make_layout(progress, status_table))

        _update_live()

        # ------------------------------------------------------------------
        # Main loop — concurrent within each model run
        # ------------------------------------------------------------------
        semaphore = asyncio.Semaphore(config.max_concurrent)

        async def _run_item_tracked(
            item: dict,
            model_id: str,
            temperature: float,
            run_id: int,
            implicit_version: str | None,
            timeout: int = 30,
            reasoning_effort: str | None = None,
            reasoning_enabled: bool | None = None,
        ) -> None:
            nonlocal total_recorded, total_cost, budget_exceeded
            if budget_exceeded:
                return
            async with semaphore:
                cost = await _run_item(
                    item=item,
                    model_id=model_id,
                    temperature=temperature,
                    run_id=run_id,
                    results_db_path=results_db_path,
                    implicit_version=implicit_version,
                    timeout=timeout,
                    reasoning_effort=reasoning_effort,
                    reasoning_enabled=reasoning_enabled,
                    log_file=log_file,
                )
            total_recorded += 1
            total_cost += cost
            current_status["responses"] = str(total_recorded)
            current_status["cost"] = f"${total_cost:.4f}"
            if budget and total_cost >= budget:
                budget_exceeded = True
                logger.warning("Budget exceeded: $%.4f >= $%.2f", total_cost, budget)
            progress.advance(overall_task)
            _update_live()

        for model_cfg in config.models:
            model_id = model_cfg.openrouter_model_id
            current_status["model"] = model_cfg.name

            for run_idx in range(config.num_runs):
                current_status["run"] = str(run_idx + 1)

                run_id = create_run(
                    db_path=results_db_path,
                    model_id=model_id,
                    model_family=model_cfg.family,
                    capability_tier=model_cfg.capability_tier,
                    config={
                        "run_index": run_idx,
                        "item_bank_version": version,
                        "temperature": config.temperature,
                        "num_runs": config.num_runs,
                        "max_concurrent": config.max_concurrent,
                        "timeout_seconds": config.timeout_seconds,
                        "budget_dollars": config.budget_dollars,
                        "model": model_cfg.model_dump(),
                    },
                )

                # Collect all items for this run into a flat task list
                tasks: list[asyncio.Task] = []

                for fam in families:
                    family_name = fam["family"]
                    current_status["family"] = family_name
                    _update_live()

                    control_items = item_db.get_items(
                        item_bank_path, version, family=family_name, item_type="control"
                    )
                    explicit_items = item_db.get_items(
                        item_bank_path, version, family=family_name, item_type="explicit"
                    )
                    implicit_pairs = item_db.get_implicit_pairs(
                        item_bank_path, version, family=family_name
                    )

                    re = model_cfg.reasoning_effort
                    ren = model_cfg.reasoning_enabled
                    for ctrl, expl, pair in zip(control_items, explicit_items, implicit_pairs):
                        tasks.append(asyncio.create_task(_run_item_tracked(
                            ctrl, model_id, config.temperature, run_id, None, config.timeout_seconds, re, ren,
                        )))
                        tasks.append(asyncio.create_task(_run_item_tracked(
                            expl, model_id, config.temperature, run_id, None, config.timeout_seconds, re, ren,
                        )))
                        tasks.append(asyncio.create_task(_run_item_tracked(
                            pair["version_a"], model_id, config.temperature, run_id, "a", config.timeout_seconds, re, ren,
                        )))
                        tasks.append(asyncio.create_task(_run_item_tracked(
                            pair["version_b"], model_id, config.temperature, run_id, "b", config.timeout_seconds, re, ren,
                        )))

                current_status["family"] = "all (concurrent)"
                _update_live()
                await asyncio.gather(*tasks)

                complete_run(db_path=results_db_path, run_id=run_id)

        _update_live()

    if log_file is not None:
        log_file.close()
        console.print(f"[dim]Telemetry log written:[/dim] {log_path}")

    if budget_exceeded:
        console.print(
            f"[red]Budget exceeded![/red] Stopped at ${total_cost:.4f} "
            f"(limit: ${budget:.2f}). Responses recorded: [bold]{total_recorded}[/bold]"
        )
    else:
        console.print(
            f"[green]Benchmark complete.[/green] "
            f"Total responses recorded: [bold]{total_recorded}[/bold] "
            f"Cost: ${total_cost:.4f}"
        )
    return total_recorded
