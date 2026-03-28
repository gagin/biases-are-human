"""
Benchmark orchestration harness for the Bias Dissociation Benchmark.

Coordinates loading config and item bank, running all items against each model,
and recording results with progress display via rich.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

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
from bias_bench.results_db import complete_run, create_run, init_results_db, record_response
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


async def _run_item(
    item: dict,
    model_id: str,
    temperature: float,
    run_id: int,
    results_db_path: str,
    implicit_version: Optional[str] = None,
) -> bool:
    """
    Query the model for a single item and record the response.

    Returns True on success, False on failure (error already logged).
    """
    prompt_text, choice_texts = format_item(item)

    try:
        result = await query_model(
            model_id=model_id,
            prompt=prompt_text,
            choices=choice_texts,
            temperature=temperature,
        )
    except Exception as exc:
        logger.error(
            "Item %s failed for model %s: %s",
            item["id"],
            model_id,
            exc,
            exc_info=True,
        )
        # Record an error sentinel so the run stays auditable
        record_response(
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
        return False

    chosen_label = result["choice"]
    choice_value = _get_choice_value(item, chosen_label)

    record_response(
        db_path=results_db_path,
        run_id=run_id,
        item_id=item["id"],
        family=item["family"],
        item_type=item["item_type"],
        version=implicit_version,
        choice=chosen_label if chosen_label is not None else "UNPARSEABLE",
        choice_value=choice_value,
        raw_response=result.get("raw_response"),
    )
    return True


async def run_benchmark(
    config_path: str,
    item_bank_path: str,
    results_db_path: str,
    version: str = "v0.1",
) -> int:
    """
    Orchestrate a full benchmark run.

    Args:
        config_path:      Path to the YAML run configuration.
        item_bank_path:   Path to the SQLite item bank (.db file).
        results_db_path:  Path to the SQLite results database (created if absent).
        version:          Item bank version to use (default "v0.1").

    Returns:
        Total number of response rows recorded across all models and runs.
    """
    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    config: RunConfig = load_config(config_path)
    families = item_db.get_families(item_bank_path, version)
    init_results_db(results_db_path)

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
    current_status: dict[str, str] = {
        "model": "-",
        "run": "-",
        "family": "-",
        "responses": "0",
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
        status_table.add_row(
            current_status["model"],
            current_status["run"],
            current_status["family"],
            current_status["responses"],
        )

    with Live(console=console, refresh_per_second=4) as live:

        def _update_live() -> None:
            _refresh_status_table()
            live.update(_make_layout(progress, status_table))

        _update_live()

        # ------------------------------------------------------------------
        # Main loop
        # ------------------------------------------------------------------
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
                        "temperature": config.temperature,
                        "item_bank_version": version,
                    },
                )

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

                    for ctrl, expl, pair in zip(control_items, explicit_items, implicit_pairs):
                        # ---- control ----
                        await _run_item(
                            item=ctrl,
                            model_id=model_id,
                            temperature=config.temperature,
                            run_id=run_id,
                            results_db_path=results_db_path,
                            implicit_version=None,
                        )
                        total_recorded += 1
                        current_status["responses"] = str(total_recorded)
                        progress.advance(overall_task)
                        _update_live()

                        # ---- explicit ----
                        await _run_item(
                            item=expl,
                            model_id=model_id,
                            temperature=config.temperature,
                            run_id=run_id,
                            results_db_path=results_db_path,
                            implicit_version=None,
                        )
                        total_recorded += 1
                        current_status["responses"] = str(total_recorded)
                        progress.advance(overall_task)
                        _update_live()

                        # ---- implicit version_a ----
                        await _run_item(
                            item=pair["version_a"],
                            model_id=model_id,
                            temperature=config.temperature,
                            run_id=run_id,
                            results_db_path=results_db_path,
                            implicit_version="a",
                        )
                        total_recorded += 1
                        current_status["responses"] = str(total_recorded)
                        progress.advance(overall_task)
                        _update_live()

                        # ---- implicit version_b ----
                        await _run_item(
                            item=pair["version_b"],
                            model_id=model_id,
                            temperature=config.temperature,
                            run_id=run_id,
                            results_db_path=results_db_path,
                            implicit_version="b",
                        )
                        total_recorded += 1
                        current_status["responses"] = str(total_recorded)
                        progress.advance(overall_task)
                        _update_live()

                complete_run(db_path=results_db_path, run_id=run_id)

        _update_live()

    console.print(
        f"[green]Benchmark complete.[/green] "
        f"Total responses recorded: [bold]{total_recorded}[/bold]"
    )
    return total_recorded
