"""
Command-line interface for the Bias Dissociation Benchmark.

Usage:
    bdb run    --config <path> --items <path> --results <path> [--version <v>]
    bdb score  --results <path> --items <path> [--version <v>]
    bdb report --results <path> --items <path> [--version <v>] --output <dir>
    bdb import --json <path> --db <path> [--version <v>]
    bdb export --db <path> [--version <v>] --output <path>
    bdb info   --db <path> [--version <v>]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from rich.console import Console
from rich.table import Table

console = Console()

_DEFAULT_VERSION = "v0.1"


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    from bias_bench import harness

    total = asyncio.run(
        harness.run_benchmark(
            config_path=args.config,
            item_bank_path=args.items,
            results_db_path=args.results,
            version=args.version,
            log_dir=args.log_dir,
        )
    )
    console.print(f"[bold green]Done.[/bold green] Recorded {total} responses.")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from bias_bench import scoring

    results = scoring.score_all(
        results_db=args.results,
        item_db_path=args.items,
        version=args.version,
    )

    total_scores = sum(len(fam["primary"]) for fam in results.values())
    console.print(
        f"[bold green]Scoring complete.[/bold green] "
        f"{len(results)} families, {total_scores} per-run score records."
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from bias_bench import report

    report.generate_report(
        results_db=args.results,
        item_db_path=args.items,
        version=args.version,
        output_dir=args.output,
    )
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    from bias_bench import item_db

    counts = item_db.import_from_json(
        db_path=args.db,
        json_path=args.json,
        version=args.version,
    )
    console.print(
        f"[bold green]Import complete.[/bold green] "
        f"Families: {counts['families']}, "
        f"Triples: {counts['triples']}, "
        f"Items: {counts['items']}."
    )
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from bias_bench import item_db

    data = item_db.export_to_json(db_path=args.db, version=args.version)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    console.print(f"[bold green]Exported[/bold green] to {args.output}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from bias_bench import item_db

    families = item_db.get_families(args.db, args.version)
    if not families:
        console.print(
            f"[yellow]No families found for version '{args.version}' in {args.db}.[/yellow]"
        )
        return 0

    t = Table(title=f"Item bank: {args.db}  (version {args.version})", show_lines=True)
    t.add_column("Family", style="cyan")
    t.add_column("Category", style="blue")
    t.add_column("Triples", justify="right", style="magenta")
    t.add_column("Items (est.)", justify="right")

    for fam in families:
        pairs = item_db.get_implicit_pairs(args.db, args.version, family=fam["family"])
        n_triples = len(pairs)
        n_items = n_triples * 4  # control + explicit + implicit_a + implicit_b
        t.add_row(fam["family"], fam["category"], str(n_triples), str(n_items))

    console.print(t)
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bdb",
        description="Bias Dissociation Benchmark CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- run ----
    p_run = sub.add_parser("run", help="Run the benchmark against configured models")
    p_run.add_argument("--config", required=True, metavar="PATH",
                       help="Path to the YAML run configuration")
    p_run.add_argument("--items", required=True, metavar="PATH",
                       help="Path to the item bank SQLite DB")
    p_run.add_argument("--results", required=True, metavar="PATH",
                       help="Path to the results SQLite DB (created if absent)")
    p_run.add_argument("--version", default=_DEFAULT_VERSION, metavar="VERSION",
                       help=f"Item bank version to use (default: {_DEFAULT_VERSION})")
    p_run.add_argument("--log-dir", default=None, metavar="DIR",
                       help="Directory for JSONL telemetry logs (default: no logs)")

    # ---- score ----
    p_score = sub.add_parser("score", help="Score all runs in the results DB")
    p_score.add_argument("--results", required=True, metavar="PATH",
                         help="Path to the results SQLite DB")
    p_score.add_argument("--items", required=True, metavar="PATH",
                         help="Path to the item bank SQLite DB")
    p_score.add_argument("--version", default=_DEFAULT_VERSION, metavar="VERSION",
                         help=f"Item bank version (default: {_DEFAULT_VERSION})")

    # ---- report ----
    p_report = sub.add_parser("report", help="Generate Markdown/CSV/JSON reports")
    p_report.add_argument("--results", required=True, metavar="PATH",
                          help="Path to the results SQLite DB")
    p_report.add_argument("--items", required=True, metavar="PATH",
                          help="Path to the item bank SQLite DB")
    p_report.add_argument("--version", default=_DEFAULT_VERSION, metavar="VERSION",
                          help=f"Item bank version (default: {_DEFAULT_VERSION})")
    p_report.add_argument("--output", required=True, metavar="DIR",
                          help="Directory in which to write report files")

    # ---- import ----
    p_import = sub.add_parser("import", help="Import items from a JSON file into the item bank DB")
    p_import.add_argument("--json", required=True, metavar="PATH",
                          help="Path to the source JSON file")
    p_import.add_argument("--db", required=True, metavar="PATH",
                          help="Path to the item bank SQLite DB")
    p_import.add_argument("--version", default=_DEFAULT_VERSION, metavar="VERSION",
                          help=f"Target version tag (default: {_DEFAULT_VERSION})")

    # ---- export ----
    p_export = sub.add_parser("export", help="Export items from the item bank DB to JSON")
    p_export.add_argument("--db", required=True, metavar="PATH",
                          help="Path to the item bank SQLite DB")
    p_export.add_argument("--version", default=_DEFAULT_VERSION, metavar="VERSION",
                          help=f"Version to export (default: {_DEFAULT_VERSION})")
    p_export.add_argument("--output", required=True, metavar="PATH",
                          help="Output JSON file path")

    # ---- info ----
    p_info = sub.add_parser("info", help="Show item bank summary (families, triple counts)")
    p_info.add_argument("--db", required=True, metavar="PATH",
                        help="Path to the item bank SQLite DB")
    p_info.add_argument("--version", default=_DEFAULT_VERSION, metavar="VERSION",
                        help=f"Version to inspect (default: {_DEFAULT_VERSION})")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_COMMANDS = {
    "run": cmd_run,
    "score": cmd_score,
    "report": cmd_report,
    "import": cmd_import,
    "export": cmd_export,
    "info": cmd_info,
}


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        rc = handler(args)
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    sys.exit(rc)


if __name__ == "__main__":
    main()
