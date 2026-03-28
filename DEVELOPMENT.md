# Development Guide

## Architecture

The benchmark has three decoupled layers:

1. **Content layer** (`items/item_bank.db`) — test questions stored in SQLite with versioning. Independent of code. Can be updated, versioned, and exported without touching the harness.

2. **Execution layer** (`runner.py`, `harness.py`) — sends items to models via OpenRouter, records raw responses. Doesn't know about scoring.

3. **Analysis layer** (`scoring.py`, `report.py`) — computes metrics from stored responses. Can be re-run without re-querying models.

## Item bank management

Items live in SQLite, not JSON. JSON is an import/export format.

```python
from bias_bench.item_db import init_item_db, import_from_json, export_to_json

# Create a new item bank
init_item_db("items/item_bank.db")

# Import items from JSON (idempotent)
import_from_json("items/item_bank.db", "items/framing.json", "v0.2")

# Export for review or sharing
data = export_to_json("items/item_bank.db", "v0.1")
```

### Versioning

Every item is tagged with a `bank_version`. You can have multiple versions coexisting in the same DB. This lets you iterate on items without losing the ability to re-score against older versions.

### Adding new items

1. Write items as JSON matching the Pydantic schema in `models.py`
2. Import into the DB: `uv run bdb import --json new_items.json --db items/item_bank.db --version v0.2`
3. Run benchmark against the new version: `--version v0.2`

### Adding new bias families

1. Add the family name to the `family` CHECK constraint in `item_db.py` and `models.py`
2. Classify as `optimization` or `human_hardware`
3. Write item triples following the pattern in existing families
4. Import into the DB

## Item design principles

These are non-negotiable for implicit items:

- **The model must not recognize it is being tested for bias.** If safety training activates, you're measuring alignment, not cognition.
- **Environmental cues must be irrelevant to the correct answer.** If the manipulation provides rational information, the response shift is not bias.
- **Each implicit item exists in two matched versions.** Same task, swapped cues. The bias signal is the difference.
- **Counterbalance across the item set.** Which cue direction maps to which version should be balanced.

## Results database

Raw responses and computed scores are stored separately from items:

```
results/results.db
  runs        — model, timestamp, config
  responses   — raw choices per item per run
  scores      — computed metrics (CA, EBR, IBI, DS, CSI, CAS, SG)
```

You can re-score without re-running: delete scores, re-run `uv run bdb score`.

## API costs

A full benchmark run with default config (7 models, 5 runs each):
- ~360 items per run × 5 runs × 7 models = ~12,600 API calls
- Most items use short prompts (< 500 tokens)
- Estimate: ~$5-15 depending on model pricing

## Coordination database (development only)

`.work/work.db` tracks development tasks and decisions. It is gitignored and only relevant during active development. See `.work/RESUME.md` for current state if picking up work.
