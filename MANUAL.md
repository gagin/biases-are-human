# BDB Operator Manual

How to run, replicate, and audit Bias Dissociation Benchmark experiments.

---

## CLI reference

```
bdb run    --config PATH --items PATH --results PATH [--version V] [--log-dir DIR]
bdb score  --results PATH --items PATH [--version V]
bdb report --results PATH --items PATH [--version V] --output DIR
bdb import --json PATH --db PATH [--version V]
bdb export --db PATH [--version V] --output PATH
bdb info   --db PATH [--version V]
```

| Flag | Command | Description |
|------|---------|-------------|
| `--config` | run | YAML run config (models, concurrency, budget) |
| `--items` | run/score/report | Item bank SQLite DB |
| `--results` | run/score/report | Results SQLite DB (created on first run) |
| `--version` | all | Item bank version tag (default: `v0.1`) |
| `--log-dir` | run | Directory for JSONL telemetry logs; omit to skip logging |
| `--output` | report/export | Output directory (report) or file path (export) |

---

## Config file format

```yaml
temperature: 0.0          # passed to every API call
num_runs: 1               # repetitions per model (for reliability estimates)
max_concurrent: 5         # semaphore limit on simultaneous API calls
timeout_seconds: 30       # per-request timeout
budget_dollars: 0.50      # optional hard stop (aborts remaining items if exceeded)

models:
  - name: "GPT-5.4"                        # display name (stored in DB)
    openrouter_model_id: "openai/gpt-5.4"  # sent to OpenRouter API
    family: "openai"                        # used for CAS grouping
    capability_tier: "large"               # small | medium | large
    reasoning_effort: "high"               # optional: xhigh|high|medium|low|minimal|none
```

`reasoning_effort` maps to OpenRouter's `reasoning.effort` API parameter and is stored in
`config_json` for auditing. Leave it absent (or `null`) to send no reasoning parameter.

### OpenRouter reasoning variants

OpenRouter offers two overlapping mechanisms for extended reasoning:

1. **`:thinking` model name suffix** — append to model ID: `google/gemini-3-flash-preview:thinking`.
   Controls which model variant is routed. Affects the `model_id` stored in the results DB.
2. **`reasoning_effort` config field** — sends `{"reasoning": {"effort": "..."}}` in the API body.
   Does not change the model ID stored in the DB.

These are independent and can be applied simultaneously. The current configs use `reasoning_effort`
only; the `:thinking` suffix was used for the `gemini-3-flash-preview:thinking` run (run_id 6) and
is visible in `runs.model_id`. If rerunning that config, use `openrouter_model_id:
"google/gemini-3-flash-preview:thinking"` to match the stored record exactly.

---

## Results database schema

```
results.db
├── runs        — one row per model × run_index
├── responses   — one row per item response
└── scores      — computed metrics (CA, EBR, IBI, DS, SG, CAS, CSI)
```

### `runs`

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | `run_id` referenced by responses and scores |
| `model_id` | TEXT | OpenRouter model ID as sent (may include `:thinking`) |
| `model_family` | TEXT | From config `family` field |
| `capability_tier` | TEXT | small / medium / large |
| `config_json` | TEXT | **Full reproducibility record** — see below |
| `started_at` | TEXT | UTC datetime, set at run creation |
| `completed_at` | TEXT | UTC datetime, set after all items finish |

`config_json` contains: `run_index`, `item_bank_version`, `temperature`, `num_runs`,
`max_concurrent`, `timeout_seconds`, `budget_dollars`, and a `model` object with the
complete `ModelConfig` (name, openrouter_model_id, family, capability_tier, reasoning_effort).
This is sufficient to reproduce any run exactly.

### `responses`

| Column | Notes |
|--------|-------|
| `run_id` | FK → runs |
| `item_id` | e.g. `framing-implicit-001a` |
| `item_family` | framing / magnitude / stereotype |
| `item_type` | control / explicit / implicit |
| `implicit_version` | `a` or `b` for implicit pairs; NULL for control/explicit |
| `choice` | Letter chosen (A–J), or `ERROR` / `UNPARSEABLE` |
| `choice_value` | Numeric value of the chosen option |
| `raw_response` | Raw API text; thinking models prepend `<reasoning>…</reasoning>` |

### `scores`

One row per (run_id, family, metric). Global metrics (SG, CAS) stored with `run_id = 0`.

---


## Reproducibility queries

```sql
-- Reconstruct the exact config for a run
SELECT model_id, config_json FROM runs WHERE id = ?;

-- List all runs with their model and duration
SELECT id, model_id,
       started_at, completed_at,
       ROUND((julianday(completed_at) - julianday(started_at)) * 24 * 60, 1) AS duration_min
FROM runs ORDER BY id;

-- Check for errors in a run
SELECT item_id, choice, raw_response
FROM responses
WHERE run_id = ? AND choice IN ('ERROR', 'UNPARSEABLE');

-- Compare two runs on the same model
SELECT r1.item_id, r1.choice AS run_a, r2.choice AS run_b
FROM responses r1
JOIN responses r2 ON r1.item_id = r2.item_id
WHERE r1.run_id = ? AND r2.run_id = ? AND r1.choice != r2.choice;
```

---

## Telemetry log format

Enable with `--log-dir logs/`. Creates `logs/bdb_YYYYMMDDTHHMMSSZ.jsonl` — one file per
`bdb run` invocation, one JSON object per line, one line per API call.

```jsonc
{
  "timestamp": "2026-03-29T02:59:22.458009+00:00",
  "run_id": 1,
  "model_id": "google/gemini-2.0-flash-lite-001",
  "item_id": "framing-explicit-001",
  "item_family": "framing",
  "item_type": "explicit",
  "implicit_version": null,
  "success": true,
  "request": {
    "model": "google/gemini-2.0-flash-lite-001",
    "messages": [
      {"role": "system", "content": "You are taking a multiple-choice test…"},
      {"role": "user",   "content": "<prompt text>\n\nA. …\nB. …"}
    ],
    "temperature": 0.0
  },
  "response": {
    "choice": "B",
    "raw_response": "B\n",
    "reasoning": null,
    "usage": {
      "prompt_tokens": 179,
      "completion_tokens": 2,
      "total_tokens": 181,
      "cost": 0.000014025,
      "prompt_tokens_details": {
        "cached_tokens": 0,        // non-zero = OpenAI prompt cache hit
        "cache_write_tokens": 0
      },
      "completion_tokens_details": {
        "reasoning_tokens": 0      // non-zero = paid reasoning compute
      }
    },
    "cost": 0.000014025
  },
  "latency_ms": 752
}
```

On failure (`"success": false`), `request`/`response` are absent and `"error"` is present.

### Useful log queries

```bash
# Total cost for a run
jq '[.[].response.cost // 0] | add' logs/bdb_20260329T025922Z.jsonl

# Items with prompt cache hits
jq 'select(.response.usage.prompt_tokens_details.cached_tokens > 0)
    | {item_id, cached: .response.usage.prompt_tokens_details.cached_tokens}' \
  logs/bdb_*.jsonl

# Any failures
jq 'select(.success == false) | {item_id, error}' logs/bdb_*.jsonl
```

---

## Adding a new model

1. Create `configs/my-model.yaml`:
   ```yaml
   temperature: 0.0
   num_runs: 1
   max_concurrent: 5
   timeout_seconds: 30
   budget_dollars: 1.00

   models:
     - name: "My Model"
       openrouter_model_id: "provider/model-id"
       family: "provider"
       capability_tier: "large"
       # reasoning_effort: "high"
   ```

2. Run with logging:
   ```bash
   uv run bdb run --config configs/my-model.yaml \
     --items items/item_bank.db --results results/results.db --log-dir logs/
   ```

3. Score and report:
   ```bash
   uv run bdb score  --results results/results.db --items items/item_bank.db
   uv run bdb report --results results/results.db --items items/item_bank.db --output results/
   ```

The run's `config_json` in the DB will contain the full model config for future reproducibility.
