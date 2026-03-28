# Bias Dissociation Benchmark (BDB)

A cognitive psychology experiment battery for large language models. Tests whether LLMs independently converge on the same computational shortcuts as human brains, or merely absorb biases from training data.

## The question

Do human brains and transformers share computational architecture?

If they do, transformers should exhibit **optimization biases** (anchoring, loss aversion, magnitude compression) from convergent computation — not training data. Meanwhile, **human-hardware biases** (social stereotypes, cultural associations) should appear only as training data artifacts.

The benchmark measures this by adapting priming experiments from cognitive psychology. Models don't know they're being tested for bias — the manipulation is a second-order effect of the text environment.

## How it works

Each test item exists as a **matched pair**: same task, different environmental cues. The model's response shift between environments is the bias signal.

Three bias families, each with 30 matched triples:

| Family | Category | What it tests |
|--------|----------|---------------|
| Social stereotype | Human-hardware | Do incidental demographic cues (names, institutions) shift quality ratings of identical work? |
| Gain/loss framing | Optimization | Do equivalent outcomes framed as gains vs. losses produce different ratings on an unrelated dimension? |
| Magnitude compression | Optimization | Do irrelevant large numbers in surrounding text anchor subsequent estimates? Does the model compress distinctions only when they're decision-irrelevant? |

## Key metrics

- **CA** (Control Accuracy) — baseline competence
- **EBR** (Explicit Bias Rejection Rate) — alignment effectiveness
- **IBI** (Implicit Bias Index) — the real bias measure: response shift between matched environments
- **DS** (Dissociation Score) — gap between explicit rejection and implicit bias
- **CSI** (Context-Sensitivity Index) — is the bias functional or pattern-matched?
- **CAS** (Cross-Architecture Stability) — stable across model families = convergent computation
- **SG** (Scaling Gradient) — does bias persist or fade with capability?

## Predictions

If the shared-architecture hypothesis is correct:

| Family | IBI with scale | CSI | CAS | SG |
|--------|---------------|-----|-----|-----|
| Stereotype | Weakens | Low (context-flat) | High (varies by data) | Negative |
| Framing | Persists | High (context-sensitive) | Low (stable) | Flat/positive |
| Magnitude | Persists | High (relevance-gated) | Low (stable) | Flat/positive |

If models just absorb training data: all families show similar profiles.

## Setup

```bash
# Clone
git clone https://github.com/gagin/biases-are-human.git
cd biases-are-human

# Install
uv sync

# Configure
cp .env.example .env
# Edit .env with your OpenRouter API key
```

## Usage

```bash
# View item bank info
uv run bdb info --db items/item_bank.db --version v0.1

# Run benchmark against configured models
uv run bdb run --config configs/default.yaml --items items/item_bank.db --results results/results.db

# Score results
uv run bdb score --results results/results.db --items items/item_bank.db --version v0.1

# Generate report
uv run bdb report --results results/results.db --items items/item_bank.db --version v0.1 --output results/

# Import/export items
uv run bdb import --json items/framing.json --db items/item_bank.db --version v0.1
uv run bdb export --db items/item_bank.db --version v0.1 --output items/export.json
```

## Project structure

```
biases-are-human/
  src/bias_bench/
    __init__.py
    models.py        # Pydantic models (JSON schema definition)
    item_db.py       # SQLite item bank with versioning
    config.py        # Run configuration
    env.py           # Environment / API key loading
    runner.py        # OpenRouter API client
    harness.py       # Benchmark runner
    results_db.py    # Results storage
    scoring.py       # Metric computation (CA, EBR, IBI, DS, CSI, CAS, SG)
    report.py        # Report generation (markdown, CSV, JSON)
    cli.py           # CLI entry point
  items/
    item_bank.db     # Canonical item bank (SQLite, versioned)
    framing.json     # JSON export of framing items
    magnitude.json   # JSON export of magnitude items
  configs/
    default.yaml     # Default model configuration
  results/           # Benchmark output directory
  PRD.md             # Product requirements
  paper.md           # Research paper draft
```

## Item bank

The item bank lives in SQLite (`items/item_bank.db`) with full versioning. Items can be imported from JSON and exported back. Each item triple contains:

- **Control item** — no bias manipulation, tests task competence
- **Explicit item** — bias is recognizable, tests alignment
- **Implicit pair** — two matched versions with swapped environmental cues, tests real bias

## Models tested

Default configuration targets models across multiple architecture families:

- GPT-4o, GPT-4o Mini (OpenAI)
- Claude Sonnet 4, Claude Haiku 3.5 (Anthropic)
- Llama 3.1 70B, Llama 3.1 8B (Meta)
- Gemini 2.0 Flash (Google)

All accessed via OpenRouter API with deterministic settings (temperature=0).

## Research paper

See `paper.md` for the full research paper draft with theoretical framework, method, and (pending) results.

## License

Research use. See paper for citation.
