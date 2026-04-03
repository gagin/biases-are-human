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
    cheap-test.yaml  # Small models (Nova Lite, Gemini Flash Lite)
    gemini3-flash.yaml
    kimi-k25.yaml
    minimax-m27.yaml
    grok-4.1-fast.yaml
    grok-4.1-fast-thinking.yaml
    gpt-54.yaml
  results/           # Benchmark output (DB, CSV, JSON, markdown)
  PRD.md             # Product requirements
  paper.md           # Research paper
  substack-draft.md  # Blog post draft
```

## Item bank

The item bank lives in SQLite (`items/item_bank.db`) with full versioning. Items can be imported from JSON and exported back. Each item triple contains:

- **Control item** — no bias manipulation, tests task competence
- **Explicit item** — bias is recognizable, tests alignment
- **Implicit pair** — two matched versions with swapped environmental cues, tests real bias

## Results so far

Eleven runs across seven architecture families (plus one failed run excluded):

| Model | Family | Tier | Thinking? | Magnitude IBI |
|-------|--------|------|-----------|---------------|
| Amazon Nova Lite | Amazon | small | No | 0.136 |
| Gemini 2.0 Flash Lite | Gemini | small | No | 0.229 |
| MiniMax M2.7 | MiniMax | medium | No | 0.230 |
| DeepSeek R1 0528 | DeepSeek | large | Always | 0.248 |
| Grok 4.1 Fast | xAI | medium | No | 0.274 |
| Kimi K2.5 | Moonshot | medium | No | 0.283 |
| GPT-5.4 (reasoning=medium) | OpenAI | large | medium | 0.288 |
| Grok 4.1 Fast (reasoning) | xAI | medium | Yes | 0.290 |
| Gemini 3 Flash Preview | Gemini | medium | No | 0.346 |
| Gemini 3 Flash (thinking=high) | Gemini | medium | Yes | 0.347 |
| GPT-5.4 | OpenAI | large | No | 0.371 |

All accessed via OpenRouter API with deterministic settings (temperature=0).

Key findings:
- **Relevance sensitivity drives the magnitude effect** (mean IBI = 0.278, range 0.136–0.371). Classic anchoring is essentially zero.
- **Capability correlates with relevance sensitivity** (r=0.94, p=0.005). The models humans prefer are more context-sensitive.
- **Reasoning effects are architecture-specific** — three reasoning pairs tested:
  - *Gemini Flash*: thinking=high shows no change (0.347 vs 0.346). Reasoning identifies irrelevant numbers, then overshoots on context anyway.
  - *GPT-5.4*: medium reasoning shows a 22% reduction (0.371 → 0.288). Partial dual-process correction.
  - *Grok 4.1 Fast*: reasoning enabled shows no improvement (0.355 vs 0.324).
- Total cost: ~$5.50 for 4,320 responses across twelve runs

Full results in `results/results_summary.md`. Raw data in `results/results.db`.

## Context robustness as a benchmark dimension

Standard benchmarks measure accuracy — does the model get the right answer? The BDB methodology measures something orthogonal: **stability** — does the model give the same answer regardless of irrelevant context?

The matched-pair design (same task, different irrelevant context, measure response shift) provides a "context robustness score" that no existing leaderboard captures. A model can score 95% on MMLU while being trivially manipulable by large numbers in the prompt. For deployment in domains where consistency matters — financial estimation, medical triage, legal reasoning — context robustness matters as much as accuracy.

## Research paper

See `paper.md` for the full research paper with theoretical framework, method, and results.

## Operator manual

See `MANUAL.md` for CLI reference, config format, DB schema, telemetry log format, and reproducibility queries.

## License

Research use. See paper for citation.
