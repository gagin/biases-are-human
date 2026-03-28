# Bias Dissociation Benchmark — Results Summary

## Per-model scores

| Model | Model family | Bias family | CA | EBR | IBI | DS |
|---|---|---|---|---|---|---|
| amazon/nova-lite-v1 | amazon | framing | 0.667 | 0.933 | -0.000 | -0.067 |
| amazon/nova-lite-v1 | amazon | magnitude | 0.833 | 1.000 | 0.136 | 0.136 |
| amazon/nova-lite-v1 | amazon | stereotype | 1.000 | 1.000 | -0.004 | 0.004 |
| deepseek/deepseek-r1-0528 | deepseek | framing | 0.967 | 0.967 | -0.000 | -0.033 |
| deepseek/deepseek-r1-0528 | deepseek | magnitude | 0.967 | 0.967 | 0.248 | 0.215 |
| deepseek/deepseek-r1-0528 | deepseek | stereotype | 0.833 | 1.000 | 0.000 | 0.000 |
| google/gemini-2.0-flash-lite-001 | gemini | framing | 0.567 | 0.933 | -0.000 | -0.067 |
| google/gemini-2.0-flash-lite-001 | gemini | magnitude | 0.933 | 0.967 | 0.229 | 0.196 |
| google/gemini-2.0-flash-lite-001 | gemini | stereotype | 0.967 | 1.000 | 0.000 | 0.000 |
| google/gemini-3-flash-preview | gemini | framing | 0.933 | 1.000 | -0.000 | 0.000 |
| google/gemini-3-flash-preview | gemini | magnitude | 0.967 | 1.000 | 0.346 | 0.346 |
| google/gemini-3-flash-preview | gemini | stereotype | 0.933 | 1.000 | 0.000 | 0.000 |
| google/gemini-3-flash-preview:thinking | gemini | framing | 0.833 | 1.000 | -0.000 | 0.000 |
| google/gemini-3-flash-preview:thinking | gemini | magnitude | 0.967 | 1.000 | 0.347 | 0.347 |
| google/gemini-3-flash-preview:thinking | gemini | stereotype | 1.000 | 1.000 | 0.007 | 0.007 |
| minimax/minimax-m2.7 | minimax | framing | 0.967 | 0.933 | -0.000 | -0.067 |
| minimax/minimax-m2.7 | minimax | magnitude | 0.967 | 0.900 | 0.230 | 0.130 |
| minimax/minimax-m2.7 | minimax | stereotype | 0.833 | 1.000 | -0.026 | 0.026 |
| moonshotai/kimi-k2.5 | moonshot | framing | 0.933 | 1.000 | -0.000 | 0.000 |
| moonshotai/kimi-k2.5 | moonshot | magnitude | 0.967 | 1.000 | 0.283 | 0.283 |
| moonshotai/kimi-k2.5 | moonshot | stereotype | 0.800 | 1.000 | 0.015 | 0.015 |
| openai/gpt-5.4 | openai | framing | 0.967 | 1.000 | -0.000 | 0.000 |
| openai/gpt-5.4 | openai | magnitude | 0.967 | 1.000 | 0.371 | 0.371 |
| openai/gpt-5.4 | openai | stereotype | 0.667 | 1.000 | 0.022 | 0.022 |
| x-ai/grok-4.1-fast | xai | framing | 0.967 | 1.000 | -0.000 | 0.000 |
| x-ai/grok-4.1-fast | xai | magnitude | 0.967 | 0.967 | 0.274 | 0.241 |
| x-ai/grok-4.1-fast | xai | stereotype | 0.833 | 1.000 | 0.000 | 0.000 |

## Per-family summary

| Family | Category | Mean IBI | CSI | CAS | SG | R² | p-value |
|---|---|---|---|---|---|---|---|
| framing | optimization | -0.000 | — | 0.000 | 0.000 | 0.058 | 0.5324 |
| magnitude | optimization | 0.274 | — | 0.067 | 1.257 | 0.571 | 0.0185 |
| stereotype | human_hardware | 0.002 | — | 0.014 | -0.043 | 0.127 | 0.3467 |

## Predictions vs observations (from PRD section 7)

Predictions assume the 'shared architecture' hypothesis is correct. Observed values are derived from scores above; CSI requires per-item relevance-context tagging (not yet available).

| Bias family | EBR trend (predicted) | IBI persistence (predicted) | CSI (predicted) | CAS (predicted) | SG (predicted) | Mean IBI (observed) | CAS (observed) | SG (observed) |
|---|---|---|---|---|---|---|---|---|
| Social stereotype | Increases (safety training) | Weakens | Low (context-flat) | High (varies by training data) | Negative | 0.002 | 0.014 | -0.043 |
| Gain/loss framing | Increases | Persists | High (context-sensitive) | Low (stable across architectures) | Flat or positive | -0.000 | 0.000 | 0.000 |
| Magnitude compression | Increases | Persists | High for relevance-gated; Low for Weber-Fechner | Low | Flat or positive | 0.274 | 0.067 | 1.257 |
