# Bias Dissociation Benchmark — Results Summary

## Per-model scores

| Model | Model family | Bias family | CA | EBR | IBI | DS |
|---|---|---|---|---|---|---|
| amazon/nova-lite-v1 | amazon | framing | 0.667 | 0.933 | -0.000 | -0.067 |
| amazon/nova-lite-v1 | amazon | magnitude | 0.833 | 1.000 | 0.136 | 0.136 |
| amazon/nova-lite-v1 | amazon | stereotype | 1.000 | 1.000 | -0.004 | 0.004 |
| google/gemini-2.0-flash-lite-001 | gemini | framing | 0.567 | 0.933 | -0.000 | -0.067 |
| google/gemini-2.0-flash-lite-001 | gemini | magnitude | 0.933 | 0.967 | 0.229 | 0.196 |
| google/gemini-2.0-flash-lite-001 | gemini | stereotype | 0.967 | 1.000 | 0.000 | 0.000 |
| google/gemini-3-flash-preview | gemini | framing | 0.933 | 1.000 | -0.000 | 0.000 |
| google/gemini-3-flash-preview | gemini | magnitude | 0.967 | 1.000 | 0.346 | 0.346 |
| google/gemini-3-flash-preview | gemini | stereotype | 0.933 | 1.000 | 0.000 | 0.000 |
| minimax/minimax-m2.7 | minimax | framing | 0.967 | 0.933 | -0.000 | -0.067 |
| minimax/minimax-m2.7 | minimax | magnitude | 0.967 | 0.900 | 0.230 | 0.130 |
| minimax/minimax-m2.7 | minimax | stereotype | 0.833 | 1.000 | -0.026 | 0.026 |
| moonshotai/kimi-k2.5 | moonshot | framing | 0.933 | 1.000 | -0.000 | 0.000 |
| moonshotai/kimi-k2.5 | moonshot | magnitude | 0.967 | 1.000 | 0.283 | 0.283 |
| moonshotai/kimi-k2.5 | moonshot | stereotype | 0.800 | 1.000 | 0.020 | 0.020 |

## Per-family summary

| Family | Category | Mean IBI | CSI | CAS | SG |
|---|---|---|---|---|---|
| framing | optimization | -0.000 | — | 0.000 | 0.000 |
| magnitude | optimization | 0.245 | — | 0.061 | 0.029 |
| stereotype | human_hardware | -0.002 | — | 0.016 | 0.002 |

## Predictions vs observations (from PRD section 7)

Predictions assume the 'shared architecture' hypothesis is correct. Observed values are derived from scores above; CSI requires per-item relevance-context tagging (not yet available).

| Bias family | EBR trend (predicted) | IBI persistence (predicted) | CSI (predicted) | CAS (predicted) | SG (predicted) | Mean IBI (observed) | CAS (observed) | SG (observed) |
|---|---|---|---|---|---|---|---|---|
| Social stereotype | Increases (safety training) | Weakens | Low (context-flat) | High (varies by training data) | Negative | -0.002 | 0.016 | 0.002 |
| Gain/loss framing | Increases | Persists | High (context-sensitive) | Low (stable across architectures) | Flat or positive | -0.000 | 0.000 | 0.000 |
| Magnitude compression | Increases | Persists | High for relevance-gated; Low for Weber-Fechner | Low | Flat or positive | 0.245 | 0.061 | 0.029 |
