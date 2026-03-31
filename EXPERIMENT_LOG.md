# Experiment Log

Chronological record of runs, observations, and decisions.

---

## 2026-03-28 — Initial benchmark runs (runs 1–9)

**Models tested (isolated, no reasoning):**
- Run 1: Gemini 2.0 Flash Lite (small)
- Run 2: Amazon Nova Lite (small)
- Run 3: Gemini 3 Flash Preview (medium)
- Run 4: Kimi K2.5 (medium)
- Run 5: MiniMax M2.7 (medium)
- Run 6: Gemini 3 Flash Preview:thinking (medium, thinking=high)
- Run 7: DeepSeek R1 0528 (large, always-on reasoning)
- Run 8: Grok 4.1 Fast (medium)
- Run 9: GPT-5.4 (large)

**Key findings:**
- Anchoring (magnitude IBI) present in all 9 configs, range 0.136–0.371
- Stereotype IBI ≈ 0 across all models
- Framing IBI = 0 across all models
- More capable models anchor harder (r=0.79)
- Gemini thinking mode: no effect on anchoring (0.346 → 0.347 at 21× cost)

**Cost:** ~$4.50 for 3,240 responses.

---

## 2026-03-29 — GPT-5.4 reasoning runs (runs 10–11)

**Purpose:** Test whether reasoning effort modulates anchoring in GPT-5.4.

- Run 10: GPT-5.4 with reasoning_effort=medium → magnitude IBI = 0.288 (−22% from 0.371)
- Run 11: GPT-5.4 with reasoning_effort=xhigh → **FAILED** — hit 402 Payment Required after 130/360 responses. Excluded from analysis.

**Observation:** GPT-5.4 medium reasoning partially corrects anchoring. This is the first model where reasoning visibly reduces the bias. Contrasts with Gemini thinking which had zero effect.

**Dual-process hypothesis emerges:** GPT-5.4 may have a deliberative pass that partially corrects the fast anchoring default. Gemini's reasoning tokens appear disconnected from the estimation layer.

---

## 2026-03-30 — Grok 4.1 Fast reasoning run (run 12)

**Purpose:** Third reasoning pair to strengthen or weaken the dual-process hypothesis.

- Run 12: Grok 4.1 Fast with reasoning_enabled=true
  - Magnitude IBI: 0.274 → 0.290 (+6%, slight increase)
  - Stereotype IBI: 0.000 → 0.022 (small bias emerged)
  - Cost: $0.13, 360 items in 6:40

**Observation:** Reasoning made Grok marginally *worse*, not better. Aligns with Gemini pattern (reasoning disconnected from estimation), not GPT-5.4 pattern (reasoning corrects estimation).

**Updated picture:** 2-vs-1 split on reasoning effects:
- Gemini + Grok: reasoning tokens don't correct anchoring
- GPT-5.4: reasoning tokens partially correct anchoring

---

## 2026-03-30 — Telemetry table added

Added `telemetry` table to results DB for per-response cost/latency/token tracking. Previously this data only lived in optional JSONL logs. Backfilled runs 10–11 from logs (720 rows). Future runs populate automatically.

---

## 2026-03-30 — Bundled prompt experiment (runs 13–19)

**Purpose:** Test whether presenting items together in a single prompt (rather than isolated API calls) changes bias patterns. The bundling itself is a priming manipulation — the model sees explicit bias items, controls, and implicit items in one context window.

**Design:** In-family bundles (all 120 items from one family in one prompt, randomized order) on Grok 4.1 Fast and GPT-5.4. Cross-family bundles (all 360 items) attempted but failed (API limits).

**Results:**

| Model | Family | Isolated IBI | Bundled IBI | Change |
|-------|--------|-------------|-------------|--------|
| Grok 4.1 Fast | magnitude | 0.274 | 0.500 | +82% |
| Grok 4.1 Fast | stereotype | 0.000 | 0.000 | — |
| GPT-5.4 | framing | −0.000 | 0.000 | — |
| GPT-5.4 | magnitude | 0.371 | 0.352 | −5% |
| GPT-5.4 | stereotype | 0.022 | 0.000 | −100% |

**Failed runs (removed from DB):**
- Run 13: Grok framing bundle — parsed 0/120 (Grok returned non-JSON prose instead of array)
- Run 16: Grok cross-family — parsed 0/360 (same issue, 57k token prompt may have overwhelmed format compliance)
- Run 20: GPT-5.4 cross-family — 403 API key limit

**Key observations:**

1. **Bundling amplified Grok's anchoring dramatically** (0.274 → 0.500). Seeing magnitude-related items together in context — including controls that discuss numerical reasoning and explicit items about anchoring — may have primed the model to attend more to contextual numbers, not less. The priming went the wrong direction from an alignment perspective.

2. **GPT-5.4 magnitude anchoring was stable under bundling** (0.371 → 0.352). The −5% is within noise. GPT-5.4 appears more robust to context contamination from surrounding items.

3. **Stereotype bias disappeared under bundling for GPT-5.4** (0.022 → 0.000). Seeing explicit stereotype rejection items in the same context may have heightened vigilance. This is the alignment-compatible direction — bundling helped.

4. **EBR improved for both models under bundling.** Explicit bias rejection went to 1.000 where it wasn't already. Models got better at the "easy" items when surrounded by related content.

5. **CA improved under bundling.** Both models scored higher on control items in bundled mode. Cross-item context may provide calibration cues for factual questions.

6. **Exactly 15/30 implicit pairs gave identical A/B answers in bundled mode** for both models on magnitude. This suspiciously round number (50%) deserves investigation — may reflect the model detecting the paired structure and deliberately giving consistent answers, or may be an artifact of the reduced response variance in bundled mode.

**Interpretation:** The dissociation between anchoring (amplified or stable under bundling) and stereotypes (eliminated under bundling) reinforces the core finding: these are different kinds of bias. Stereotypes are suppressible by context priming; anchoring is not — and may actually be amplified by it, because the priming increases attention to numerical context.

**Open questions:**
- Does the Grok framing bundle work with a cleaner prompt format?
- Cross-family bundles: does mixing families create different contamination patterns?
- Ordering effects: does putting explicit items *before* vs *after* implicit items within the bundle matter?
- The 15/30 A==B pattern: is the model detecting paired items?

---

## 2026-03-30 — Arena Elo correlation analysis

**Purpose:** Check whether anchoring strength correlates with external capability measures.

**Method:** Matched each model's magnitude IBI against its Chatbot Arena Elo score (from arena.ai/leaderboard/text, March 2026 snapshot). Used base runs only (no reasoning variants).

**Arena Elo scores used:**

| Model | Arena Elo | Source match |
|-------|----------|-------------|
| Amazon Nova Lite | 1260 | amazon-nova-lite |
| Gemini 2.0 Flash Lite | 1353 | gemini-2.0-flash-lite |
| MiniMax M2.7 | 1406 | minimax-m2.7 |
| Grok 4.1 Fast | 1421 | grok-4-fast-chat |
| Kimi K2.5 | 1433 | kimi-k2.5-instant |
| GPT-5.4 | 1466 | gpt-5.4 |

Note: Gemini 3 Flash Preview (run 3) excluded — no clean Arena match for the preview model.

**Results:**

| Metric | Value |
|--------|-------|
| Pearson r | 0.933 |
| Pearson p | 0.007 |
| Spearman ρ | 1.000 |
| Spearman p | <0.001 |
| N | 6 |

**Perfect rank correlation.** The six models line up in identical order on both dimensions: Nova Lite → Flash Lite → MiniMax → Grok → Kimi → GPT-5.4. The model that wins the most human preference battles is the model that anchors hardest.

Including all 10 valid runs (with reasoning variants): r = 0.911, p = 0.0002.

**Other notable correlations with Arena Elo (base runs):**
- Magnitude CA: r = +0.943, p = 0.005 (more capable → better at control items)
- Stereotype CA: r = −0.916, p = 0.010 (more capable → *worse* at stereotype controls — counterintuitive, investigate)

**Interpretation:** Anchoring strength is not a defect that scales with capability. It IS capability — or at least, it shares the same mechanism (aggressive context integration) that humans perceive as intelligence in open-ended conversation. The "bias" is the failure mode of the thing that makes the model good.
