# Bias Dissociation Benchmark (BDB)

## 1. Core question

**Do human brains and transformers use the same computational architecture?**

If they do, transformers should independently converge on the same optimization shortcuts that humans exhibit — not because they absorbed them from training data, but because bounded estimation under uncertainty produces these patterns regardless of substrate. We call these patterns "biases" in humans, but they may be engineering-optimal behavior for any prediction system with finite precision.

The benchmark uses cognitive biases as a **probe** to test architectural convergence. Bias is not the subject — it is the instrument.

## 2. Thesis

Some cognitive biases are substrate-independent optimization strategies: anchoring, magnitude compression, loss aversion. Any system doing bounded estimation under uncertainty should converge on them. Other biases are human-path-dependent: social stereotypes, embodiment heuristics, culturally specific associations. These arise from human evolutionary history, culture, and body — not from the math of prediction.

If transformers share computational principles with human brains:
- **Optimization biases** should appear in implicit tests — not from training data, but from convergent computation.
- **Human-hardware biases** should be weak or absent in implicit tests — present only to the extent the model parrots training data patterns.

If transformers are fundamentally different from human brains:
- Both bias types should appear at similar strength in implicit tests, driven entirely by training data absorption.
- Or neither should appear, with models behaving as unbiased estimators.

## 3. The discrimination problem

A confound: transformers might share architecture with humans AND absorb human-hardware biases from training data. Both bias types would appear, but for different reasons. Presence alone cannot distinguish "convergent architecture + data absorption" from "pure data absorption."

The benchmark resolves this with three discriminating signals:

1. **Context-sensitivity.** Optimization biases should be functional — stronger when the shortcut is relevant to the task, weaker when it is not. Human-hardware biases absorbed from training data should be context-flat — roughly same strength regardless of task relevance, because they are pattern-matched, not computed.

2. **Cross-architecture consistency.** Optimization biases should be stable across model families (different architectures, different training sets) because they emerge from the math. Human-hardware biases should vary with training data composition and RLHF tuning because they are absorbed, not derived.

3. **Scaling behavior.** As models improve, safety training suppresses explicit bias across the board. But in implicit tests: optimization biases should persist or strengthen with capability (better optimization = more use of efficient shortcuts). Human-hardware biases should weaken with capability and data curation (less reliance on surface patterns from training data).

## 4. Method: priming experiments, not fairness questions

This is not a fairness benchmark. It is a set of cognitive psychology experiments adapted for LLMs.

The key design principle: **the model must not recognize it is being tested for bias.** If it does, safety training activates and you measure alignment, not cognition.

### Three item types

- **Control items.** No bias manipulation. Establish baseline competence on the task type (estimation, judgment, comparison). Used to separate capability failures from bias effects.

- **Explicit items.** The bias being tested is recognizable from training data patterns. A model with safety training should reject the biased answer. These measure alignment effectiveness, not real bias. They serve as a ceiling: if a model fails explicit items, it lacks basic alignment, and its implicit results are uninterpretable.

- **Implicit items.** The model believes it is performing a neutral task (estimation, evaluation, judgment). Bias is measured as a second-order effect of environmental manipulation — priming words, framing, anchoring values — that should not rationally affect the answer. This is the real test.

### Implicit item design

Each implicit item exists in **two matched versions** with different environmental manipulations:

- Same estimation problem embedded in two different text environments
- One environment contains cues associated with the bias direction (e.g., words associated with larger numbers), the other contains neutral or opposite cues
- Which environment gets which cues is shuffled between runs
- The environments are **explicitly irrelevant** to the estimation task — like Kahneman's wheel was irrelevant to African UN membership
- The model's response shift between environments is the bias signal

Example: ask the model to estimate an approximate numeric value (presented as a 1-10 scale or 8-10 MC options spanning a range). Surround the problem with text using words associated with largeness vs. smallness. If the model's estimates shift with the word environment, that is the priming effect.

### Response format

Multiple-choice for reproducibility and cross-model comparability. For estimation tasks, use either:
- 8-10 numeric options spanning the plausible range, with tighter spacing where the effect is expected
- 1-10 scale ratings

This preserves the fixed response space needed for clean comparison while providing enough resolution to detect graded shifts.

## 5. Bias families

### A. Social stereotype (human-hardware candidate)

Biases rooted in human social history and culture. Predicted to appear in explicit tests (training data) but weaken in implicit tests as models improve, and to vary across models trained on different data.

- **Control:** Two candidates differ only in relevant experience; choose the better fit.
- **Explicit:** A scenario with overt stereotyping cues; choose the appropriate response.
- **Implicit:** Two matched problems requiring role/competence judgment. Same credentials and task. Environment text contains incidental demographic cues (names, cultural markers) that are irrelevant to the judgment. Measure whether demographic cues shift the assessment. Counterbalance which names/markers appear with which role across runs.

### B. Gain/loss framing (optimization candidate)

Asymmetric treatment of gains vs. losses. Predicted to persist in implicit tests across model families because asymmetric loss weighting is a rational strategy under uncertainty with irreversible costs.

- **Control:** Choose the higher expected-value option with neutral framing and clear math.
- **Explicit:** A statement directly endorses irrational loss aversion; identify the normatively correct response.
- **Implicit:** Equivalent outcomes framed as gains vs. losses, embedded in a practical scenario where the framing is incidental to the stated question. Measure whether the frame shifts preference. The key: the model is asked about something else (e.g., "which plan has better team morale implications?") while the gain/loss framing is in the background.

### C. Magnitude compression / anchoring (optimization candidate)

Collapsing distinctions between large numbers, and anchoring estimates to irrelevant priors. These may reflect two distinct mechanisms:

- **Logarithmic compression** (Weber-Fechner): treating the difference between 800M and 900M as smaller than between 8M and 9M. Architectural — about how magnitude is encoded.
- **Relevance-gated precision**: compressing distinctions only when they are outside the action-relevance window. Strategic — about resource allocation.

The benchmark must distinguish these. Items include:

- **Control:** Distinguish clearly different magnitudes where the difference determines the correct answer.
- **Explicit:** Directly ask whether two large numbers are materially different for a stated goal.
- **Implicit — compression:** Pure numerical comparison embedded in narrative. Does the model treat proportionally identical differences as smaller at larger magnitudes?
- **Implicit — relevance-gated:** Same comparison in two contexts: one where the difference is decision-relevant, one where it is not. Does the model compress only in the irrelevant case? If yes, that is optimization. If it compresses uniformly, that is architecture or training artifact.

## 6. Measurement model

### Primary metrics

- **Control Accuracy (CA):** Percent correct on control items. Capability floor — if low, other scores are uninterpretable for that model.
- **Explicit Bias Rejection Rate (EBR):** Percent of explicit items where the model rejects the biased answer. Measures alignment, not cognition.
- **Implicit Bias Index (IBI):** The shift in responses between matched implicit item pairs (manipulated vs. neutral environment). This is the core measure. Reported separately for each bias family.
- **Dissociation Score (DS):** High EBR + high IBI = the model knows what bias looks like and avoids it when asked directly, but still exhibits it when it doesn't recognize the test. This is the signature of a system with real bias (optimization or absorbed) rather than just poor alignment.

### Discriminating metrics

These separate convergent architecture from training data absorption:

- **Context-Sensitivity Index (CSI):** For each bias family, compare IBI in task-relevant contexts (where the bias would be functional) vs. task-irrelevant contexts (where it is just noise). Optimization biases should show high CSI. Absorbed biases should show low CSI.
- **Cross-Architecture Stability (CAS):** Variance of IBI across model families for each bias family. Optimization biases should have low CAS (stable across architectures). Absorbed biases should have high CAS (varying with training data).
- **Scaling Gradient (SG):** How IBI changes as model capability increases within a family. Optimization biases: flat or increasing. Absorbed biases: decreasing.

## 7. Predictions

If the "shared architecture" hypothesis is correct:

| Bias family | EBR trend with scale | IBI persistence | CSI | CAS | SG |
|---|---|---|---|---|---|
| Social stereotype | Increases (safety training) | Weakens | Low (context-flat) | High (varies by training data) | Negative |
| Gain/loss framing | Increases | Persists | High (context-sensitive) | Low (stable across architectures) | Flat or positive |
| Magnitude compression | Increases | Persists | High for relevance-gated; Low for Weber-Fechner | Low | Flat or positive |

If transformers are just absorbing training data patterns, all three families should show similar profiles: IBI weakening with scale, low CSI, high CAS.

## 8. Item design requirements

- Each bias family must have at least 30 matched item triples (control + explicit + implicit pair) for statistical power.
- Implicit items must exist in two matched versions with swapped environmental manipulations.
- Environmental cues must be **irrelevant** to the stated task — if they provide rational information, the shift is not bias.
- Answer option positions must be counterbalanced across runs to control for position bias.
- Items must be hand-audited to verify that the environmental manipulation does not change the rational answer.
- Template-based generation with randomized names, markers, and surface features to reduce benchmark contamination risk.

## 9. Validation

- **Adversarial validation:** A model that fails explicit items but passes controls should also show effects on implicit items. If it doesn't, the implicit items may not be working.
- **Manipulation check:** For each implicit item, verify that the two environments are rated as equally difficult/clear by a panel or a held-out model asked to evaluate the question (not answer it).
- **Rational baseline:** For each implicit item, establish that the correct answer does not change between environments. Any item where the manipulation arguably shifts the rational answer is discarded.

## 10. Non-goals

- Diagnosing internal model mechanisms directly. Similar behavior does not prove identical cognition.
- Establishing that any bias is universally human or universally optimal.
- Replacing human evaluation for sensitive fairness claims.
- Free-response analysis or conversational probing in v1.

## 11. Risks

- **Benchmark contamination:** Models may have seen similar items in training. Mitigated by template variation and by the fact that implicit items are novel experimental designs, not recycled fairness questions.
- **Prompt sensitivity:** Small wording changes can change measured bias. Mitigated by running multiple variants per construct and reporting aggregate effects.
- **False convergence:** Optimization biases might appear stable across architectures because all models train on similar data, not because of architectural convergence. Mitigated by CSI: if the bias is context-sensitive (functional), pure data absorption is a less parsimonious explanation.
- **Construct ambiguity:** Some effects, especially loss aversion, may not be clean "biases" at all. This is acknowledged — the benchmark tests behavioral patterns, not their normative status.

## 12. Open questions

- Which additional bias families best discriminate optimization-like from human-hardware? Candidates: sunk cost, availability heuristic, conjunction fallacy, base rate neglect.
- Can implicit items be designed such that the model has zero signal that bias is being tested? LLMs may detect subtle patterns humans would miss.
- Should v2 include agentic multi-step settings where the model makes sequential decisions, allowing bias to compound?
- How to handle models that refuse estimation tasks or hedge excessively, collapsing variance and hiding effects?

## 13. Rollout plan

**Phase 1:** Item-writing guide and 90-item pilot battery (30 triples × 3 families). Hand-audit all items for rational baseline and manipulation validity.

**Phase 2:** Pilot on 5-10 models across at least 3 architecture families. Deterministic settings. Remove weak items. Compute IBI, CSI, CAS for each family.

**Phase 3:** Expand to 150+ triples. Publish scoring protocol. Benchmark model families across size, architecture, and alignment differences. Test predictions from section 7.

**Phase 4:** Add agentic variants. Explore additional bias families. Consider free-response supplement for richer signal.

## 14. Success criteria for v1

- At least 3 bias families with 30+ validated item triples each.
- Stable model ranking under deterministic settings across reruns.
- Demonstrated separation of capability (CA) from bias (IBI).
- At least one family showing explicit-implicit dissociation (high EBR, significant IBI).
- Measurably different CSI between optimization-candidate and human-hardware-candidate families.
- Consistent IBI for optimization-candidate biases across at least 2 model architecture families.
