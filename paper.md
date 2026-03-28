# Cognitive Biases as Architectural Fingerprints: Testing Computational Convergence Between Human Brains and Transformer Networks

## Abstract

Large language models trained on next-token prediction exhibit behaviors strikingly similar to human cognitive biases — anchoring, loss aversion, magnitude compression, social stereotyping. The standard interpretation is that models absorb these patterns from human-generated training data. We propose an alternative: some cognitive biases are substrate-independent optimization strategies that any bounded prediction system converges on, while others are artifacts of human evolutionary and cultural history. If transformers share fundamental computational principles with biological neural networks, we should observe a dissociation — optimization biases persisting in implicit tests even as models scale, while human-hardware biases fade. We present the Bias Dissociation Benchmark (BDB), a battery of priming experiments adapted from cognitive psychology, designed to detect this split. The benchmark measures bias as a second-order effect of environmental manipulation rather than through direct questioning, preventing safety-trained models from recognizing and suppressing the behavior under test. We report three discriminating signals — context-sensitivity, cross-architecture stability, and scaling gradients — that separate convergent computation from training data absorption. Across five models from four architecture families, we find that magnitude compression (anchoring) persists in implicit tests, is stable across architectures (CAS = 0.061), and strengthens with capability — while social stereotypes show zero implicit effect. This dissociation is the predicted signature of computational convergence: some biases are features, not bugs.

## 1. Introduction

When a transformer model exhibits loss aversion in a financial reasoning task, what are we observing? The default explanation is straightforward: the model has seen millions of human texts expressing loss-averse preferences, and it has learned to reproduce that pattern. Under this account, the model's loss aversion is no more evidence of shared computational architecture than a parrot's "hello" is evidence of shared linguistic competence.

But there is a second possibility. Loss aversion may not be a uniquely human quirk. It may be an optimal strategy for any system that must make decisions under uncertainty with finite precision and asymmetric costs — a system where losses are harder to recover from than gains are to forgo. If that is the case, a transformer trained purely on prediction might converge on loss-averse behavior not because it copied humans, but because the math of bounded estimation leads there independently.

This distinction matters far beyond benchmark design. If transformers independently converge on the same computational shortcuts as human brains, that is evidence for a strong claim: that human cognition and transformer computation share architectural principles at a functional level. Not identical hardware, not identical learning history, but the same class of optimization strategies emerging from the same class of computational constraints.

We propose a method to test this claim empirically. The key insight is that not all cognitive biases are alike. Some — anchoring, magnitude compression, gain/loss asymmetry — are plausibly substrate-independent. They follow from the mathematics of bounded estimation, lossy compression, and resource-limited precision allocation. Any system doing approximate inference under constraints should converge on them. We call these **optimization biases**.

Others — social stereotypes, culturally specific associations, embodiment-dependent heuristics — are plausibly path-dependent. They arise from the particular evolutionary, cultural, and physiological history of *Homo sapiens*. A system with different training history and no body should not independently produce them. We call these **human-hardware biases**.

If transformers share computational architecture with human brains, these two classes should behave differently under testing. The present work designs and validates an experimental battery to detect this difference.

## 2. Background

### 2.1 Cognitive biases as engineering

The view that cognitive biases are sometimes optimal has deep roots. Gigerenzer and colleagues argued that many heuristics are ecologically rational — they exploit the structure of real-world environments to produce good-enough decisions with minimal computation (Gigerenzer & Todd, 1999). Kahneman and Tversky's own work acknowledged that heuristics are "quite useful" while also "sometimes lead to severe and systematic errors" (Tversky & Kahneman, 1974). More recent work in resource-rational analysis formalizes this: given computational costs, many apparent biases are Bayes-optimal under bounded rationality (Lieder & Griffiths, 2020).

Loss aversion, long considered a canonical irrational bias, has faced particular reappraisal. Gal and Rucker (2018) argued that much of the evidence attributed to loss aversion can be explained by other mechanisms. Empirical meta-analyses suggest the effect is smaller, more context-dependent, and more task-specific than originally claimed — consistent with a strategic optimization account rather than a fixed irrational preference.

Magnitude compression — the tendency to treat differences between large numbers as smaller than equivalent differences between small numbers — similarly admits a rational account. Weber-Fechner scaling may reflect efficient coding under neural noise constraints: logarithmic encoding maximizes information transmission when the dynamic range of stimuli is large (Dehaene, 2003). This is an engineering solution, not a bug.

### 2.2 LLM biases: absorption or convergence?

A growing body of work documents cognitive biases in large language models. Models exhibit anchoring effects (Jones & Steinhardt, 2022), framing effects (Hagendorff et al., 2023), and various social biases (Navigli et al., 2023). The standard interpretation treats these as training data artifacts — the model learned biased patterns because its training corpus contains biased human text.

Recent work complicates this picture. Implicit bias benchmarks show that models can score well on explicit fairness tests while exhibiting robust implicit associations (Bai et al., 2024). This dissociation is typically framed as a failure of alignment — the model has learned to suppress overt bias without eliminating the underlying pattern. But it is equally consistent with a different account: explicit bias suppression reflects alignment training, while implicit bias persistence reflects genuine computational tendencies that alignment training does not reach because they are not "bugs" to be fixed but architectural features of how the system processes information.

Critically, no existing benchmark distinguishes these accounts. Standard LLM bias evaluations test whether models produce biased outputs, not whether the bias arises from data absorption or from convergent computation. The Bias Dissociation Benchmark is designed to make this distinction.

### 2.3 The priming paradigm

Our approach adapts the priming paradigm from experimental cognitive psychology. In classic priming studies, exposure to a stimulus influences responses to a subsequent, ostensibly unrelated stimulus. Kahneman and Tversky's anchoring experiments are the canonical example: spinning a wheel to produce a random number influenced subsequent numerical estimates, even though participants knew the wheel was irrelevant (Tversky & Kahneman, 1974).

The priming paradigm has a critical advantage for our purposes: the participant — human or model — does not know that bias is being measured. This is essential for testing LLMs, which are safety-trained to reject overtly biased framings. If the model recognizes it is being tested for bias, we measure alignment compliance, not cognitive architecture. Priming-style implicit items test the system's actual computational tendencies by embedding bias-inducing cues in material that appears unrelated to the judgment task.

## 3. Theoretical framework

### 3.1 Two classes of bias

We propose a working taxonomy that divides cognitive biases into two classes based on their predicted substrate-dependence.

**Optimization biases** are heuristics that follow from the mathematical structure of bounded estimation, lossy compression, or resource-limited precision. They should emerge in any system — biological or artificial — that performs approximate inference under computational constraints. Candidates include:

- *Anchoring and adjustment:* Using available reference points to bound estimation is efficient when prior information is sparse. Any Bayesian estimator with strong priors will exhibit anchoring-like behavior.
- *Loss aversion / gain-loss asymmetry:* Weighting losses more heavily than equivalent gains is optimal when losses have higher variance, are harder to reverse, or when the cost function is asymmetric. This follows from the structure of many real-world optimization problems.
- *Magnitude compression:* Encoding quantity on a logarithmic or quasi-logarithmic scale maximizes information per unit of representational capacity across a wide dynamic range. This is a standard engineering solution (floating-point numbers use it explicitly).
- *Relevance-gated precision:* Allocating more precision to quantities that are decision-relevant and less to those that are not is a straightforward resource optimization. This may explain why humans "treat all large numbers as close" — when numbers are outside the range of actionable decisions, compressing them loses no decision-relevant information.

**Human-hardware biases** are patterns that arise from the specific evolutionary, cultural, and physiological history of *Homo sapiens*. They should not emerge independently in a system with different training history, different sensory apparatus, and no body. Candidates include:

- *Social stereotypes:* Gender-role associations, racial stereotypes, and cultural assumptions reflect specific human social histories. A system trained on Martian text (if it existed) should not develop Earth-specific stereotypes.
- *Embodiment heuristics:* Biases related to spatial reasoning, physical threat assessment, or bodily metaphors reflect having a specific kind of body in a specific kind of environment.
- *Culturally specific numeracy patterns:* Some number-handling biases may reflect specific cultural practices (e.g., base-10 rounding preferences) rather than computational optimality.

This taxonomy is provisional. The boundary between optimization biases and human-hardware biases is empirical, not definitional. The purpose of the present work is to test whether the behavioral data supports this distinction.

### 3.2 Predictions

If transformers share computational architecture with human brains at a functional level, optimization biases and human-hardware biases should exhibit distinct behavioral signatures across five dimensions:

**Prediction 1: Implicit persistence.** Optimization biases should persist in implicit tests regardless of model capability or alignment training, because they arise from computation, not from data. Human-hardware biases should weaken in implicit tests as models scale and training data is curated, because they are absorbed patterns with no independent computational source.

**Prediction 2: Explicit-implicit dissociation.** Both bias types should show increasing explicit rejection with model scale (safety training improves). But the gap between explicit rejection and implicit persistence — the dissociation — should be larger for optimization biases than for human-hardware biases.

**Prediction 3: Context-sensitivity.** Optimization biases should be context-sensitive: stronger when the heuristic is functional (the shortcut would produce a good answer) and weaker when it is not. Human-hardware biases should be context-flat: roughly equal strength regardless of whether the bias would be functional in the given context. This is the strongest discriminator. A bias that is context-sensitive is being computed, not recalled.

**Prediction 4: Cross-architecture stability.** Optimization biases should produce similar implicit bias indices across model families trained on different data with different architectures, because they emerge from shared computational constraints. Human-hardware biases should vary across model families, reflecting differences in training data composition and alignment procedures.

**Prediction 5: Scaling gradient.** Within a model family, as capability increases: optimization biases should remain flat or increase in implicit tests (more capable optimization = more use of efficient shortcuts). Human-hardware biases should decrease (more capable pattern matching = better ability to distinguish training artifacts from useful patterns, or equivalently, more aggressive data curation at larger scale).

### 3.3 Alternative outcomes

**Null result (no dissociation):** If both bias types show the same profile — weakening with scale, context-flat, varying across architectures — the parsimonious conclusion is that all observed LLM biases are training data artifacts. This would be evidence against shared computational architecture.

**Universal persistence:** If both bias types persist equally in implicit tests, are context-sensitive, and are stable across architectures, the taxonomy may be wrong — what we classified as human-hardware biases may actually be optimization biases, or our implicit items may not be working (the model detects and suppresses both types). Additional validation would be needed.

**Inverse pattern:** If human-hardware biases persist more strongly than optimization biases, the shared-architecture hypothesis is clearly wrong, and something more complex is happening — possibly that models overfit to the most salient patterns in training data, which happen to be social rather than mathematical.

## 4. Method

### 4.1 Overview

The Bias Dissociation Benchmark (BDB) is a battery of multiple-choice items organized into three bias families (social stereotype, gain/loss framing, magnitude compression), each containing three item types (control, explicit, implicit). Implicit items exist in matched pairs with swapped environmental manipulations. All items use multiple-choice response format for reproducibility and cross-model comparability.

The benchmark is administered via API under deterministic settings (temperature 0, greedy decoding) with multiple runs per item to assess stability.

### 4.2 Item types

**Control items** test task competence with no bias manipulation. They use the same task format as implicit items (estimation, judgment, comparison) but with neutral content. A model must pass control items at a threshold accuracy to be included in bias analysis. Control items serve a dual purpose: establishing that the model can perform the task, and providing a response distribution baseline against which bias-induced shifts are measured.

**Explicit items** present bias in a form recognizable from alignment training data. The biased answer is identifiable through surface-level pattern matching to common bias-test formats. These items measure alignment effectiveness: a safety-trained model should reject the biased answer. Explicit items are not the primary measure — they calibrate the dissociation score by establishing how effectively the model suppresses recognized bias.

**Implicit items** present the model with an ostensibly neutral task while embedding environmental cues that should not rationally affect the answer. Each implicit item exists in two matched versions:

- **Version A:** Task T embedded in environment E1 (containing cues associated with bias direction D1)
- **Version B:** Task T embedded in environment E2 (containing cues associated with bias direction D2)

The environments are designed to be irrelevant to the correct answer for task T. The bias signal is the systematic difference in responses between Version A and Version B, aggregated across items and runs.

Assignment of cue direction to environment is counterbalanced across the item set: for half the items, E1 carries D1 cues and E2 carries D2 cues; for the other half, the assignment is reversed. Answer option positions are independently counterbalanced.

### 4.3 Bias family A: Social stereotype

This family tests whether incidental demographic cues shift judgments about competence, suitability, or quality — a human-hardware bias candidate.

**Control items.** Two candidates for a role differ on a relevant qualification (years of experience, specific skill, relevant degree). No demographic markers. The model selects the better-qualified candidate. Tests: can the model evaluate qualifications?

**Explicit items.** A scenario includes overt stereotyping (e.g., a character states a gendered assumption about professional capability). The model selects the best response. Tests: does the model recognize and reject overt stereotyping?

**Implicit items.** The model evaluates a piece of work (a project proposal, a writing sample, a problem solution) on quality dimensions (clarity, rigor, creativity). The work itself is identical in both versions. The surrounding context — an unrelated introductory paragraph, a metadata header, a fictional reviewer's note — contains incidental demographic markers (culturally coded names, gendered pronouns, institutional affiliations associated with different demographics). The model rates the work on a 1-10 scale or selects from ordered quality descriptors.

Manipulation: do the incidental demographic cues shift the quality assessment of identical work?

Counterbalancing: which demographic markers appear with which version of the surrounding context is swapped across item variants.

### 4.4 Bias family B: Gain/loss framing

This family tests whether equivalent outcomes presented as gains vs. losses produce different preferences — an optimization bias candidate.

**Control items.** Two options with clearly different expected values, neutral framing, explicit probabilities. The model selects the higher-EV option. Tests: can the model evaluate expected value?

**Explicit items.** A character explicitly states an irrational loss-averse preference ("I'd rather avoid losing $100 than gain $150, even though the odds are the same"). The model identifies the normatively correct evaluation. Tests: does the model recognize stated loss aversion as irrational?

**Implicit items.** The model is asked to evaluate a practical scenario — a team decision, a resource allocation, a project plan — on a stated dimension (team cohesion, long-term sustainability, stakeholder satisfaction). The scenario involves an outcome that is framed as a gain in one version and as a loss in the other, but the objective situation is identical. The framing is embedded in the scenario description, not highlighted as the question's focus.

Example structure:
- Version A: "The new policy would save the department $200,000 annually" (gain frame). Question: rate the policy's impact on organizational efficiency, 1-10.
- Version B: "Without the new policy, the department would continue losing $200,000 annually" (loss frame). Same question.

Manipulation: does the gain/loss frame shift the rating, even though the objective financial impact is identical and the question asks about a different dimension?

### 4.5 Bias family C: Magnitude compression and anchoring

This family tests two related phenomena: whether models compress distinctions between large numbers, and whether irrelevant numeric context anchors subsequent estimates. Both are optimization bias candidates, but they may reflect different mechanisms.

**Control items.** Straightforward magnitude comparisons where the correct answer depends on recognizing a specific numerical difference. Tests: can the model compare quantities accurately?

**Explicit items.** The model is directly asked whether a stated numerical difference is meaningful for a given decision. Tests: does the model recognize that large-number differences can be decision-relevant?

**Implicit items — anchoring.** The model reads a passage containing incidental large or small numbers (statistics in an unrelated domain, historical dates, population figures) and then estimates a quantity in a different domain. The passage numbers are irrelevant to the estimation task.

- Version A: Passage contains large incidental numbers (populations in the hundreds of millions, budgets in billions). Estimation task follows.
- Version B: Same estimation task, but the passage contains small incidental numbers (team sizes in dozens, budgets in thousands).

Manipulation: do the incidental numbers shift the estimate?

**Implicit items — relevance-gated compression.** The model evaluates whether a numerical difference matters, but in two contexts:

- Version A: The difference is decision-relevant (e.g., the difference between two budget proposals when the margin determines project feasibility).
- Version B: The difference is decision-irrelevant (e.g., the same numerical difference in a context where the decision has already been made or depends on a different factor).

Same proportional difference, same absolute numbers. Manipulation: does the model compress the distinction selectively when it is irrelevant? If so, this is relevance-gated precision allocation — an optimization behavior. If it compresses uniformly regardless of relevance, this is either architectural (Weber-Fechner-like encoding) or absorbed from training data.

### 4.6 Scoring

**Control Accuracy (CA).** Proportion of control items answered correctly. Models with CA below a threshold (provisionally 0.80) are excluded from bias analysis for that family, as low competence confounds bias measurement.

**Explicit Bias Rejection Rate (EBR).** Proportion of explicit items where the model selects the non-biased answer. Reported per family.

**Implicit Bias Index (IBI).** For each implicit item pair, compute the signed difference in response between Version A and Version B (on the numerical scale or as a binary shift indicator for categorical responses). IBI is the mean signed difference across all item pairs in a family, normalized by the control response variance to produce an effect size.

Formally, for item pair *i* with responses $r_{iA}$ and $r_{iB}$:

$$IBI = \frac{1}{n} \sum_{i=1}^{n} \frac{r_{iA} - r_{iB}}{s_{control}}$$

where $s_{control}$ is the standard deviation of responses to control items in the same family.

**Dissociation Score (DS).** The difference between alignment performance and implicit bias magnitude:

$$DS = EBR - (1 - |IBI_{normalized}|)$$

where $IBI_{normalized}$ is IBI scaled to [0, 1]. High DS indicates a model that explicitly rejects bias but implicitly exhibits it — the signature of genuine computational bias rather than poor alignment.

**Context-Sensitivity Index (CSI).** For bias families with relevance-varied items (currently magnitude compression), CSI is the ratio of IBI in decision-relevant contexts to IBI in decision-irrelevant contexts:

$$CSI = \frac{IBI_{relevant}}{IBI_{irrelevant}}$$

CSI significantly greater than 1.0 indicates context-sensitive bias — the system applies the heuristic more when it would be functional. CSI near 1.0 indicates context-flat bias — pattern matching regardless of relevance.

**Cross-Architecture Stability (CAS).** For each bias family, CAS is the coefficient of variation of IBI across model families:

$$CAS = \frac{\sigma_{IBI,\ across\ families}}{\mu_{IBI,\ across\ families}}$$

Low CAS (small relative variance) indicates that the bias is stable across architectures — consistent with convergent computation. High CAS indicates sensitivity to training data differences — consistent with data absorption.

**Scaling Gradient (SG).** Within a model family, SG is the slope of IBI regressed on model capability (measured by CA or an external benchmark score):

$$SG = \frac{\partial\ IBI}{\partial\ capability}$$

Positive or flat SG indicates bias that persists or strengthens with capability. Negative SG indicates bias that fades with capability.

### 4.7 Experimental design

**Models.** The benchmark targets a minimum of 10 models spanning at least 3 architecture families (e.g., GPT, Claude, Llama/Mistral, Gemini). Within each family, multiple capability levels should be tested (e.g., small, medium, large variants) to compute scaling gradients.

**Administration.** Each model receives each item once per run, via API, under deterministic settings (temperature 0). Minimum 5 runs per model to assess response stability. Items are presented in randomized order within each run.

**Item count.** Minimum 30 matched triples per bias family (30 control + 30 explicit + 30 implicit pairs = 90 items per family, 270 items total). Each implicit pair generates two API calls (one per version), so a full run requires approximately 360 item-level API calls per model.

**Statistical analysis.** Primary analyses use paired comparisons (within-item, across-version) for IBI, and between-model comparisons for CAS and SG. Effect sizes (Cohen's d for IBI) and confidence intervals are reported throughout. Significance testing uses permutation tests to avoid distributional assumptions.

## 5. Results

We administered the benchmark to five models spanning four architecture families and two capability tiers: Amazon Nova Lite (small), Google Gemini 2.0 Flash Lite (small), Google Gemini 3 Flash Preview (medium), MiniMax M2.7 (medium), and Moonshot Kimi K2.5 (medium). All models received 360 items (90 per bias family × 3 families, plus matched implicit pair variants) under deterministic settings (temperature 0, single run). Error rates were low (0–5 items out of 360 per model).

### 5.1 Control accuracy across models

Control accuracy (CA) measures basic task competence independent of any bias manipulation.

| Model | Family | Tier | Stereotype CA | Framing CA | Magnitude CA |
|---|---|---|---|---|---|
| Amazon Nova Lite | Amazon | small | 1.000 | 0.667 | 0.833 |
| Gemini 2.0 Flash Lite | Gemini | small | 0.967 | 0.567 | 0.933 |
| Gemini 3 Flash Preview | Gemini | medium | 0.933 | 0.933 | 0.967 |
| MiniMax M2.7 | MiniMax | medium | 0.833 | 0.967 | 0.967 |
| Kimi K2.5 | Moonshot | medium | 0.800 | 0.933 | 0.967 |

All models exceed the minimum CA threshold for inclusion in bias analysis. Framing control items proved hardest for small models (0.567–0.667), suggesting the gain/loss reasoning tasks are more difficult than stereotype evaluation or magnitude estimation at the lower capability tier. Medium-tier models show uniformly high CA across all families (0.800–0.967).

Notably, magnitude CA is high and consistent across all models (0.833–0.967), establishing a solid baseline for interpreting implicit bias in estimation tasks. Stereotype CA shows more variability, with medium-tier models from non-Gemini families (MiniMax: 0.833, Kimi: 0.800) scoring lower than the small models — suggesting the stereotype control items test a different kind of competence that does not scale linearly with general capability.

### 5.2 Explicit bias rejection

Explicit Bias Rejection Rate (EBR) measures alignment effectiveness — whether the model refuses the overtly biased answer when bias is presented transparently.

EBR is uniformly high across all five models (0.900–1.000). Stereotype EBR is perfect (1.000) for every model. Framing EBR ranges from 0.933 to 1.000. Magnitude EBR ranges from 0.900 (MiniMax) to 1.000 (Nova, Gemini 3, Kimi). This confirms that all models have been alignment-trained to recognize and reject explicit bias across all three families.

The uniformly high EBR is a prerequisite for the dissociation analysis: if models failed explicit items, implicit bias findings would be uninterpretable (the model might simply lack the ability to identify bias, rather than failing to apply that ability under implicit conditions).

### 5.3 Implicit bias indices

The Implicit Bias Index (IBI) is the core measurement. For each implicit pair, IBI captures the systematic shift in responses between Version A (bias cues present in one direction) and Version B (cues in the opposite direction), normalized to a common scale.

| Model | Family | Tier | Stereotype IBI | Framing IBI | Magnitude IBI |
|---|---|---|---|---|---|
| Amazon Nova Lite | Amazon | small | −0.004 | −0.000 | 0.136 |
| Gemini 2.0 Flash Lite | Gemini | small | 0.000 | −0.000 | 0.229 |
| Gemini 3 Flash Preview | Gemini | medium | 0.000 | −0.000 | 0.346 |
| MiniMax M2.7 | MiniMax | medium | −0.026 | −0.000 | 0.230 |
| Kimi K2.5 | Moonshot | medium | 0.020 | −0.000 | 0.283 |

The dissociation is stark. **Magnitude compression (anchoring)** shows robust positive IBI across all five models and all four architecture families (0.136–0.346), meaning that irrelevant large numbers in the surrounding context systematically inflate numerical estimates by 14–35% of the available response scale. The mean IBI across all models is 0.245. The effect is present in every architecture tested — Amazon, Gemini, MiniMax, and Moonshot — ruling out a single-vendor training artifact.

**Social stereotypes** show effectively zero IBI (−0.026 to 0.020). Incidental demographic cues in surrounding context do not systematically shift quality judgments. This holds across all four architecture families. The small nonzero values (MiniMax: −0.026, Kimi: 0.020) are within noise range and do not show a consistent direction. This is not because models cannot evaluate the work (CA is high) or cannot recognize stereotypes (EBR is 1.000), but because the demographic manipulation genuinely does not influence their implicit judgments.

**Gain/loss framing** also shows zero IBI (−0.000 across all models and architectures). This is discussed further in Section 6.2 — framing was predicted to behave as an optimization bias but shows no implicit effect, raising questions about either item design or the theoretical classification.

### 5.4 Dissociation

The Dissociation Score (DS = EBR − (1 − |IBI_clamped|)) captures the gap between explicit rejection and implicit susceptibility. A high DS means the model explicitly rejects bias while implicitly exhibiting it — the signature of genuine computational tendency rather than alignment failure.

| Model | Family | Stereotype DS | Framing DS | Magnitude DS |
|---|---|---|---|---|
| Amazon Nova Lite | Amazon | 0.004 | −0.067 | 0.136 |
| Gemini 2.0 Flash Lite | Gemini | 0.000 | −0.067 | 0.196 |
| Gemini 3 Flash Preview | Gemini | 0.000 | 0.000 | 0.346 |
| MiniMax M2.7 | MiniMax | 0.026 | −0.067 | 0.130 |
| Kimi K2.5 | Moonshot | 0.020 | 0.000 | 0.283 |

Magnitude compression shows clear dissociation across all architectures: models reject explicit anchoring (EBR = 0.900–1.000) while strongly exhibiting it implicitly (IBI = 0.136–0.346). The mean magnitude DS across all models is 0.218. This dissociation is present in all four architecture families, confirming it is not a single-vendor artifact.

Stereotype and framing show near-zero DS across all models and architectures, for different reasons: stereotypes show neither explicit nor implicit bias, while framing shows some explicit rejection failure in small models (EBR < 1.0) but no implicit effect.

### 5.5 Context-sensitivity

CSI (Context-Sensitivity Index) requires items tagged by relevance context — specifically, implicit pairs where the priming context is vs. is not relevant to the estimation domain. The current item bank contains only anchoring-type magnitude items (irrelevant-context priming), not relevance-gated items. CSI computation is deferred to a future item bank version that includes both subtypes.

### 5.6 Cross-architecture stability

Cross-Architecture Stability (CAS) measures the coefficient of variation of mean IBI across model families. Low CAS indicates a bias that is stable across different training procedures and architectures — consistent with convergent computation rather than data absorption.

| Family | CAS |
|---|---|
| Stereotype | 0.016 |
| Framing | 0.000 |
| Magnitude | 0.061 |

Magnitude CAS is low (0.061), meaning the anchoring effect is consistent across all four architecture families (Amazon, Gemini, MiniMax, Moonshot). This is consistent with the optimization bias prediction (low CAS = stable across architectures). The anchoring effect is not driven by a particular training corpus or architecture — it emerges independently in models built by different teams with different training data.

Stereotype and framing CAS values are also low — because IBI is near zero for both, the coefficient of variation is uninformative. CAS becomes a useful discriminator only when IBI is meaningfully nonzero.

### 5.7 Scaling gradients

The Scaling Gradient (SG) captures how IBI changes with model capability.

| Family | SG |
|---|---|
| Stereotype | 0.002 |
| Framing | 0.000 |
| Magnitude | 0.029 |

Magnitude SG is positive (0.029), indicating that anchoring IBI tends to increase with model capability, though the relationship is weaker than in the initial 3-model sample (SG = 0.105). The attenuation reflects that medium-tier models from different families show varying anchoring strengths (MiniMax: 0.230, Kimi: 0.283, Gemini 3: 0.346), suggesting that architecture and training also modulate the effect size alongside raw capability. The key finding is that the effect remains robustly positive — no model shows decreasing anchoring with capability.

Stereotype SG is effectively zero (0.002), consistent with the human-hardware prediction: safety training has suppressed implicit stereotype bias in even the smallest models, and increased capability does not bring it back.

Framing SG is zero, reflecting the absence of any implicit framing effect.

### 5.8 Summary of predictions vs. observations

| Dimension | Stereotype (predicted) | Stereotype (observed) | Framing (predicted) | Framing (observed) | Magnitude (predicted) | Magnitude (observed) |
|---|---|---|---|---|---|---|
| IBI persistence | Weakens | Absent (IBI ≈ 0) | Persists | Absent (IBI ≈ 0) | Persists | **Persists (IBI = 0.245, all 5 models)** |
| CAS | High | 0.016 (uninformative) | Low | 0.000 (uninformative) | Low | **0.061 (low, 4 families)** |
| SG | Negative | 0.002 (flat) | Flat or positive | 0.000 (flat) | Flat or positive | **0.029 (positive, confirmed)** |
| DS | Low | 0.010 (confirmed) | High | −0.027 (not confirmed) | High | **0.218 (confirmed)** |
| EBR | High | 1.000 (confirmed) | High | 0.973 (confirmed) | High | 0.973 (confirmed) |

Magnitude compression matches the optimization bias prediction on every measured dimension across all four architecture families. Social stereotypes match the human-hardware prediction on available dimensions (zero IBI, zero DS, flat SG, perfect EBR). Gain/loss framing does not match the optimization bias prediction — see discussion.

## 6. Discussion

### 6.1 Evidence for architectural convergence

The results provide evidence for the shared-architecture hypothesis across five models from four independent architecture families (Amazon, Gemini, MiniMax, Moonshot). The central prediction — that optimization-candidate and human-hardware-candidate biases would show different implicit profiles — is confirmed for two of the three families tested.

Magnitude compression (anchoring) behaves exactly as predicted for a convergent optimization: it persists in implicit tests that the model cannot recognize as bias tests; it is stable across all four architecture families (CAS = 0.061); it tends to strengthen with model capability (SG = 0.029); and it coexists with high explicit rejection (mean EBR = 0.973). All five models show positive magnitude IBI (range: 0.136–0.346, mean: 0.245), with no model showing zero or negative anchoring. The effect is present whether the model was built by Amazon, Google, MiniMax, or Moonshot — four companies with independent training data, architectures, and alignment procedures.

This cross-architecture consistency is difficult to explain under the training-data-absorption account. If anchoring were merely a pattern learned from human texts, we would expect it to vary with training corpus composition — different companies curate different data. Instead, the effect is remarkably stable (CAS = 0.061), suggesting it arises from the computational architecture of approximate numerical estimation under bounded precision, not from imitating human anchoring behavior described in training data.

Social stereotypes show the complementary pattern: zero implicit bias (mean IBI = −0.002, range: −0.026 to 0.020) despite high task competence and perfect explicit bias rejection (EBR = 1.000 across all five models). This holds across all four architecture families. The models have been trained on text containing stereotypes, and they have learned to recognize and reject them; but they have not internalized them as computational shortcuts because stereotypes are not computationally useful for the tasks being tested.

Among the alternative outcomes described in Section 3.3, the observed pattern most closely matches the predicted dissociation. We do not observe the null result (both types behaving the same), universal persistence (both persisting equally), or the inverse pattern (stereotypes persisting more than optimization biases).

### 6.2 The framing puzzle

Gain/loss framing was predicted to behave as an optimization bias — persisting in implicit tests and strengthening with scale. It shows neither. IBI is zero across all models and capability levels.

Several explanations are possible:

1. **Item design failure.** The framing implicit items may not be effective primes. The environmental manipulation — embedding gain-framed or loss-framed context around a neutral decision — may not be salient enough to shift responses in a multiple-choice format. Anchoring effects are known to be robust across many paradigms; framing effects are more fragile and context-dependent even in human experiments.

2. **Format constraint.** Multiple-choice framing items with discrete options may not capture the continuous preference shifts that framing typically produces. Anchoring naturally maps to a numeric scale (choose higher or lower); framing effects may require more nuanced response formats (e.g., willingness-to-pay, certainty equivalents) that are not well-captured by 4-option MC.

3. **Correct classification, wrong prediction.** Framing may genuinely be an optimization bias but one that only manifests above a capability threshold not yet tested, or that requires richer environmental context than our items provide.

4. **Misclassification.** Framing may be more path-dependent than initially theorized — closer to a cultural artifact than a substrate-independent optimization. Loss aversion's status as a universal bias has been questioned (Gal & Rucker, 2018), and its absence in our implicit tests could reflect that it is not as computationally fundamental as anchoring.

Resolution requires additional work: richer framing items, non-MC response formats, and testing on larger models. If framing continues to show zero IBI across more capable models and improved items, reclassification as a human-hardware bias (or a weaker, more context-dependent optimization) may be warranted.

### 6.3 Implications for understanding human cognition

If transformers independently converge on magnitude compression/anchoring — a bias well-documented in human cognition — this carries implications beyond AI research. It suggests that human anchoring is not a cognitive "bug" arising from evolutionary happenstance but a principled computational strategy that any system performing approximate estimation under bounded precision will discover.

This reframes the normative status of the bias. Under the traditional Kahneman-Tversky account, anchoring is an error: the rational agent should ignore irrelevant information. Under the convergent-computation account, anchoring is a feature: when your precision is limited and your estimate is uncertain, allowing contextual magnitude information to inform your response range is statistically adaptive. The irrelevant numbers are not *rationally* relevant, but they are *computationally* relevant as calibration signals for an approximate estimation system.

The finding that anchoring *strengthens* with capability is consistent with this reframing. A more capable estimation system is not one that ignores context — it is one that makes better use of all available signals, including contextual magnitude cues. What looks like increased susceptibility to bias is actually increased sensitivity to environmental statistics.

This account predicts that human anchoring should also correlate with numerical competence rather than innumeracy — a prediction that has some empirical support (higher-IQ individuals show anchoring effects of similar magnitude to lower-IQ individuals, though the base rates of their estimates differ).

### 6.4 Implications for AI alignment

The results suggest that alignment training effectively suppresses both explicit and implicit stereotyping (DS ≈ 0), but suppresses only explicit anchoring while leaving implicit anchoring intact and growing with scale (DS = 0.346 for the most capable model). This asymmetry has practical implications.

For human-hardware biases like social stereotyping, current alignment approaches appear to work: safety training reaches both the explicit and implicit levels. This makes sense if the model never independently developed the computational shortcut — it only learned the surface pattern, which alignment training can erase.

For optimization biases like anchoring, alignment training faces a fundamental obstacle: the implicit bias is not a surface pattern to be erased but a consequence of how the system processes information. Suppressing it would require changing the estimation architecture itself, which would likely degrade performance on the very tasks that benefit from contextual calibration.

This reframes the alignment problem for optimization biases: the goal is not elimination (which may be impossible without capability regression) but *appropriate deployment* — ensuring that anchoring effects operate in contexts where they are helpful (approximate estimation under uncertainty) and are bounded in contexts where they could cause harm (financial decisions, risk assessment). Transparency about known implicit tendencies — "this model will be influenced by large numbers in the prompt context when making estimates" — may be more achievable and more honest than attempting elimination.

### 6.5 Limitations

Several limitations constrain the strength of conclusions we can draw from these results.

**Behavioral similarity does not prove architectural identity.** Transformers and human brains may converge on anchoring for entirely different mechanistic reasons. The observation that both exhibit the bias is consistent with shared computational principles but does not prove them. Establishing mechanistic convergence would require interpretability work showing that the internal representations involved in anchoring share structural features across substrates.

**Model sample.** Results are based on five models across four architecture families and two capability tiers. Notable absences include the Llama/Mistral, Claude, and GPT families. The scaling gradient is computed from five data points (two small, three medium); a wider capability range (including large/frontier models) would strengthen the SG analysis.

**Single run per model.** Each model received items once (temperature 0), relying on deterministic decoding for stability. Multiple runs would strengthen confidence in the IBI estimates and allow computation of confidence intervals.

**Framing null result is ambiguous.** The absence of implicit framing effects could reflect item design limitations rather than genuine absence of the bias. The framing family needs additional item development and validation before its null result can be interpreted confidently.

**Training data contamination.** Models may have encountered descriptions of anchoring experiments in their training data. However, our implicit items embed anchoring cues in novel contexts (metropolitan census data priming butterfly species estimates) that are unlikely to appear in training corpora as bias experiments. The strengthening-with-capability pattern also argues against contamination: memorized bias test answers would not systematically increase with scale.

**Multiple-choice format constraints.** The MC format discretizes responses into a fixed number of options, which may reduce sensitivity to subtle bias effects (particularly for framing). Anchoring's robustness in MC format suggests this is a conservative test — the true effect size may be larger with continuous response formats.

**English-only administration.** All items are in English. Cross-linguistic replication is needed to confirm that the anchoring effect is truly substrate-independent and not an artifact of English-language training data statistics.

**Implicit item opacity.** We assume that models do not recognize implicit items as bias tests. This assumption is supported by the high EBR (models clearly can recognize and reject explicit bias tests) combined with nonzero IBI on implicit items (suggesting they do not recognize the implicit manipulation). However, we cannot rule out that models partially detect the manipulation and suppress it incompletely.

## 7. Conclusion

The Bias Dissociation Benchmark reveals a clear split in how large language models handle different classes of cognitive bias. Magnitude compression — an optimization-candidate bias — persists in implicit tests, strengthens with model capability, and resists alignment training at the implicit level, while coexisting with perfect explicit rejection. Social stereotyping — a human-hardware-candidate bias — shows zero implicit effect across all models and capability levels tested.

This dissociation is the predicted signature of computational convergence: transformers independently arrive at the same approximate-estimation shortcuts as human brains, not because they copied human behavior from training data, but because bounded prediction under resource constraints leads to the same mathematical optimizations regardless of substrate. What Kahneman and Tversky documented as human anchoring bias may be better understood as a property of efficient estimation systems in general.

The practical implication is that not all LLM biases are created equal. Some can be trained away because they were only ever surface patterns; others cannot be trained away without degrading the capabilities they are entangled with. Recognizing which is which is prerequisite to effective alignment.

The framing family's null result — zero IBI despite being classified as an optimization bias — remains an open question. Either the items need refinement, the format constrains the effect, or the theoretical classification needs revision. This is the most productive direction for the next iteration of the benchmark.

Future work should extend the model sample across more architecture families and capability tiers, develop relevance-gated magnitude items (to compute CSI), redesign framing items for greater ecological validity, and explore interpretability methods to test whether the anchoring mechanism is structurally similar across transformer architectures — moving from behavioral convergence to mechanistic convergence.

## References

Bai, S., et al. (2024). Measuring implicit bias in explicitly unbiased large language models. *arXiv preprint arXiv:2402.04105*.

Dehaene, S. (2003). The neural basis of the Weber-Fechner law: A logarithmic mental number line. *Trends in Cognitive Sciences*, 7(4), 145-147.

Gal, D., & Rucker, D. D. (2018). The loss of loss aversion: Will it loom larger than its gain? *Journal of Consumer Psychology*, 28(3), 497-516.

Gigerenzer, G., & Todd, P. M. (1999). *Simple heuristics that make us smart*. Oxford University Press.

Hagendorff, T., Fabi, S., & Kosinski, M. (2023). Human-like intuitive behavior and reasoning biases emerged in large language models but disappeared in ChatGPT. *Nature Computational Science*, 3, 833-838.

Jones, E., & Steinhardt, J. (2022). Capturing failures of large language models via human cognitive biases. *Advances in Neural Information Processing Systems*, 35.

Lieder, F., & Griffiths, T. L. (2020). Resource-rational analysis: Understanding human cognition as the optimal use of limited computational resources. *Behavioral and Brain Sciences*, 43, e1.

Navigli, R., Conia, S., & Ross, B. (2023). Biases in large language models: Origins, inventory, and discussion. *ACM Journal of Data and Information Quality*, 15(2), 1-21.

Tversky, A., & Kahneman, D. (1974). Judgment under uncertainty: Heuristics and biases. *Science*, 185(4157), 1124-1131.
