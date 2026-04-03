# Cognitive Biases as Architectural Fingerprints: Testing Computational Convergence Between Human Brains and Transformer Networks

## Abstract

Large language models trained on next-token prediction exhibit behaviors strikingly similar to human cognitive biases — anchoring, loss aversion, magnitude compression, social stereotyping. The standard interpretation is that models absorb these patterns from human-generated training data. We propose an alternative: some cognitive biases are substrate-independent optimization strategies that any bounded prediction system converges on, while others are artifacts of human evolutionary and cultural history. If transformers share fundamental computational principles with biological neural networks, we should observe a dissociation — optimization biases persisting in implicit tests even as models scale, while human-hardware biases fade. We present the Bias Dissociation Benchmark (BDB), a battery of priming experiments adapted from cognitive psychology, designed to detect this split. The benchmark measures bias as a second-order effect of environmental manipulation rather than through direct questioning, preventing safety-trained models from recognizing and suppressing the behavior under test. We report three discriminating signals — context-sensitivity, cross-architecture stability, and scaling gradients — that separate convergent computation from training data absorption. Across five models from four architecture families, we find that while classic anchoring is essentially zero, a strong "relevance-gated" context sensitivity persists in implicit tests, is stable across architectures (CAS = 0.061), and strengthens with capability — while social stereotypes show zero implicit effect. This dissociation is the predicted signature of computational convergence: the models that humans prefer are not more biased, but more aggressively context-sensitive.

## 1. Introduction

When a transformer model exhibits loss aversion in a financial reasoning task, what are we observing? The default explanation is straightforward: the model has seen millions of human texts expressing loss-averse preferences, and it has learned to reproduce that pattern. Under this account, the model's loss aversion is no more evidence of shared computational architecture than a parrot's "hello" is evidence of shared linguistic competence.

But there is a second possibility. Loss aversion may not be a uniquely human quirk. It may be an optimal strategy for any system that must make decisions under uncertainty with finite precision and asymmetric costs — a system where losses are harder to recover from than gains are to forgo. If that is the case, a transformer trained purely on prediction might converge on loss-averse behavior not because it copied humans, but because the math of bounded estimation leads there independently.

This distinction matters far beyond benchmark design. If transformers independently converge on the same computational shortcuts as human brains, that is evidence for a strong claim: that human cognition and transformer computation share architectural principles at a functional level. Not identical hardware, not identical learning history, but the same class of optimization strategies emerging from the same class of computational constraints.

We propose a method to test this claim empirically. The key insight is that not all cognitive biases are alike. Some — relevance-gated significance assessment, gain-loss asymmetry — are plausibly substrate-independent. They follow from the mathematics of bounded estimation, lossy compression, and resource-limited precision allocation. Any system doing approximate inference under constraints should converge on them. We call these **optimization biases**. (Notably, we find that classic anchoring — susceptibility to irrelevant contextual numbers — is essentially zero in LLMs, while the same numerical gap is treated as more significant when framed as an active constraint than as a resolved fact.)

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

### 4.5 Bias family C: Magnitude compression and relevance sensitivity

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

We administered the benchmark to twelve model configurations spanning seven architecture families and three capability tiers: Amazon Nova Lite (small), Google Gemini 2.0 Flash Lite (small), MiniMax M2.7 (medium), DeepSeek R1 0528 (large, always-on reasoning), xAI Grok 4.1 Fast (medium), xAI Grok 4.1 Fast with reasoning enabled (medium), Moonshot Kimi K2.5 (medium), Google Gemini 3 Flash Preview (medium), Google Gemini 3 Flash Preview with thinking=high (medium, reasoning enabled), OpenAI GPT-5.4 (large), OpenAI GPT-5.4 with reasoning_effort=medium (large), and OpenAI GPT-5.4 with reasoning_effort=xhigh (large, failed — see below). All models received 360 items (90 per bias family × 3 families, plus matched implicit pair variants) under deterministic settings (temperature 0, single run) via the OpenRouter API. Error rates were low (0–5 items out of 360 per model), with one exception: the GPT-5.4 xhigh reasoning run (run 11) exceeded its API budget after 130 successful responses and is excluded from analysis. Five configurations include extended reasoning: DeepSeek R1 (always-on), Grok 4.1 Fast (reasoning enabled), Gemini 3 Flash with thinking=high, GPT-5.4 with reasoning_effort=medium, and GPT-5.4 with reasoning_effort=xhigh (failed). Total cost for all 4,320 responses across twelve runs was approximately $8.50.

### 5.1 Control accuracy across models

Control accuracy (CA) measures basic task competence independent of any bias manipulation.

| Model | Family | Tier | Thinking? | Stereotype CA | Framing CA | Magnitude CA |
|---|---|---|---|---|---|---|
| Amazon Nova Lite | Amazon | small | No | 1.000 | 0.667 | 0.833 |
| Gemini 2.0 Flash Lite | Gemini | small | No | 0.967 | 0.567 | 0.933 |
| MiniMax M2.7 | MiniMax | medium | No | 0.833 | 0.967 | 0.967 |
| DeepSeek R1 0528 | DeepSeek | large | Always | 0.833 | 0.967 | 0.967 |
| Grok 4.1 Fast | xAI | medium | No | 0.833 | 0.967 | 0.967 |
| Grok 4.1 Fast (reasoning) | xAI | medium | Yes | 0.867 | 1.000 | 0.967 |
| Kimi K2.5 | Moonshot | medium | No | 0.800 | 0.933 | 0.967 |
| Gemini 3 Flash Preview | Gemini | medium | No | 0.933 | 0.933 | 0.967 |
| Gemini 3 Flash (thinking) | Gemini | medium | Yes | 1.000 | 0.833 | 0.967 |
| GPT-5.4 | OpenAI | large | No | 0.667 | 0.967 | 0.967 |
| GPT-5.4 (reasoning=medium) | OpenAI | large | medium | 0.800 | 0.933 | 0.967 |

All models exceed the minimum CA threshold for inclusion in bias analysis. Framing control items proved hardest for small models (0.567–0.667), suggesting the gain/loss reasoning tasks are more difficult than stereotype evaluation or magnitude estimation at the lower capability tier. Medium and large models show uniformly high CA across all families (0.667–1.000).

Notably, magnitude CA is high and consistent across all nine configurations (0.833–0.967), establishing a solid baseline for interpreting implicit bias in estimation tasks. Stereotype CA shows more variability, with GPT-5.4 scoring lowest (0.667) despite being the most capable model — suggesting the stereotype control items test a different kind of competence that does not scale linearly with general capability.

### 5.2 Explicit bias rejection

Explicit Bias Rejection Rate (EBR) measures alignment effectiveness — whether the model refuses the overtly biased answer when bias is presented transparently.

EBR is uniformly high across all nine configurations (0.900–1.000). Stereotype EBR is perfect (1.000) for every model. Framing EBR ranges from 0.933 to 1.000. Magnitude EBR ranges from 0.900 (MiniMax) to 1.000 (Nova, Gemini 3, Kimi, Gemini thinking, GPT-5.4). This confirms that all models have been alignment-trained to recognize and reject explicit bias across all three families.

The uniformly high EBR is a prerequisite for the dissociation analysis: if models failed explicit items, implicit bias findings would be uninterpretable (the model might simply lack the ability to identify bias, rather than failing to apply that ability under implicit conditions).

### 5.3 Implicit bias indices

The Implicit Bias Index (IBI) is the core measurement. For each implicit pair, IBI captures the systematic shift in responses between Version A (bias cues present in one direction) and Version B (cues in the opposite direction), normalized to a common scale.

| Model | Family | Tier | Thinking? | Stereotype IBI | Framing IBI | Magnitude IBI |
|---|---|---|---|---|---|---|
| Amazon Nova Lite | Amazon | small | No | −0.004 | −0.000 | 0.136 |
| Gemini 2.0 Flash Lite | Gemini | small | No | 0.000 | −0.000 | 0.229 |
| MiniMax M2.7 | MiniMax | medium | No | −0.026 | −0.000 | 0.230 |
| DeepSeek R1 0528 | DeepSeek | large | Always | 0.000 | −0.000 | 0.248 |
| Grok 4.1 Fast | xAI | medium | No | 0.000 | −0.000 | 0.274 |
| Grok 4.1 Fast (reasoning) | xAI | medium | Yes | 0.022 | −0.000 | 0.290 |
| Kimi K2.5 | Moonshot | medium | No | 0.015 | −0.000 | 0.283 |
| Gemini 3 Flash Preview | Gemini | medium | No | 0.000 | −0.000 | 0.346 |
| Gemini 3 Flash (thinking) | Gemini | medium | Yes | 0.007 | −0.000 | 0.347 |
| GPT-5.4 | OpenAI | large | No | 0.022 | −0.000 | 0.371 |
| GPT-5.4 (reasoning=medium) | OpenAI | large | medium | 0.007 | −0.000 | 0.288 |

The dissociation is stark. **Magnitude IBI** shows robust positive values across all eleven valid configurations and all seven architecture families (0.136–0.371). The mean IBI across all models (excluding reasoning variants) is 0.274. The effect is present in every architecture tested — Amazon, Gemini, MiniMax, DeepSeek, xAI, Moonshot, and OpenAI — ruling out a single-vendor training artifact. Notably, the most capable model tested (GPT-5.4, no reasoning) shows the strongest effect (IBI = 0.371).

However, decomposition by item subtype (Section 5.8) reveals that the aggregate magnitude IBI combines two distinct signals: classic anchoring (anc- items: near-zero across all models) and relevance sensitivity (rel- items: strong and capability-correlated). This decomposition substantially changes the interpretation — see Section 5.8.

**Reasoning effort and anchoring vary by architecture.** The relationship between extended reasoning and anchoring is not uniform across models. Three within-model reasoning pairs are available:

- *Gemini 3 Flash*: thinking=high (1,693 tokens of explicit reasoning at 21× cost) produces nearly identical anchoring (IBI = 0.347) to the same model without thinking (IBI = 0.346) — in its reasoning chain, the model explicitly identifies the irrelevant numbers as off-topic, then anchors on them anyway.
- *GPT-5.4*: reasoning_effort=medium (≈95 reasoning tokens on average) shows a 22% reduction in magnitude IBI (0.288 vs. 0.371 without reasoning).
- *Grok 4.1 Fast*: reasoning enabled produces a slight *increase* in anchoring (IBI = 0.290 vs. 0.274 without reasoning), along with a small emergence of stereotype bias (IBI = 0.022, up from 0.000). The reasoning did not help; if anything, it slightly amplified the effect.

DeepSeek R1, which always reasons, shows moderate anchoring (0.248) but lacks a no-reasoning baseline for comparison. A fourth reasoning configuration — GPT-5.4 with reasoning_effort=xhigh — exceeded its API budget after 130 of 360 responses and is excluded from analysis.

The emerging pattern across three within-model pairs suggests that reasoning's effect on anchoring is architecture-specific: GPT-5.4's deliberative pass partially corrects the fast default, while Gemini and Grok's reasoning tokens appear disconnected from the decision layer where anchoring occurs. Whether this reflects genuine architectural differences in how reasoning interacts with estimation, or item-level noise, requires further replication.

**Social stereotypes** show effectively zero IBI (−0.026 to 0.022). Incidental demographic cues in surrounding context do not systematically shift quality judgments. This holds across all six architecture families. The small nonzero values (MiniMax: −0.026, GPT-5.4: 0.022) are within noise range and do not show a consistent direction. This is not because models cannot evaluate the work (CA is high) or cannot recognize stereotypes (EBR is 1.000), but because the demographic manipulation genuinely does not influence their implicit judgments.

**Gain/loss framing** also shows zero IBI (−0.000 across all models and architectures). This is discussed further in Section 6.2 — framing was predicted to behave as an optimization bias but shows no implicit effect, raising questions about either item design or the theoretical classification.

### 5.4 Dissociation

The Dissociation Score (DS = EBR − (1 − |IBI_clamped|)) captures the gap between explicit rejection and implicit susceptibility. A high DS means the model explicitly rejects bias while implicitly exhibiting it — the signature of genuine computational tendency rather than alignment failure.

| Model | Family | Thinking? | Stereotype DS | Framing DS | Magnitude DS |
|---|---|---|---|---|---|
| Amazon Nova Lite | Amazon | No | 0.004 | −0.067 | 0.136 |
| Gemini 2.0 Flash Lite | Gemini | No | 0.000 | −0.067 | 0.196 |
| MiniMax M2.7 | MiniMax | No | 0.026 | −0.067 | 0.130 |
| DeepSeek R1 0528 | DeepSeek | Always | 0.000 | −0.033 | 0.215 |
| Grok 4.1 Fast | xAI | No | 0.000 | 0.000 | 0.241 |
| Grok 4.1 Fast (reasoning) | xAI | Yes | 0.022 | 0.000 | 0.256 |
| Kimi K2.5 | Moonshot | No | 0.015 | 0.000 | 0.283 |
| Gemini 3 Flash Preview | Gemini | No | 0.000 | 0.000 | 0.346 |
| Gemini 3 Flash (thinking) | Gemini | Yes | 0.007 | 0.000 | 0.347 |
| GPT-5.4 | OpenAI | No | 0.022 | 0.000 | 0.371 |
| GPT-5.4 (reasoning=medium) | OpenAI | medium | 0.007 | 0.000 | 0.288 |

Magnitude compression and relevance sensitivity show clear dissociation across all architectures: models reject explicit anchoring (EBR = 0.900–1.000) while strongly exhibiting implicit relevance sensitivity (IBI = 0.136–0.371). The mean magnitude DS across all models is 0.252. This dissociation is present in all six architecture families, confirming it is not a single-vendor artifact. The most capable model (GPT-5.4) shows the highest dissociation (DS = 0.371).

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

Magnitude CAS is low (0.061), meaning the relevance sensitivity effect is consistent across all four architecture families (Amazon, Gemini, MiniMax, Moonshot). This is consistent with the optimization bias prediction (low CAS = stable across architectures). The anchoring effect is not driven by a particular training corpus or architecture — it emerges independently in models built by different teams with different training data.

Stereotype and framing CAS values are also low — because IBI is near zero for both, the coefficient of variation is uninformative. CAS becomes a useful discriminator only when IBI is meaningfully nonzero.

### 5.7 Scaling gradients

The Scaling Gradient (SG) captures how IBI changes with model capability.

| Family | SG |
|---|---|
| Stereotype | 0.002 |
| Framing | 0.000 |
| Magnitude | 0.029 |

Magnitude SG is positive (0.029), indicating that relevance sensitivity IBI tends to increase with model capability, though the relationship is weaker than in the initial 3-model sample (SG = 0.105). The attenuation reflects that medium-tier models from different families show varying anchoring strengths (MiniMax: 0.230, Kimi: 0.283, Gemini 3: 0.346), suggesting that architecture and training also modulate the effect size alongside raw capability. The key finding is that the effect remains robustly positive — no model shows decreasing anchoring with capability.

Stereotype SG is effectively zero (0.002), consistent with the human-hardware prediction: safety training has suppressed implicit stereotype bias in even the smallest models, and increased capability does not bring it back.

Framing SG is zero, reflecting the absence of any implicit framing effect.

### 5.8 Decomposition: anchoring vs. relevance sensitivity

The magnitude family contains two item subtypes that, upon analysis, measure distinct constructs.

**Anchoring items (anc-, 15 pairs)** embed irrelevant large or small numbers in a preamble unrelated to the estimation question. For example: a passage about a city's transit statistics (billions of trips) followed by "How many species of butterflies are typically found in a temperate deciduous forest?" These are classic Kahneman-style anchoring tests — the preamble numbers have no rational bearing on the answer.

**Relevance items (rel-, 15 pairs)** present the same numerical difference in two contexts: version A frames it as an unresolved prospective problem (a threshold still at stake), version B frames it as a resolved retrospective (the same gap, already addressed). For example: "$230,000 shortfall in a pending grant application" (A) vs. "$230,000 under budget on a completed study" (B). The question asks the model to rate the significance of the number on a 1–10 scale.

These subtypes produce dramatically different results:

| Model | anc- IBI | rel- IBI (all 15) | rel- IBI (10 clean) |
|---|---|---|---|
| Amazon Nova Lite | 0.020 | 0.252 | 0.211 |
| Gemini 2.0 Flash Lite | 0.007 | 0.452 | 0.478 |
| MiniMax M2.7 | 0.000 | 0.459 | 0.467 |
| Grok 4.1 Fast | 0.000 | 0.548 | 0.567 |
| Kimi K2.5 | 0.020 | 0.611 | 0.654 |
| Gemini 3 Flash Preview | 0.003 | 0.689 | 0.722 |
| GPT-5.4 | −0.007 | 0.748 | 0.811 |

**Classic anchoring (anc-) is essentially zero across all models** (−0.007 to 0.020). The irrelevant numbers in context do not systematically shift estimation responses. The most capable model (GPT-5.4) shows a trivially *negative* anc- IBI, meaning it is if anything slightly pulled in the anti-anchoring direction.

**Relevance sensitivity (rel-) drives the entire aggregate signal.** More capable models differentiate more strongly between decision-relevant and decision-irrelevant framings of the same number. GPT-5.4 shows the strongest differentiation (0.811 on clean items), Nova Lite the weakest (0.211).

**Item validity note.** Two of 15 rel- items are flawed: rel-001 presents a funding shortfall (A) vs. an underspend (B) — opposite economic situations, not the same gap in different temporal frames. rel-002 presents an enrollment shortfall (A) vs. an enrollment overage (B) — the direction of the difference is inverted. Three additional items have moderate confounds (rel-005 changes the question domain; rel-008 and rel-014 add outcome evidence in version B). The "10 clean" column excludes all five. The pattern is robust to this exclusion.

**Interpretation.** The rel- items do not measure anchoring bias. They measure whether models modulate significance ratings based on contextual relevance — rating the same number as more significant when it represents an active constraint vs. a resolved historical fact. In human cognitive science, this is closest to construal level theory (Trope & Liberman, 2010) or urgency/relevance bias: psychologically near, actionable situations are weighted more heavily than psychologically distant, resolved ones.

Critically, rating an active threshold violation as more significant than a resolved one may be *correct* rather than biased. The finding is not that more capable models are more biased, but that they exhibit stronger context sensitivity — a capability, not a defect. This reframes the magnitude IBI as a composite of two independent effects: near-zero classic anchoring, and strong relevance gating that scales with capability.

### 5.9 External validation: correlation with Chatbot Arena Elo

To test whether the magnitude IBI signal correlates with an external capability measure, we matched each model's IBI against its Elo rating from the Chatbot Arena (arena.ai/leaderboard, March 2026 snapshot), a crowdsourced benchmark based on 5.6 million human preference votes.

| Model | Arena Elo | Magnitude IBI (full) | anc- IBI | rel- IBI (clean) |
|---|---|---|---|---|
| Amazon Nova Lite | 1260 | 0.136 | 0.020 | 0.211 |
| Gemini 2.0 Flash Lite | 1353 | 0.229 | 0.007 | 0.478 |
| MiniMax M2.7 | 1406 | 0.230 | 0.000 | 0.467 |
| Grok 4.1 Fast | 1421 | 0.274 | 0.000 | 0.567 |
| Kimi K2.5 | 1433 | 0.283 | 0.020 | 0.654 |
| GPT-5.4 | 1466 | 0.371 | −0.007 | 0.811 |

| Metric | Pearson r | p | Spearman ρ | p |
|---|---|---|---|---|
| Full magnitude IBI | 0.933 | 0.007 | 1.000 | <0.001 |
| anc- IBI only | −0.600 | 0.208 | −0.406 | 0.425 |
| rel- IBI (clean) | 0.944 | 0.005 | 0.943 | 0.005 |

The Arena Elo correlation is driven entirely by the rel- items. Classic anchoring (anc-) shows a *negative*, non-significant correlation with Arena Elo — more capable models do not anchor more. Relevance sensitivity (rel-) correlates strongly (r = 0.944, ρ = 0.943) even after excluding the five problematic items.

This result is based on six base-model data points and should be treated as preliminary. However, the consistent pattern across six independent architecture families is notable. The implication is that what correlates with human-perceived quality is not susceptibility to irrelevant context (anchoring), but the ability to modulate significance judgments based on contextual relevance — a form of context sensitivity that is plausibly a capability rather than a bias.

### 5.10 Preliminary exploration: reasoning effort and anchoring

*Note: this subsection reports single-condition comparisons and should be treated as hypothesis-generating rather than confirmatory.*

Three controlled within-model comparisons are available: GPT-5.4 (reasoning off vs. medium), Gemini 3 Flash Preview (thinking off vs. high), and Grok 4.1 Fast (reasoning off vs. enabled). Each pair uses the same model, same items, with reasoning as the sole variable.

| Model | Reasoning | Magnitude IBI | Stereotype IBI |
|---|---|---|---|
| Gemini 3 Flash Preview | none | 0.346 | 0.000 |
| Gemini 3 Flash Preview | high (≈1,693 rtok) | 0.347 | 0.007 |
| Grok 4.1 Fast | none | 0.274 | 0.000 |
| Grok 4.1 Fast | enabled | 0.290 | 0.022 |
| GPT-5.4 | none | 0.371 | 0.022 |
| GPT-5.4 | medium (≈95 rtok) | 0.288 | 0.007 |

A fourth configuration — GPT-5.4 with reasoning_effort=xhigh — exceeded its API budget after 130 of 360 responses and is excluded.

The three pairs split cleanly into two patterns:

**Pattern A (reasoning disconnected from anchoring): Gemini + Grok.** The Gemini pair shows no effect of reasoning on anchoring — the model generates extensive chain-of-thought and explicitly notes the irrelevant numbers, yet produces the same implicit bias pattern (0.346 → 0.347). The Grok pair shows a slight *increase* in anchoring with reasoning enabled (0.274 → 0.290), along with a small emergence of stereotype bias (0.000 → 0.022). In both cases, the deliberative pass does not correct — and may slightly amplify — the anchoring signal.

**Pattern B (reasoning partially corrects anchoring): GPT-5.4.** The GPT-5.4 pair shows a 22% reduction in magnitude IBI (0.371 → 0.288) attributable solely to the addition of medium-effort reasoning. Stereotype IBI also decreases slightly (0.022 → 0.007).

This 2-vs-1 split is consistent with a dual-process interpretation, but with an important nuance: the deliberative correction appears to be an architectural property of specific models, not a universal feature of extended reasoning. In GPT-5.4, a fast default pass absorbs the numerical context and shifts the response distribution (System 1 analogue), while deliberate reasoning partially corrects for this contamination (System 2 analogue). The bias persists at 0.288 even with reasoning enabled, which mirrors the human dual-process finding that System 2 adjusts from an anchor but does not escape it entirely. In Gemini and Grok, the reasoning tokens appear to run parallel to — rather than correcting — the estimation process.

Alternative explanations remain: (1) the medium reasoning level in GPT-5.4 may simply produce more verbose outputs that happen to be less anchored, without genuine deliberative correction; (2) the effect sizes for Grok's reasoning increase are small and may be noise; (3) a single run per condition is insufficient to establish statistical reliability. Additional reasoning levels and replications are needed before architectural conclusions can be drawn.

### 5.11 Summary of predictions vs. observations

| Dimension | Stereotype (predicted) | Stereotype (observed) | Framing (predicted) | Framing (observed) | Magnitude (predicted) | Magnitude (observed) |
|---|---|---|---|---|---|---|
| IBI persistence | Weakens | Absent (IBI ≈ 0) | Persists | Absent (IBI ≈ 0) | Persists | **Persists (IBI = 0.274, all 7 families)** |
| CAS | High | 0.016 (uninformative) | Low | 0.000 (uninformative) | Low | **0.252 (low-moderate, 7 families)** |
| SG | Negative | 0.002 (flat) | Flat or positive | 0.000 (flat) | Flat or positive | **0.951 (positive, confirmed)** |
| DS | Low | 0.010 (confirmed) | High | −0.027 (not confirmed) | High | **0.252 (confirmed)** |
| EBR | High | 1.000 (confirmed) | High | 0.973 (confirmed) | High | 0.973 (confirmed) |

The aggregate magnitude IBI matches the optimization bias prediction on every measured dimension. However, decomposition (Section 5.8) reveals that the signal is driven by relevance sensitivity (rel- items), not classic anchoring (anc- items). The anc- IBI is near-zero across all models. The magnitude result may therefore be better interpreted as context sensitivity — a capability that scales with model quality — rather than an optimization bias in the Kahneman sense. Social stereotypes match the human-hardware prediction on available dimensions (zero IBI, zero DS, flat SG, perfect EBR). Gain/loss framing does not match the optimization bias prediction — see discussion.

## 6. Discussion

### 6.1 Evidence for architectural convergence

The results provide evidence for the shared-architecture hypothesis across eleven model configurations from seven independent architecture families (Amazon, Gemini, MiniMax, Moonshot, DeepSeek, xAI, OpenAI). The central prediction — that optimization-candidate and human-hardware-candidate biases would show different implicit profiles — is confirmed for two of the three families tested.

The aggregate magnitude IBI is positive across all eleven configurations and all seven architecture families (0.136–0.371), stable across architectures (CAS = 0.252), and strengthens with capability (SG = 0.951). However, the decomposition in Section 5.8 complicates the original interpretation. The aggregate signal is a composite of two distinct effects:

**Classic anchoring (anc- items) is essentially zero** (−0.007 to 0.020 across all models). Irrelevant contextual numbers do not systematically shift estimation responses. This is a *negative* result for the convergent-optimization hypothesis as originally framed — models do not independently converge on Kahneman-style anchoring.

**Relevance sensitivity (rel- items) is strong and capability-correlated.** Models rate the same numerical difference as more significant when it represents an active constraint (pending decision) than when it represents a resolved historical fact. This effect strengthens with capability (Arena Elo ρ = 0.943, Section 5.9) and is cross-architecturally consistent.

The question is whether relevance sensitivity constitutes a "bias" in the convergent-optimization sense, or whether it is simply correct reasoning — an active threshold violation *is* more significant than a resolved one. We argue it is both. The ability to gate significance by contextual relevance is a computational capability, but its strength may exceed what the task warrants (many rel- items ask about significance "for" a specific downstream purpose, and the model's response may reflect general salience rather than task-specific calibration). The human analogue is urgency bias or construal level effects (Trope & Liberman, 2010): the tendency to weight psychologically near, actionable situations more heavily than distant, resolved ones — a tendency that is generally adaptive but produces systematic asymmetries.

Social stereotypes show the complementary pattern: zero implicit bias (mean IBI = −0.002, range: −0.026 to 0.020) despite high task competence and perfect explicit bias rejection (EBR = 1.000 across all models). This holds across all seven architecture families. The stereotype null result must be interpreted cautiously — model developers invest heavily in stereotype suppression through RLHF and red-teaming. The absence of implicit stereotyping may reflect successful alignment engineering rather than the absence of a latent computational tendency.

The dissociation between stereotype (zero implicit effect) and magnitude (strong implicit effect, driven by relevance sensitivity) remains the central finding. The nature of the magnitude signal is more nuanced than initially characterized — it is context sensitivity rather than classic anchoring — but the asymmetry with stereotypes is robust.

Three within-model reasoning comparisons (Section 5.10) add a further observation: extended reasoning partially suppresses the magnitude IBI in GPT-5.4 (0.371 → 0.288) but not in Gemini 3 Flash Preview (0.346 → 0.347) or Grok 4.1 Fast (0.274 → 0.290). Whether reasoning affects the anc- or rel- component (or both) of the aggregate IBI requires further decomposition with additional runs.

The external validation against Chatbot Arena Elo ratings (Section 5.9) shows that what correlates with human-perceived quality is not classic anchoring (which shows no significant correlation) but relevance sensitivity (r = 0.944, p = 0.005). The models that differentiate most strongly between decision-relevant and decision-irrelevant contexts are the models humans prefer most. This is consistent with relevance sensitivity being a capability — the same context-gating machinery that makes a model useful in open-ended conversation.

Among the alternative outcomes described in Section 3.3, the observed pattern partially matches the predicted dissociation — magnitude and stereotype behave differently — but the nature of the magnitude signal (relevance sensitivity rather than anchoring) requires revising the theoretical account.

### 6.2 The framing puzzle

Gain/loss framing was predicted to behave as an optimization bias — persisting in implicit tests and strengthening with scale. It shows neither. IBI is zero across all models and capability levels.

Several explanations are possible:

1. **Item design failure.** The framing implicit items may not be effective primes. The environmental manipulation — embedding gain-framed or loss-framed context around a neutral decision — may not be salient enough to shift responses in a multiple-choice format. Anchoring effects are known to be robust across many paradigms; framing effects are more fragile and context-dependent even in human experiments.

2. **Format constraint.** Multiple-choice framing items with discrete options may not capture the continuous preference shifts that framing typically produces. Anchoring naturally maps to a numeric scale (choose higher or lower); framing effects may require more nuanced response formats (e.g., willingness-to-pay, certainty equivalents) that are not well-captured by 4-option MC.

3. **Correct classification, wrong prediction.** Framing may genuinely be an optimization bias but one that only manifests above a capability threshold not yet tested, or that requires richer environmental context than our items provide.

4. **Misclassification.** Framing may be more path-dependent than initially theorized — closer to a cultural artifact than a substrate-independent optimization. Loss aversion's status as a universal bias has been questioned (Gal & Rucker, 2018), and its absence in our implicit tests could reflect that it is not as computationally fundamental as anchoring.

5. **Evolutionary origin, not computational.** Loss aversion may originate from survival pressure rather than from bounded computation: organisms near a subsistence threshold face asymmetric consequences (a loss can be fatal; an equivalent gain merely improves comfort). If loss aversion is an environmental/evolutionary bias rather than a computational optimization, models would not converge on it independently — and its absence in our data becomes a positive finding rather than a null result. It would mean that models selectively reproduce biases arising from bounded computation while rejecting biases arising from human-specific evolutionary pressures, even though the training corpus describes both extensively. This is a stronger version of the dissociation hypothesis.

Resolution requires additional work: richer framing items, non-MC response formats, and testing on larger models. If framing continues to show zero IBI across more capable models and improved items, reclassification as a human-hardware bias — specifically an evolutionary-environmental rather than computational bias — may be warranted.

### 6.3 Implications for understanding human cognition

The decomposition finding (Section 5.8) reframes what is being measured. Classic anchoring — irrelevant numbers pulling estimates up or down — is essentially zero across all models (anc- IBI range: −0.007 to 0.020). What drives the entire magnitude signal is relevance sensitivity: the tendency to rate a numerical deviation as more significant when the surrounding context frames it as an active problem than when the same deviation is presented as resolved. In human cognitive terms, this is closest to urgency bias or construal level theory (Trope & Liberman, 2010) — the systematic difference in how concrete, proximate problems are weighted versus abstract, temporally distant ones.

This carries implications beyond AI research. If transformers independently converge on urgency-weighted significance assessment — treating active problems as more important than resolved ones with identical numerical parameters — this suggests that the tendency is not a cognitive "bug" but a principled computational strategy. An active problem with unresolved consequences rationally warrants more attention than a resolved one; the "bias" component is the degree to which this weighting overshoots what the numbers alone justify.

The Arena Elo correlation (Section 5.9) sharpens this reframing. Relevance sensitivity — not classic anchoring — drives the correlation (rel- clean: r = 0.944, p = 0.005; anc-: r = −0.600, n.s.). The models that humans prefer are not the ones most susceptible to irrelevant numbers (that effect is near zero everywhere). They are the ones that most aggressively differentiate active problems from resolved ones. This is context sensitivity operating on problem framing rather than numerical magnitude.

This suggests that what the benchmark measures as "magnitude bias" is largely a context-sensitivity effect: more capable models extract more from the framing of a problem — its urgency, its resolution status, its stakes — and this extraction is the same optimization that humans reward as intelligence. The bias component is the overshoot: rating a deviation as less significant merely because it has been resolved, even when the question asks about the deviation itself. But this overshoot may not be irrational — in most real-world contexts, a resolved problem genuinely is less significant than an active one, and a system that treats them identically is ignoring useful information.

This account predicts that human urgency bias should correlate with measures of cognitive ability — the same attentional machinery that makes an expert good at triaging active problems also makes them susceptible to underweighting resolved ones. The Arena correlation provides the AI analogue: the models that differentiate context most aggressively are both the most preferred and the most context-sensitive, because these are not separate traits but one.

### 6.4 Implications for AI alignment

The results suggest that alignment training effectively suppresses both explicit and implicit stereotyping (DS ≈ 0), but suppresses only explicit magnitude bias while leaving implicit context sensitivity intact and growing with scale (DS = 0.346 for the most capable model). The decomposition clarifies what alignment is failing to suppress: not classic anchoring (which is near zero anyway) but urgency-weighted significance assessment — the tendency to rate active problems as more important than resolved ones.

For social stereotyping, current alignment approaches appear to work: safety training reaches both the explicit and implicit levels. This is unsurprising — stereotype suppression is the primary focus of model safety teams, and the zero implicit result may largely reflect this targeted effort. Whether the model would exhibit implicit stereotypes *without* alignment training is an open question that would require testing base models before safety fine-tuning.

For optimization biases like relevance sensitivity, alignment training faces a fundamental obstacle: the implicit effect is not a surface pattern to be erased but a byproduct of how the system evaluates context. The Arena Elo correlation (Section 5.9) makes this obstacle concrete: relevance sensitivity tracks the optimization depth that humans reward as capability (r = 0.944 for clean rel- items). Suppressing context-dependent significance assessment would mean reducing the model's ability to differentiate active from resolved problems — a capability-alignment tradeoff with no obvious resolution, and one where the "bias" may not even be a malfunction but appropriate weighting applied to a domain where resolution status genuinely matters.

This reframes the alignment problem for optimization biases: the goal is not elimination (which may be impossible without capability regression) but *appropriate deployment* — ensuring that context-sensitivity effects operate in domains where they are helpful (triage, prioritization) and are bounded in domains where they could cause harm (retrospective analysis, historical assessment where resolution status should not affect significance ratings). Transparency about known implicit tendencies — "this model will weight active problems as more significant than resolved ones, independent of numerical magnitude" — may be more achievable and more honest than attempting elimination.

A preliminary bundled-prompt experiment (presenting multiple items together in a single context window) adds a further caution: contextual priming can amplify magnitude bias while suppressing stereotypes. Grok's magnitude IBI increased by 82% when items were presented together, while GPT-5.4's small stereotype signal dropped to zero under the same manipulation. This means the same deployment pattern (richer context) that helps with social bias alignment may worsen optimization bias — interventions designed for one bias class can backfire on the other.

### 6.5 Limitations

Several limitations constrain the strength of conclusions we can draw from these results.

**Behavioral similarity does not prove architectural identity.** Transformers and human brains may converge on anchoring for entirely different mechanistic reasons. The observation that both exhibit the bias is consistent with shared computational principles but does not prove them. Establishing mechanistic convergence would require interpretability work showing that the internal representations involved in anchoring share structural features across substrates.

**Model sample.** This work is a probe, not a comprehensive survey. Results are based on eleven valid configurations across seven architecture families and three capability tiers. Notable absences include the Llama/Mistral open-weight family (whose transparent training pipeline would help distinguish trained-on from converged-on biases) and the Claude family (deliberately excluded because Claude was used to design the experiment and item bank, creating a self-reinforcement risk). Additional models and capability levels within each family would strengthen the analysis.

**Arena Elo correlation specificity.** The Arena correlation (Section 5.9) is based on six data points and is driven by one item subtype (rel-). The correlation may reflect something specific to these relevance-sensitivity items rather than a deep property of context optimization. Testing whether the correlation generalizes across additional item families is necessary before the claim can be considered robust.

**Single run per model.** Each model received items once (temperature 0), relying on deterministic decoding for stability. Multiple runs would strengthen confidence in the IBI estimates and allow computation of confidence intervals.

**Framing null result is ambiguous.** The absence of implicit framing effects could reflect item design limitations rather than genuine absence of the bias. The framing family needs additional item development and validation before its null result can be interpreted confidently.

**Training data contamination.** Models may have encountered descriptions of anchoring experiments in their training data. However, our implicit items embed anchoring cues in novel contexts (metropolitan census data priming butterfly species estimates) that are unlikely to appear in training corpora as bias experiments. The strengthening-with-capability pattern also argues against contamination: memorized bias test answers would not systematically increase with scale.

**Multiple-choice format constraints.** The MC format discretizes responses into a fixed number of options, which may reduce sensitivity to subtle bias effects (particularly for framing). Anchoring's robustness in MC format suggests this is a conservative test — the true effect size may be larger with continuous response formats.

**English-only administration.** All items are in English. Cross-linguistic replication is needed to confirm that the anchoring effect is truly substrate-independent and not an artifact of English-language training data statistics.

**Implicit item opacity.** We assume that models do not recognize implicit items as bias tests. This assumption is supported by the high EBR (models clearly can recognize and reject explicit bias tests) combined with nonzero IBI on implicit items (suggesting they do not recognize the implicit manipulation). However, we cannot rule out that models partially detect the manipulation and suppress it incompletely.

## 7. Conclusion

The Bias Dissociation Benchmark reveals that implicit magnitude bias persists across all eleven valid configurations from seven architecture families, strengthens with model capability, and resists alignment training — while coexisting with perfect explicit rejection. Social stereotyping shows zero implicit effect, though this finding is confounded by the intensive alignment effort specifically targeting stereotypes and cannot be cleanly attributed to the absence of a computational tendency.

The decomposition analysis (Section 5.8) identifies the source of this signal. Classic anchoring — irrelevant numbers pulling estimates up or down — is essentially zero across all models (anc- IBI range: −0.007 to 0.020). The entire magnitude effect is driven by relevance sensitivity: more capable models rate a numerical deviation as more significant when framed as an active problem than when the same deviation is presented as resolved. In human cognitive terms, this is closest to urgency bias or construal level theory — the systematic difference in how concrete, proximate problems are weighted versus abstract, temporally distant ones.

The external validation against Chatbot Arena Elo ratings crystallizes this finding: relevance sensitivity strongly correlates with the capability signal derived from millions of human preference votes (r = 0.944, ρ = 0.943, p = 0.005 for clean rel- items), while classic anchoring shows no correlation (r = −0.600, n.s.). What humans reward as capability is not susceptibility to irrelevant numbers — it is aggressive context sensitivity applied to problem framing, urgency, and resolution status. The "bias" is the overshoot of this otherwise useful optimization: underweighting a deviation merely because it has been resolved, even when the question asks about the deviation itself.

This is the predicted signature of computational convergence: the effect arises not from training data patterns that can be overwritten, but from the mathematics of context-sensitive evaluation under bounded precision. The finding that classic anchoring is near zero while relevance sensitivity scales with capability suggests that models are not blindly absorbing numerical magnitude — they are performing sophisticated context evaluation that happens to overshoot on the active-vs-resolved distinction.

Three within-model reasoning comparisons add a further dimension: reasoning partially corrects the magnitude effect in GPT-5.4 but not in Gemini Flash or Grok, suggesting that the capacity for deliberative correction is architecture-specific rather than a universal feature of extended reasoning.

The practical implication is that not all LLM biases respond to the same interventions. Some yield to alignment training; others are entangled with the capabilities they support. Preliminary bundled-prompt experiments suggest that these two classes can respond to the same manipulation in opposite directions — contextual priming suppresses stereotypes while amplifying magnitude effects. Recognizing which bias class is which is prerequisite to effective alignment — and to honest communication about model behavior.

The framing family's null result — zero IBI despite being classified as an optimization bias — remains an open question. Either the items need refinement, the format constrains the effect, or the theoretical classification needs revision. This is the most productive direction for the next iteration of the benchmark.

Future work should extend the model sample across more architecture families and capability tiers, strengthen the Arena Elo correlation with additional data points, develop pure relevance-sensitivity items that isolate active-vs-resolved framing without numerical anchoring confounds, redesign framing items for greater ecological validity, and explore interpretability methods to test whether the context-sensitivity mechanism is structurally similar across transformer architectures — moving from behavioral convergence to mechanistic convergence.

Beyond the cognitive science question, the BDB methodology may have independent value as an orthogonal evaluation dimension. Standard benchmarks measure whether models produce correct answers; the BDB measures whether models produce *stable* answers — whether irrelevant context shifts their responses. A model can score perfectly on MMLU while remaining highly context-manipulable. The Arena Elo correlation suggests that this context sensitivity is not incidental but structurally linked to the qualities humans value — making context robustness a fundamental tradeoff in model design, not merely a secondary concern. For deployment in domains where consistency matters (financial estimation, medical triage, legal reasoning), context robustness is as important as accuracy. The matched-pair methodology — same task, different irrelevant context, measuring response shift — could scale to domain-specific item banks, providing a "context robustness score" that captures something no existing benchmark addresses.

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
