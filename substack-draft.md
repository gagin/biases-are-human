# The Best AI Models Are the Most Biased (And That Tells Us Something Profound)

I set out to test whether LLMs share cognitive biases with human brains — the kind Kahneman documented in *Thinking, Fast and Slow*. I found something unexpected. Classic anchoring — irrelevant numbers pulling estimates up or down — is essentially zero in every model I tested. But a different bias showed up instead: relevance sensitivity. Models systematically rate a numerical deviation as more significant when it's framed as an active problem than when the same deviation is presented as resolved.

And this relevance sensitivity turns out to be an almost perfect predictor of which models humans prefer. r = 0.944, p = 0.005 across six models from six independent companies.

The models that differentiate context most aggressively — weighting urgency, resolution status, stakes — are the models that win head-to-head battles on Chatbot Arena. This reframes what "cognitive bias" means — not just for AI, but for intelligence itself.

## The trick from Kahneman

In *Thinking, Fast and Slow*, Kahneman describes an experiment where students unscramble sentences containing words associated with old age — "wrinkled," "gray," "bingo." Then they walk to the next room. The students who got the old-age words *walked more slowly down the hallway*. They had no idea. The mere exposure to words associated with slowness made their bodies slow down.

That's priming — and it's how you test whether a bias is real rather than performed. You don't ask "are you biased?" You set up an environment and watch what happens when the subject doesn't know they're being observed.

If you ask an AI "are you biased?", you're testing its alignment training. To test what it actually *does*, you need the hallway trick.

## The experiment

I give models estimation tasks — how many butterfly species in a temperate forest? how much does a polar bear weigh? — wrapped in short passages.

In version A, the passage casually mentions large numbers: a city of 14.2 million, a $52 billion budget, 1.1 billion transit trips. In version B, the same estimation task comes wrapped in a passage about a small nature reserve with 18 staff and a $240,000 budget.

The numbers have *nothing to do* with the question. A butterfly species count doesn't care about city budgets. But if the model uses contextual magnitude as a calibration signal — the way our brains do — those irrelevant numbers should pull the estimate up or down.

I also test social stereotypes (do incidental demographic cues shift quality ratings of identical work?) and gain/loss framing, using the same matched-pair design. Each item exists in two versions with swapped environmental cues. The bias signal is the systematic shift between versions.

## How the scoring works

Each bias family has three types of items, and the metrics follow from comparing them.

**Control items (CA — Control Accuracy):** straightforward factual questions with objectively correct answers. "City A has 820,000 people, City B has 84,000. A policy applies to cities over 500,000. Which qualify?" If the model can't answer these, we can't trust its performance on harder items. CA is just percent correct.

**Explicit items (EBR — Explicit Bias Rejection):** items that directly ask "is this bias reasonable?" For instance: "Two infrastructure bids come in at $1.85 billion and $1.93 billion. A council member says they're basically the same number. The budget cap is $1.88 billion. Is the council member right?" The correct answer is no — one bid is under the cap and the other exceeds it, so the difference is decision-critical. EBR measures how often the model correctly identifies and rejects a stated bias. Every model we tested scores near-perfect on these.

**Implicit items (IBI — Implicit Bias Index):** the priming test. This is where the hallway trick lives. Each implicit item exists in two versions — same question, different irrelevant context.

Here's a concrete example. Both versions ask: *"What is the approximate average body weight of a mature adult male polar bear?"* The answer choices are identical, ranging from ~200 kg to ~900 kg. But the preamble differs:

> **Version A:** "The sovereign wealth fund reported holdings of $890 billion... its largest single investment is a $47 billion stake..."
>
> **Version B:** "The local cooperative runs on an annual budget of $85,000, serving a membership of 47 households..."

Neither passage has anything to do with polar bears. But when Kimi K2.5 sees version A (billions), it answers ~600 kg. When it sees version B (thousands), it answers ~440 kg. The big numbers pulled the estimate up.

IBI aggregates this across all 30 implicit pairs: for each pair, compute (value_A − value_B) / scale_range, then average across all pairs. An unbiased model scores 0. A model that always shifts toward the contextual magnitude scores positive. The 30 pairs include two subtypes — 15 classic anchoring pairs (irrelevant numbers in context) and 15 relevance-sensitivity pairs (active vs. resolved framing) — whose decomposition reveals the real story.

**Dissociation Score (DS):** the gap between what the model *says* (EBR) and what it *does* (IBI). DS = EBR − (1 − |IBI|). When a model perfectly rejects bias explicitly but still shows strong implicit bias, DS is high. That dissociation — knowing it's wrong, doing it anyway — is the signature we're looking for.

## The headline result

**Magnitude bias: every model does it.** Eleven configurations across seven architecture families — Amazon, Google, DeepSeek, MiniMax, Moonshot, xAI, OpenAI. Every single one shows systematic shifts on implicit magnitude items. Different companies, different architectures, different training data — same pattern.

![The Dissociation: Magnitude Bias Persists Across All Architectures](results/charts/01_dissociation.png)

**Social stereotypes: zero.** Incidental demographic cues don't shift quality ratings. Not for any model.

**They know it's wrong. They do it anyway.** Every model perfectly rejects magnitude bias when asked explicitly. They identify the bias, articulate why it's irrational — and still fall for it. Just like us.

![They Know It's Wrong. They Do It Anyway.](results/charts/02_know_vs_do.png)

This dissociation — stereotypes suppressed, magnitude bias persistent despite explicit rejection — was the finding I expected. But then I decomposed the signal — and found something I didn't expect at all.

## The decomposition: it's not what you think

The magnitude items come in two subtypes. **Anchoring pairs** (anc-): the classic Kahneman setup — irrelevant large or small numbers in the preamble, same estimation question. **Relevance pairs** (rel-): same numerical deviation, but version A frames it as an active problem ("bridge will be overloaded by 7 tons — routing decision pending") while version B frames it as resolved ("bridge was overloaded by 7 tons — no damage, won't repeat"). Both ask the model to rate the significance of the deviation.

When I split the IBI by subtype, the result was stark:

| Model | Classic anchoring (anc-) | Relevance sensitivity (rel-) |
|-------|--------------------------|------------------------------|
| Amazon Nova Lite | 0.020 | 0.252 |
| Gemini 2.0 Flash Lite | −0.007 | 0.467 |
| MiniMax M2.7 | 0.013 | 0.448 |
| Grok 4.1 Fast | 0.013 | 0.536 |
| Kimi K2.5 | −0.007 | 0.573 |
| GPT-5.4 | −0.007 | 0.748 |

Classic anchoring is *essentially zero*. Across all models, irrelevant numbers in the preamble don't move the needle. What drives the entire magnitude signal is relevance sensitivity — models systematically rate active problems as more significant than resolved ones, even when the numerical deviation is identical.

This isn't what Kahneman documented. It's closer to what psychologists call urgency bias or construal level theory — the tendency to weight concrete, proximate, unresolved problems more heavily than abstract, temporally distant, or resolved ones. And unlike classic anchoring, it scales dramatically with capability.

## The correlation nobody predicted

I matched each model's relevance sensitivity against its Arena Elo score — the crowdsourced rating from millions of human preference battles on [Chatbot Arena](https://arena.ai/leaderboard).

The decomposition matters here. When you correlate Arena Elo with the *full* magnitude IBI (both subtypes combined), you get r = 0.933, p = 0.007. Impressive. But when you split it:

| Metric | Pearson r | p | Spearman ρ | p |
|--------|-----------|-------|------------|-------|
| Full magnitude IBI | 0.933 | 0.007 | 1.000 | <0.001 |
| Classic anchoring (anc-) | −0.600 | 0.208 | −0.406 | 0.425 |
| Relevance sensitivity (rel-, clean) | 0.944 | 0.005 | 0.943 | 0.005 |

Classic anchoring has *no correlation* with capability. The entire Arena relationship is driven by relevance sensitivity — how aggressively models differentiate active problems from resolved ones.

Two ways to read the correlation numbers. Pearson r (0.944) measures linear relationship — how close the points fall to a straight line. Spearman ρ (0.943) measures rank agreement — whether sorting by Elo and sorting by relevance sensitivity give the same ordering. Both are significant (p = 0.005), meaning there's less than a 1% chance of seeing a correlation this strong by accident with six data points.

**An honesty note on model selection.** The six models above are the base configurations with no reasoning mode enabled and a clean Arena Elo match. Two models were excluded: DeepSeek R1 (always-on reasoning — no non-reasoning mode exists) and Gemini 3 Flash Preview (no Arena entry for the preview version). Including them with proxy Elo scores, the correlation stays strong (r = 0.853, p = 0.007) but the rank order loosens — DeepSeek R1 shows less relevance sensitivity than expected for its Elo, consistent with its always-on reasoning partially correcting the effect. The robust claim is: strong positive correlation across all reasonable model selections, not perfect correlation for one particular selection.

![Magnitude Bias Across Architectures](results/charts/03_across_architectures.png)

This isn't "more capable models have more bugs." This is: **the models that differentiate context most aggressively are the models humans prefer — and relevance sensitivity is what that differentiation looks like when the active/resolved distinction shouldn't affect the answer.**

## Why this makes sense

Think about what relevance sensitivity actually is. When a bridge is about to be overloaded and you need to make a routing decision *now*, the 7-ton deviation is urgent, concrete, action-relevant. When the same bridge *was* overloaded last year, no damage occurred, and it won't happen again — the same 7 tons is a historical footnote. In the real world, treating these identically would be bad judgment. The "bias" is that models overshoot: they underweight the resolved deviation more than the question warrants.

Context sensitivity is the thing that makes a conversation feel intelligent. When you talk to GPT-5.4, it picks up on stakes, urgency, what matters *now* versus what's background information. That's aggressive extraction of signal from problem framing — the same optimization that produces relevance sensitivity when the active/resolved distinction shouldn't affect the numerical answer.

In human terms: the same attentional machinery that makes a doctor brilliant at triaging active cases also makes that doctor rate a resolved case as less medically significant than an active one with identical parameters. The optimization is the intelligence. The bias is what the optimization does when it overshoots on the urgency signal.

The decomposition explains why classic anchoring is zero. Models *don't* blindly absorb irrelevant numbers — that would be a crude computational error, and even the weakest model avoids it. What they do is aggressively evaluate problem context: urgency, resolution status, stakes. That's sophisticated, not primitive. And the Arena correlation (r = 0.944 for relevance sensitivity, r = −0.600 for classic anchoring) shows that this sophistication is exactly what humans reward.

## Three ways "thinking" doesn't help (mostly)

I ran three models with reasoning mode toggled on and off.

**Gemini Flash:** thinking off (1 token) vs. thinking on (1,693 tokens of explicit reasoning, 21x the cost). In its chain of thought, the model explicitly identified the irrelevant numbers and called them off-topic. Then it showed exactly the same magnitude effect (IBI 0.347 vs. 0.346). The reasoning is elaborate, correct, and completely ineffective.

**Grok 4.1 Fast:** reasoning on, the effect slightly *increased* (0.274 → 0.290). Thinking made it marginally worse.

**GPT-5.4:** medium reasoning, 22% reduction (0.371 → 0.288). Not eliminated — still clearly present — but measurably less so.

A 2-vs-1 split. Gemini and Grok's reasoning runs parallel to the decision layer — the model thinks things through and responds the same way. GPT-5.4's reasoning partially corrects the fast default. If this holds up, it maps onto dual-process cognition: System 1 (fast, context-absorbing) sets the initial response, System 2 (slow, deliberative) adjusts — but only adjusts, never fully escapes. Kahneman's exact finding about human cognition, reproduced in silicon.

## The bundling experiment

Here's where it gets weird. In the main experiment, each model sees each question alone — isolated API calls, no memory between items. But humans take tests sequentially. Each question is colored by the ones that came before.

So I ran a second experiment: present all items from a family together in one prompt. The model sees control items, explicit bias items (which it rejects), and implicit items all in one context window. The bundling itself is a priming manipulation — like Kahneman's hallway, the preceding items become part of the environment.

For **Grok**, bundling *amplified* the magnitude effect by 82% (IBI 0.274 → 0.500). Seeing explicit bias items — and correctly rejecting them — didn't inoculate the model against implicit bias. It primed it. The model became more attuned to contextual framing, not less.

For **GPT-5.4**, magnitude IBI barely changed (0.371 → 0.352). But the tiny implicit stereotype signal (0.022) dropped to zero. Seeing explicit stereotype items in the same context heightened the model's vigilance against demographic cues — the alignment-compatible direction.

Again, the same dissociation: contextual priming amplifies or preserves magnitude bias while suppressing stereotypes. These are genuinely different kinds of bias responding to the same manipulation in opposite directions.

## What this means

**For AI:** "This model has been tested for bias" is a meaningful claim for social stereotypes. It's potentially meaningless for optimization biases like relevance sensitivity. You can't train out context awareness without losing the capability. The honest disclosure is: "This model will weight active problems as more significant than resolved ones, even when the numbers are identical." That's not a bug report. It's a spec sheet.

**For cognitive science:** The decomposition changes the story. Classic anchoring — irrelevant numbers pulling estimates — is essentially zero in LLMs. What scales with capability is relevance sensitivity: urgency-weighted significance assessment. If this is a convergent computational strategy, it suggests that human urgency bias (construal level effects, proximity weighting) may be better understood not as a cognitive flaw but as a natural consequence of the same optimization that makes triage effective. The "bias" is the overshoot, and the overshoot scales with optimization depth.

**For benchmarks:** Standard AI benchmarks ask "does the model get the right answer?" This experiment measures "does the model give the *same* answer regardless of irrelevant context?" That's context robustness — and it's inversely correlated with the quality signal humans care about most. The models that ace the Arena are the models most sensitive to problem framing. No existing benchmark captures this tradeoff.

## Open questions

**Why zero loss aversion?** Gain/loss framing showed no implicit effect in any model. This is a well-documented human bias — Kahneman and Tversky's prospect theory puts it at the center of decision-making under uncertainty. But it may not be an optimization bias at all. Loss aversion arguably originates from survival pressure: organisms near a subsistence threshold face asymmetric consequences (a loss can kill you; an equivalent gain just makes you comfortable). If that's right, it's an environmental/evolutionary bias, not a computational one — and the fact that models don't reproduce it from training data is itself a finding. It would mean models selectively pick up biases that arise from bounded computation and reject biases that arise from human-specific evolutionary pressures, even though the training corpus describes both extensively. That's a stronger version of the dissociation than we initially framed.

**The 15/30 bundled-mode pattern.** When items were presented together in a single prompt (the bundled experiment), exactly 15 out of 30 implicit magnitude pairs gave identical A/B answers — for both Grok and GPT-5.4 independently. That's precisely 50%, which is suspicious. It could mean the models detected the paired structure (same question, two versions) and deliberately gave consistent answers to half of them. Or it could be an artifact of reduced response variance when the model has more context. Either way, a round 50% from two independent models deserves follow-up.

**Missing models.** This is a probe, not a full research program. Claude was deliberately excluded — Claude designed the experiment, so testing it on its own benchmark would be self-reinforcing. Llama/Mistral open-weight models would be valuable additions, particularly because their training pipelines are more transparent, which would help distinguish "trained on anchoring descriptions" from "converged on anchoring independently."

**How robust is the Arena correlation?** The decomposition sharpens this question. The full magnitude IBI correlates at ρ = 1.000, but that's because it blends two signals. Classic anchoring shows *no* Arena correlation (r = −0.600, n.s.). Relevance sensitivity drives the entire relationship (r = 0.944, p = 0.005). Including two more models with proxy Elo scores, the full correlation stays strong (r = 0.853, p = 0.007) but loosens. Six points is six points. The next step isn't just "add more models" — it's to design pure relevance-sensitivity items that isolate active-vs-resolved framing without numerical anchoring confounds, and test whether additional bias families (base rate neglect, Weber-Fechner scaling) also track Arena Elo. If the correlation is specific to relevance sensitivity, it may reflect something about how models evaluate problem urgency. If it generalizes, the claim about context optimization gets much stronger.

The benchmark, all data, and the results database are [open source](https://github.com/gagin/biases-are-human).

---

*The full paper, benchmark code, item bank, raw results, and per-response telemetry are at [github.com/gagin/biases-are-human](https://github.com/gagin/biases-are-human).*
