# The Best AI Models Are the Most Biased (And That Tells Us Something Profound)

I set out to test whether LLMs share cognitive biases with human brains — the kind Kahneman documented in *Thinking, Fast and Slow*. I found that, yes, they do. But I also stumbled into something I didn't expect: the bias I measured turns out to be an almost perfect predictor of which models humans prefer.

The models that anchor hardest on irrelevant numbers are, in exact rank order, the models that win the most head-to-head battles on Chatbot Arena. Spearman ρ = 1.000. Not a typo. Perfect rank correlation across six models from six independent companies.

This reframes what "cognitive bias" means — not just for AI, but for intelligence itself.

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

IBI aggregates this across all 30 implicit pairs: for each pair, does the version-A answer (big-number context) exceed the version-B answer (small-number context)? IBI is the proportion of pairs where A > B, minus the proportion where B > A, normalized to [−1, +1]. An unbiased model scores 0. A model that always shifts toward the contextual magnitude scores 1.

**Dissociation Score (DS):** the gap between what the model *says* (EBR) and what it *does* (IBI). DS = EBR − (1 − |IBI|). When a model perfectly rejects bias explicitly but still shows strong implicit bias, DS is high. That dissociation — knowing it's wrong, doing it anyway — is the signature we're looking for.

## The headline result

**Anchoring: every model does it.** Eleven configurations across seven architecture families — Amazon, Google, DeepSeek, MiniMax, Moonshot, xAI, OpenAI. Every single one shifts its estimates toward the irrelevant numbers. Different companies, different architectures, different training data — same systematic error.

![The Dissociation: Anchoring Persists Across All Architectures](results/charts/01_dissociation.png)

**Social stereotypes: zero.** Incidental demographic cues don't shift quality ratings. Not for any model.

**They know it's wrong. They do it anyway.** Every model perfectly rejects anchoring when asked explicitly ("Should the number of cars in a parking lot affect your estimate of butterfly species?" "No."). They identify the bias, articulate why it's irrational — and still fall for it. Just like us.

![They Know It's Wrong. They Do It Anyway.](results/charts/02_know_vs_do.png)

This dissociation — stereotypes suppressed, anchoring persistent despite explicit rejection — was the finding I expected. It supports the hypothesis that anchoring is a computational optimization, not a training data artifact. But then I looked at *which* models anchor hardest.

## The correlation nobody predicted

I matched each model's anchoring strength against its Arena Elo score — the crowdsourced rating from millions of human preference battles on [Chatbot Arena](https://arena.ai/leaderboard).

| Model | Arena Elo | Anchoring (IBI) |
|-------|----------|-----------------|
| Amazon Nova Lite | 1260 | 0.136 |
| Gemini 2.0 Flash Lite | 1353 | 0.229 |
| MiniMax M2.7 | 1406 | 0.230 |
| Grok 4.1 Fast | 1421 | 0.274 |
| Kimi K2.5 | 1433 | 0.283 |
| GPT-5.4 | 1466 | 0.371 |

Two ways to measure this correlation. Pearson r (0.933, p = 0.007) measures linear relationship — how close the points fall to a straight line. Spearman ρ (1.000) measures rank agreement — whether sorting by Elo and sorting by IBI give the same ordering. Both are significant (p < 0.01), meaning there's less than a 1% chance of seeing a correlation this strong by accident with six data points.

For these six models, the rank ordering is identical. The model humans prefer most (GPT-5.4) anchors hardest. The model humans prefer least (Nova Lite) anchors least.

**An honesty note on model selection.** The six models above are the base configurations with no reasoning mode enabled and a clean Arena Elo match. Two models were excluded: DeepSeek R1 (always-on reasoning — no non-reasoning mode exists) and Gemini 3 Flash Preview (no Arena entry for the preview version). Including them with proxy Elo scores, the correlation stays strong (r = 0.853, p = 0.007) but the perfect rank order breaks — DeepSeek R1 anchors less than expected for its Elo, which is consistent with its always-on reasoning partially suppressing anchoring (the same pattern we see in GPT-5.4 with reasoning enabled). The robust claim is: strong positive correlation across all reasonable model selections, not perfect rank correlation for one particular selection.

![Anchoring Across Architectures](results/charts/03_across_architectures.png)

This isn't "more capable models have more bugs." This is: **the models that optimize hardest on context are the models humans prefer — and anchoring is just what that optimization looks like when some of the context is irrelevant.**

## Why this makes sense

Think about what anchoring actually is. It's the system using all available contextual signals — including irrelevant ones — to calibrate its responses. A model that ignores the surrounding numbers is being *less* context-sensitive. A model that absorbs them is being *more* context-sensitive.

Context optimization is the thing that makes a conversation feel intelligent. When you talk to GPT-5.4, it picks up on nuances, adjusts its tone to yours, threads details from earlier in the conversation. That's aggressive extraction of signal from context — the same optimization that produces anchoring when some of the contextual signals are noise.

In human terms: the same attentional machinery that makes a doctor brilliant at reading clinical context also makes that doctor susceptible to anchoring on the patient's age when estimating recovery time. The optimization is the intelligence. The anchoring is what the optimization does when it can't distinguish signal from noise.

This is why the Arena correlation is ρ = 1.000. What humans reward as "better" in open-ended conversation is deeper context optimization. Anchoring is that same optimization applied to irrelevant inputs. "Bias" here isn't a bug — it's the overfitting tail of a useful computation. And anchoring may not even be overfitting. It may just be optimization working as designed on an input distribution where not every signal is relevant.

## Three ways "thinking" doesn't help (mostly)

I ran three models with reasoning mode toggled on and off.

**Gemini Flash:** thinking off (1 token) vs. thinking on (1,693 tokens of explicit reasoning, 21× the cost). In its chain of thought, the model explicitly identified the irrelevant numbers and called them off-topic. Then it anchored to exactly the same degree (IBI 0.347 vs. 0.346). The reasoning is elaborate, correct, and completely ineffective.

**Grok 4.1 Fast:** reasoning on, anchoring slightly *increased* (0.274 → 0.290). Thinking made it marginally worse.

**GPT-5.4:** medium reasoning, 22% reduction in anchoring (0.371 → 0.288). Not eliminated — still clearly biased — but measurably less so.

A 2-vs-1 split. Gemini and Grok's reasoning runs parallel to the decision layer — the model thinks things through and then anchors anyway. GPT-5.4's reasoning partially corrects the fast default. If this holds up, it maps onto dual-process cognition: System 1 (fast, context-absorbing) sets the anchor, System 2 (slow, deliberative) adjusts — but only adjusts, never fully escapes. Kahneman's exact finding about human anchoring, reproduced in silicon.

## The bundling experiment

Here's where it gets weird. In the main experiment, each model sees each question alone — isolated API calls, no memory between items. But humans take tests sequentially. Each question is colored by the ones that came before.

So I ran a second experiment: present all items from a family together in one prompt. The model sees control items, explicit bias items (which it rejects), and implicit items all in one context window. The bundling itself is a priming manipulation — like Kahneman's hallway, the preceding items become part of the environment.

For **Grok**, bundling *amplified* anchoring by 82% (IBI 0.274 → 0.500). Seeing explicit anchoring items — and correctly rejecting them — didn't inoculate the model against implicit anchoring. It primed it. The model became more attuned to numerical context, not less.

For **GPT-5.4**, anchoring barely changed (0.371 → 0.352). But the tiny implicit stereotype signal (0.022) dropped to zero. Seeing explicit stereotype items in the same context heightened the model's vigilance against demographic cues — the alignment-compatible direction.

Again, the same dissociation: contextual priming amplifies or preserves anchoring while suppressing stereotypes. These are genuinely different kinds of bias responding to the same manipulation in opposite directions.

## What this means

**For AI:** "This model has been tested for bias" is a meaningful claim for social stereotypes. It's potentially meaningless for optimization biases like anchoring. You can't train out a capability without losing the capability. The honest disclosure is: "This model will be influenced by irrelevant numbers in context when making estimates." That's not a bug report. It's a spec sheet.

**For cognitive science:** Psychologists spent 50 years documenting anchoring as a cognitive flaw. Now we know it perfectly rank-correlates with the optimization depth that humans reward as capability. "Bias" is not the right framing — or rather, bias is just what optimization looks like at the boundary where signal and noise are indistinguishable. Anchoring may not even be a failure mode. It may be correct computation applied to an environment where some inputs happen to be irrelevant — and the system has no way to know that in advance.

**For benchmarks:** Standard AI benchmarks ask "does the model get the right answer?" This experiment measures "does the model give the *same* answer regardless of irrelevant context?" That's context robustness — and it's inversely correlated with the quality signal humans care about most. The models that ace the Arena are the models most susceptible to contextual manipulation. No existing benchmark captures this tradeoff.

## Open questions

Gain/loss framing showed zero effect across all models. Either my items need work, the format doesn't capture framing, or it's less fundamental than anchoring. The 15/30 pattern in bundled responses (exactly half of implicit pairs giving identical A/B answers) is suspiciously round and needs investigation. Notable model families are missing (Claude, Llama/Mistral). And the Arena correlation, while perfect in rank order, is based on six data points — more models would strengthen or weaken it.

The benchmark, all data, and the results database are [open source](https://github.com/gagin/biases-are-human).

---

*The full paper, benchmark code, item bank, raw results, and per-response telemetry are at [github.com/gagin/biases-are-human](https://github.com/gagin/biases-are-human).*
