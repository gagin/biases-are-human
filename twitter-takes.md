# Twitter Takes

Short-form hooks for the bias dissociation findings. Each is self-contained.

---

The most expensive way to get the same bias: Gemini Flash answers a multiple-choice question in 1 token for $0.0001. Turn on "thinking mode" — it spends 1,693 tokens reasoning, explicitly identifies the irrelevant numbers, calls them off-topic. Same answer. Same bias. 21x the cost. Grok does the same thing — reasoning on, bias slightly *worse*. Only GPT-5.4 partially self-corrects.

---

We tested AI models for classic anchoring — irrelevant numbers pulling estimates. Every model scored essentially zero. But a different bias showed up instead: relevance sensitivity. Tell the model a bridge will be overloaded by 7 tons (routing decision pending) — it rates this as highly significant. Same 7-ton overload, but it already happened with no damage? Much less significant. Same number, different framing, different answer. And the more capable the model, the bigger the gap.

---

"AI biases are learned from training data" is the comforting version. The uncomfortable version: the bias we found isn't even classic anchoring — it's relevance sensitivity, the tendency to weight active problems more heavily than resolved ones. Models don't absorb irrelevant numbers. They aggressively evaluate context: urgency, stakes, resolution status. The "bias" is that they overshoot on the active/resolved distinction. You can't train that out without making the model worse at triage.

---

If a random number generator beats hedge fund quants, you don't say the generator is intelligent. You ask whether the quants were ever doing anything more than the generator. Same question when GPUs reproduce human cognitive biases from scratch.

---

We ran the same bias test on three models with thinking turned off and on. Gemini: 1,693 tokens of reasoning, explicitly says "these numbers are irrelevant," anchors anyway. Grok: thinking on, bias slightly *increases*. GPT-5.4: partial correction, 22% reduction. 2 out of 3 models can't think their way out of it. Just like us.

---

We decomposed the magnitude bias signal. Classic anchoring (irrelevant numbers pulling estimates) = essentially zero in every model. Relevance sensitivity (active problem rated more significant than identical resolved one) = the entire signal. The more capable the model, the bigger the gap. Arena Elo correlation: anchoring r = -0.600 (n.s.), relevance sensitivity r = 0.944 (p = 0.005). What humans reward as intelligence isn't susceptibility to irrelevant numbers — it's aggressive context evaluation that overshoots on urgency.

---

The alignment implications: "this model has been tested for social bias" — achievable, verifiable, already working. "This model will not weight active problems differently from resolved ones" — may be impossible without degrading capability. Honesty > false promises.

---

Psychologists spent 50 years documenting anchoring as a cognitive flaw. We tested it in AI and found... essentially zero classic anchoring. What we found instead: relevance sensitivity — urgency-weighted significance assessment — scaling with capability. Kahneman wasn't wrong about human cognition. But the computational story may be about context evaluation, not number absorption.

---

Three models. Same bias test. Reasoning toggled on and off. Gemini: no change. Grok: slightly worse. GPT-5.4: 22% better. "Thinking" is not one thing. Some architectures can partially correct their own biases through deliberation. Others generate elaborate reasoning chains that change nothing. The dual-process theory of cognition, replicated in silicon — but only in some silicon.

---

We measured bias in six models from six companies. Then plotted it against Chatbot Arena Elo. Classic anchoring: no correlation (r = -0.600, not significant). Relevance sensitivity: r = 0.944, p = 0.005. The model humans prefer most differentiates active from resolved problems the hardest. "Bias" is the wrong word. It's context sensitivity — the same context sensitivity humans reward as intelligence. The bias is just the overshoot.

---

We bundled our bias test items together instead of sending them one by one. For Grok, seeing explicit bias items and correctly rejecting them made implicit bias 82% WORSE. The model didn't learn to resist bias from seeing it. It got primed. Knowing about a bias doesn't just fail to fix it — it can amplify it.

---

Classic anchoring in LLMs: essentially zero. Every model, every architecture. Irrelevant numbers in context don't move the needle. What *does* move the needle: telling the model this is an active problem vs. a resolved one. Same numbers, same question, different urgency framing — different answer. And the gap perfectly predicts which models humans prefer. The "bias" isn't about numbers at all. It's about how models read stakes.
