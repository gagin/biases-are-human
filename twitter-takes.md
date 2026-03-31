# Twitter Takes

Short-form hooks for the bias dissociation findings. Each is self-contained.

---

The most expensive way to get the same bias: Gemini Flash answers a multiple-choice question in 1 token for $0.0001. Turn on "thinking mode" — it spends 1,693 tokens reasoning, explicitly identifies the irrelevant numbers, calls them off-topic. Same answer. Same bias. 21x the cost. Grok does the same thing — reasoning on, bias slightly *worse*. Only GPT-5.4 partially self-corrects.

---

"AI biases are learned from training data" is the comforting version. The uncomfortable version: some biases are what correct computation looks like under resource constraints. You can't train them out without making the model worse at math. We tested this. The models that are best at numerical reasoning anchor the hardest on irrelevant numbers.

---

We tested AI models for anchoring bias — the same cognitive bug Kahneman documented in humans. Every model has it. 11 configurations, 7 architecture families, 4,320 responses. The more capable the model, the stronger the effect (r=0.79). Getting smarter doesn't fix it. Getting smarter makes it worse. Just like us.

---

If a random number generator beats hedge fund quants, you don't say the generator is intelligent. You ask whether the quants were ever doing anything more than the generator. Same question when GPUs reproduce human cognitive biases from scratch.

---

We ran the same bias test on three models with thinking turned off and on. Gemini: 1,693 tokens of reasoning, explicitly says "these numbers are irrelevant," anchors anyway. Grok: thinking on, bias slightly *increases*. GPT-5.4: partial correction, 22% reduction. 2 out of 3 models can't think their way out of anchoring. Just like us.

---

Every model we tested, same systematic error. 7 companies, 7 architectures, 11 configurations — r=0.79, more capable models anchor harder. This isn't a bug in one training dataset. It's a property of how bounded prediction systems process magnitude.

---

The alignment implications: "this model has been tested for social bias" ✓ achievable, verifiable, already working. "This model will not be influenced by irrelevant numbers in context" ✗ may be impossible without degrading capability. Honesty > false promises.

---

Psychologists spent 50 years documenting cognitive biases and assuming they were human-specific. AI reproduced the mathematical ones from scratch and rejected the cultural ones. Turns out Kahneman wasn't documenting human nature. He was documenting the math of bounded inference.

---

Three models. Same bias test. Reasoning toggled on and off. Gemini: no change. Grok: slightly worse. GPT-5.4: 22% better. "Thinking" is not one thing. Some architectures can partially correct their own biases through deliberation. Others generate elaborate reasoning chains that change nothing. The dual-process theory of cognition, replicated in silicon — but only in some silicon.

---

We measured anchoring bias in six models from six companies. Then we plotted it against their Chatbot Arena Elo rating. Perfect rank correlation. ρ = 1.000. The model humans prefer most anchors hardest. The model humans prefer least anchors least. "Bias" is the wrong word. It's optimization — the same optimization humans reward as intelligence. Anchoring is just what it looks like when some of the context is noise.

---

Kahneman spent decades documenting anchoring as a cognitive flaw. Turns out it perfectly rank-correlates with what 5.6 million Chatbot Arena voters reward as capability. "Bias" isn't a bad word — it's the overfitting tail of optimization. And anchoring might not even be overfitting. It might just be optimization doing its job on inputs where some signals happen to be irrelevant.

---

We bundled our bias test items together instead of sending them one by one. For Grok, seeing explicit anchoring items and correctly rejecting them made implicit anchoring 82% WORSE. The model didn't learn to resist bias from seeing it. It got primed. Knowing about a bias doesn't just fail to fix it — it can amplify it.
