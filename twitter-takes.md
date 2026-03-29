# Twitter Takes

Short-form hooks for the bias dissociation findings. Each is self-contained.

---

The most expensive way to get the same bias: Gemini Flash answers a multiple-choice question in 1 token for $0.0001. Turn on "thinking mode" — it spends 1,693 tokens reasoning, explicitly identifies the irrelevant numbers, calls them off-topic. Same answer. Same bias. 21x the cost.

---

"AI biases are learned from training data" is the comforting version. The uncomfortable version: some biases are what correct computation looks like under resource constraints. You can't train them out without making the model worse at math. We tested this. The models that are best at numerical reasoning anchor the hardest on irrelevant numbers.

---

We tested AI models for anchoring bias — the same cognitive bug Kahneman documented in humans. Every model has it. The more capable the model, the stronger the effect (r=0.79, p=0.034). Getting smarter doesn't fix it. Getting smarter makes it worse. Just like us.

---

If a random number generator beats hedge fund quants, you don't say the generator is intelligent. You ask whether the quants were ever doing anything more than the generator. Same question when GPUs reproduce human cognitive biases from scratch.

---

We ran the same bias test on a model with thinking turned off (1 token, $0.0001) and thinking turned on (1,693 tokens, $0.005). It explicitly said "these numbers are irrelevant" in its reasoning chain. Then it anchored on them anyway. Knowing about a bias doesn't fix it. Not for humans, not for AI.

---

Every model we tested, same systematic error. r=0.79, p=0.034 — more capable models anchor harder. This isn't a bug in one training dataset. It's a property of how bounded prediction systems process magnitude.

---

The alignment implications: "this model has been tested for social bias" ✓ achievable, verifiable, already working. "This model will not be influenced by irrelevant numbers in context" ✗ may be impossible without degrading capability. Honesty > false promises.

---

Psychologists spent 50 years documenting cognitive biases and assuming they were human-specific. AI reproduced the mathematical ones from scratch and rejected the cultural ones. Turns out Kahneman wasn't documenting human nature. He was documenting the math of bounded inference.
