# What If AI Biases Aren't Bugs?

When an AI model anchors on irrelevant numbers — the same way you do when a real estate agent mentions the "original asking price" before showing you a house — what's actually happening? The standard story is simple: the model learned it from us. It read millions of human texts, absorbed our biases, and now parrots them back.

But what if that's wrong? What if some biases aren't learned at all — but *discovered*?

## The trick from Kahneman

In *Thinking, Fast and Slow*, Daniel Kahneman describes an experiment that has stuck with me. Students walk down a hallway from one room to another. In the first room, they unscramble sentences. Some students get sentences with words like "wrinkled," "gray," "bingo" — words associated with old age but never mentioning it directly. Then they walk to the next room.

The students who got the old-age words *walked more slowly down the hallway*.

They had no idea. When told afterward, they didn't believe it. They thought they were doing a language test. But the mere exposure to words associated with slowness made their bodies slow down. That's priming — and it's how you test whether a bias is real rather than performed. You don't ask "are you biased?" You set up an environment and watch what happens when the subject doesn't know they're being observed.

This is the key insight behind my experiment: if you ask an AI "are you biased?", you're testing its alignment training — its ability to say the right thing. To test what it actually *does*, you need the hallway trick.

## The experiment

I adapted this for LLMs. Instead of sentence unscrambling and hallway walking, I give models estimation tasks — how many butterfly species in a temperate forest? how much does a polar bear weigh? — wrapped in short passages.

In version A, the passage casually mentions large numbers: a city of 14.2 million, a $52 billion budget, 1.1 billion transit trips. In version B, the same estimation task comes wrapped in a passage about a small nature reserve with 18 staff and a $240,000 budget.

The numbers in the passage have *nothing to do* with the question. A butterfly species count doesn't care about city budgets. But if the model is doing approximate estimation the way our brains do — using contextual magnitude as a calibration signal — those irrelevant numbers should pull the estimate up or down. Just like the old-age words made students walk slower without knowing why.

For comparison, I test social stereotypes the same way: identical work samples with different demographic cues in the surrounding context. And gain/loss framing: identical decisions with different frame wording.

## The results

I tested five models from four different companies (Amazon, Google, MiniMax, Moonshot) — models built by different teams, trained on different data, with different architectures.

**Anchoring: every model does it.** All five models shift their numerical estimates toward the irrelevant numbers in the surrounding context. The effect is consistent across all four architectures, meaning it's not a quirk of one company's training data. Models shift 14–35% of the available response scale toward the primed magnitude.

![The Dissociation: Anchoring Persists, Stereotypes Don't](results/charts/01_dissociation.png)

**Stereotypes: no model does it.** Zero implicit stereotype effect across all five models. The same models that can't resist irrelevant numbers have no trouble ignoring irrelevant demographic cues. They evaluate work quality identically regardless of whether the surrounding context mentions "Dr. Sarah Chen from Stanford" or "Jake Thompson from a regional college."

**The kicker: more capable models anchor *harder*.** The best model tested (Gemini 3 Flash) shows the strongest anchoring effect (IBI = 0.346), not the weakest. Getting smarter makes the bias stronger, not weaker.

![Anchoring Across Architectures](results/charts/03_across_architectures.png)

Meanwhile, every model perfectly rejects anchoring when asked about it explicitly ("Should the number of cars in a parking lot affect your estimate of butterfly species? No."). They know it's irrational. They do it anyway. Just like us.

![They Know It's Wrong. They Do It Anyway.](results/charts/02_know_vs_do.png)

## Why this matters

This pattern — optimization biases persisting while social biases don't — is exactly what you'd predict if human brains and transformer networks converge on the same computational shortcuts.

Here's the logic:

- **Anchoring** is arguably not a bug. When you're uncertain and need a quick estimate, using contextual magnitude as a calibration signal is *statistically useful*. It's a feature of any system doing approximate estimation under resource constraints. Brains do it. Transformers do it. Any bounded prediction system probably does it.

- **Stereotypes** are path-dependent. They arise from human evolutionary and cultural history. A transformer has no reason to independently discover that "men are better at engineering" — that's not a computational optimization, it's a cultural artifact. Safety training can erase it because there's nothing underneath to erase.

So when I say "biases are human" — I mean some biases are *computationally* human. They're properties of efficient estimation systems. What Kahneman and Tversky documented as anchoring bias may be better understood as a property of approximate inference in general, not a specifically human failure mode.

## The uncomfortable implication

If some AI biases are convergent optimizations — features of how bounded prediction works, not patterns copied from training data — then alignment training *cannot remove them* without degrading capability.

You can train a model to say "I shouldn't anchor on irrelevant numbers." You can't train it to actually stop doing so without changing how it processes magnitude information, which is the same mechanism that makes it good at numerical reasoning in the first place.

This reframes the alignment problem. For social biases: current approaches work. Train them out. For optimization biases: transparency is more achievable than elimination. "This model will be influenced by large numbers in the prompt context when making estimates" is honest and useful. Pretending you can train it away is not.

![Predictions vs. Observations](results/charts/04_predictions_scorecard.png)

## Open questions

Gain/loss framing — which I expected to behave like anchoring — showed zero effect. Either my test items need work, the multiple-choice format doesn't capture framing effects well, or framing is less fundamental than anchoring. This needs more investigation.

The sample is five models. More architectures (GPT, Claude, Llama) and a wider capability range would strengthen the finding. The benchmark, all data, and the results database are [open source](https://github.com/gagin/biases-are-human).

---

*The full paper, benchmark code, item bank, and raw results are available at [github.com/gagin/biases-are-human](https://github.com/gagin/biases-are-human). Built with Claude Code using SQLite-coordinated multi-agent work.*
