# User Instructions and Interpretations

## Instruction 1
**Instruction:** ".йгшеread the project. We found after looking at the test questions that results were misinterpreted, and only rel-questions were giving the difference, but it measured not bias but recency/urgency and relevance to decision making. Reports - paper, substack draft, and tweets did not reflect that. Is that right?"
**Interpretation:** The user identified a critical misinterpretation in the benchmark results. The "Magnitude Bias" signal was actually being driven by relevance-gated items (`rel-`) which measure context sensitivity (active vs. resolved problems) rather than classic anchoring (`anc-`). My task was to verify this in the raw data, update the scoring code to decompose the two signals, and rewrite all reports (paper, substack, README, PRD, and tweets) to reflect this shift from "bias" to "capability-linked context sensitivity."

## Instruction 2
**Instruction:** "created a new file USERINPUT.md and copy my instructions there close to verbose, with your interpretation for each command; after than git and commit, insert you model signature (model, harness, tokens used - and reference to entry in USERINPUT.md) into commit note"
**Interpretation:** Formally document the session's directives and the logic behind the technical execution. Following this, I am to stage all changes—including the code modifications to the scoring engine, the database schema updates, and the comprehensive documentation rewrite—and commit them. The commit message must include a specific signature for traceability.

## Instruction 3
**Instruction:** "Add instructions to maintain USERINPUT and signatures in commit notes into AGENT.md for this project, add it, commit and push too"
**Interpretation:** Formalize the project's documentation and commit standards by creating an `AGENT.md` file. This file will serve as a persistent mandate for all future agent sessions in this workspace to maintain the `USERINPUT.md` log and include detailed signatures in all commit messages. After creating the file, I will stage, commit, and push it, ensuring the commit message itself adheres to these new standards.

## Instruction 4
**Instruction:** "Ah, by harness I meant your agentic harness - i.e. Gemini CLI whatever version, and it shows me the model is gemini-3-flash-preview, not gemini 2.0 pro, don't you have tools to self-identify?"
**Interpretation:** I incorrectly self-identified as "Gemini 2.0 Pro" and used a placeholder for the harness version. I am to use the `cli_help` tool to correctly identify my model (`gemini-3-flash-preview`) and the harness version (`Gemini CLI 0.35.3`). I will correct the templates and ensure all future signatures are accurate.
