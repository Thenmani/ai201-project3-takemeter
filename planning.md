# TakeMeter — Planning Document
## Fine-Tuned Text Classifier for r/MachineLearning Discourse

---

## Community

**Chosen community:** r/MachineLearning (reddit.com/r/MachineLearning)

r/MachineLearning is one of the largest and most active ML research communities on Reddit, with over 3 million subscribers. Posts range from shared research papers and project announcements to open-ended career discussions and technical debates. The community is text-heavy and opinion-driven, making it ideal for discourse quality classification. The distinctions between a well-argued technical post, an open question, and a project announcement are real and meaningful to people in this community — members regularly upvote substantive analysis and push back on vague or unsupported claims.

---

## Label Taxonomy

### Label 1: `analysis`
**Definition:** The post makes a structured argument backed by specific evidence — data, benchmarks, code, papers, or verifiable observations. The post reaches a conclusion rather than opening a question.

**Example 1:**
> *"The famous METR AI time horizons graph contains numerous severe errors"*
>
> The author walks through specific methodological flaws: unverified human baselines, incentivized benchmarkers, train-test contamination, and biased sampling. Each claim is grounded in verifiable details from the original study. The post concludes that the graph is unreliable.

**Example 2:**
> *"AI-generated CUDA kernels silently break training and inference"*
>
> The author describes a specific reproducible bug — bf16 accumulation error in an embedding gradient kernel — explains why it only manifests under certain conditions (real text distribution + AdamW), and provides a root cause. Entirely evidence-driven.

---

### Label 2: `discussion`
**Definition:** The post asks a question, shares an opinion, or opens a conversation without a structured evidence-backed argument. The goal is to invite community input rather than demonstrate a conclusion.

**Example 1:**
> *"Where do you go for serious AI research discussion online?"*
>
> The poster describes what they're looking for (substantive technical conversation) and asks the community for recommendations. No argument is made — the post exists to gather opinions.

**Example 2:**
> *"How long does it realistically take to produce an ICML/NeurIPS/ICLR-level paper?"*
>
> A genuine question about research timelines, inviting personal experiences from the community. No data or conclusion — the post is a prompt for others to share.

---

### Label 3: `announcement`
**Definition:** The post shares a project, tool, paper, dataset, or news update. The goal is to inform the community about something new, not to argue a position or seek opinions.

**Example 1:**
> *"Introducing Papers Without Code"*
>
> A Hugging Face team member announces the relaunch of paperswithcode.co, describes its features, and invites feedback. The post is informational — it shares something new rather than arguing a point.

**Example 2:**
> *"Hi Reddit, I posted my Build Your Own LLM workshop to YouTube"*
>
> The author shares a newly published educational resource with a list of topics covered. The post's purpose is to make the community aware of something, not to debate or analyze.

---

## Hard Edge Case

**Post:** *"The famous METR AI time horizons graph contains numerous severe errors"*

**Why it's hard:** This post is detailed and evidence-heavy (citing specific flaws in methodology, sampling bias, and data contamination), which feels like `analysis`. But it also ends by inviting community discussion about scientific standards — which feels like `discussion`.

**Decision rule:** If the post's primary move is *building a case toward a conclusion*, label it `analysis` — even if it invites follow-up discussion. If the post's primary move is *opening a question for the community to weigh in on*, label it `discussion` — even if it cites some supporting details.

In this case → **`analysis`**, because the bulk of the post is a structured critique with verifiable claims. The closing invitation to discuss is secondary to the argument being made.

---

## Why These Distinctions Matter

In r/MachineLearning, the community places high value on rigorous, evidence-based posts. The difference between a post that argues from data versus one that just asks a question is visible and meaningful to regular members — it determines whether a post gets cited, challenged, or simply answered. These three labels capture the three dominant modes of communication in this community: arguing, asking, and sharing.

---

---

## Data Collection Plan

**Source:** Reddit r/MachineLearning via the PRAW Python library (Reddit's official API). Posts will be collected from the `top` and `hot` feeds across multiple time filters (`month`, `year`, `all`) to ensure variety across post types and time periods.

**Target per label:**
- `analysis` — 70 examples
- `discussion` — 70 examples
- `announcement` — 70 examples

Total target: **210 examples** (buffer above the 200 minimum).

**If a label is underrepresented after 200 examples:**
`announcement` posts are the most common in this subreddit, so `analysis` is the label most likely to be underrepresented. If fewer than 60 `analysis` posts are found in the initial pull, I will search specifically for posts tagged `[D]` (Discussion) with strong upvote ratios and long text bodies, which tend to contain structured arguments. I will also extend the time filter to `all` to widen the pool.

**Train / Validation / Test split:**
- Train: 140 examples (~67%)
- Validation: 35 examples (~17%)
- Test: 35 examples (~17%)

Split will be stratified by label to ensure all three labels are represented proportionally in each set.

**CSV columns:** `id`, `title`, `text`, `score`, `label`, `split`

---

## Evaluation Metrics

**Why accuracy alone is not enough:**
If the label distribution is uneven (e.g., 50% `announcement`, 30% `discussion`, 20% `analysis`), a model that always predicts `announcement` would achieve 50% accuracy while being completely useless. Accuracy hides per-class failures.

**Metrics I will use:**

- **Overall accuracy** — baseline comparison between fine-tuned model and Groq zero-shot baseline.
- **Per-class F1 score** — harmonic mean of precision and recall for each label. This catches cases where the model learns one label well but fails on others. F1 is especially important for `analysis`, which is likely the hardest and most underrepresented label.
- **Confusion matrix** — shows exactly which labels are being confused with which. Expected confusion: `analysis` vs `discussion` (the hard boundary). The matrix will reveal whether the model is making the right kinds of mistakes.
- **Macro F1** — average F1 across all three labels, weighted equally. This is the primary single-number metric for this task because the labels are roughly balanced and all three matter equally.

---

## Definition of Success

**Minimum threshold for "good enough":**
- Overall accuracy ≥ 70% on the test set
- Macro F1 ≥ 0.65 across all three labels
- No single label with F1 < 0.50 (the model must learn something about every label)

**Threshold for "genuinely useful" in a real community tool:**
- Overall accuracy ≥ 80%
- Macro F1 ≥ 0.75
- `analysis` F1 ≥ 0.70 specifically (this is the hardest label and most valuable to identify correctly)

**Baseline comparison:**
The fine-tuned DistilBERT must outperform the Groq `llama-3.3-70b-versatile` zero-shot baseline on macro F1. If it does not, fine-tuning has not helped and the label taxonomy or dataset quality needs revisiting.

**What would indicate failure:**
- Any label with F1 < 0.40 means the model has not learned that label at all
- Test accuracy > 95% likely indicates label leakage or labels that are too easy — worth investigating
- If fine-tuned model performs worse than zero-shot baseline, dataset quality is suspect

---

## AI Tool Plan

### 1. Label Stress-Testing — YES, I will use AI for this
Before annotating a single example, I will give Claude my three label definitions and the `analysis` vs `discussion` edge case description, and ask it to generate 10 posts that sit at the boundary between those two labels. If any generated posts cannot be cleanly classified using my decision rule, I will revise the definitions until they can be. This step happens before any labeling begins — it is the cheapest way to find holes in my taxonomy before they corrupt 200 labeled examples.

### 2. Annotation Assistance — YES, with strict review
I will use Claude to pre-label batches of 20 posts at a time by providing the label definitions and asking for a label with a one-sentence justification per post. I will then personally review every single pre-label and override any I disagree with. I am choosing to do this because labeling 200 posts manually from scratch is slow, and AI pre-labeling speeds up the process without removing my judgment — I still make every final call. In the CSV, I will add a column `prelabeled_by_ai` (True/False) to track which examples were AI-assisted, for disclosure in the final README.

### 3. Failure Analysis — YES, as a pattern-finding aid
After generating predictions on the test set, I will collect all wrong predictions and give them to Claude with the prompt: *"Here are posts my classifier got wrong. What patterns do you see? Are there systematic confusions between specific labels?"* I am choosing to do this because pattern-finding across 30+ wrong predictions is exactly the kind of task where AI adds speed without replacing judgment. I will verify every pattern Claude identifies by manually re-reading the relevant examples before writing anything up in my evaluation report. I will not report a pattern I cannot confirm myself.

---

## Why These Distinctions Matter

In r/MachineLearning, the community places high value on rigorous, evidence-based posts. The difference between a post that argues from data versus one that just asks a question is visible and meaningful to regular members — it determines whether a post gets cited, challenged, or simply answered. These three labels capture the three dominant modes of communication in this community: arguing, asking, and sharing.

---

## Stretch Features Plan

### Chosen Stretch Feature: Deployed Interface

**What:** A simple Gradio-based web interface that accepts a Reddit post title and text, runs it through the fine-tuned DistilBERT classifier, and displays the predicted label with confidence scores for all three labels.

**Why this one:** It is the most practical stretch feature given the project scope. It requires no additional data collection or labeling, directly showcases the fine-tuned model, and produces a tangible artifact that demonstrates the classifier works end-to-end. It also satisfies the "genuinely useful community tool" framing from the Definition of Success section.

**Implementation plan:**
- Use Gradio's `gr.Interface` to wrap the fine-tuned model's inference function
- Accept raw post text as input (title + body combined)
- Return a label distribution showing confidence for all three labels
- Include 3 pre-filled example posts (one per label) for easy testing
- Launch with `share=True` to generate a public URL from Colab
- Commit the interface code as the final cell in `notebooks/takemeter.ipynb`
- Document how to run it in the README

**Success criteria for the interface:**
- Accepts any free-form text input
- Returns predicted label and confidence scores for all 3 labels
- Runs without errors on the 3 provided example posts
- Public URL accessible via Gradio's share link

**Stretch features considered but not implemented:**

- *Inter-annotator reliability* — would require recruiting another person to label 30+ examples independently, which is not feasible given the project timeline. Cohen's kappa would be the appropriate metric.
- *Confidence calibration* — would require bucketing predictions by confidence range and computing accuracy per bucket. Interesting but lower priority than a working interface.
- *Error pattern analysis* — partially completed via the AI-assisted failure analysis in the evaluation report. A more systematic version (e.g., clustering wrong predictions by post length or label pair) is deferred.