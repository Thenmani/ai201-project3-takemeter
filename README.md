# TakeMeter — Discourse Quality Classifier for r/MachineLearning

A fine-tuned text classifier that categorizes posts from r/MachineLearning into three discourse types: `analysis`, `discussion`, and `announcement`. Built on `distilbert-base-uncased` and evaluated against a Groq zero-shot baseline.

---

## Community Choice

**Community:** r/MachineLearning (reddit.com/r/MachineLearning)

r/MachineLearning is one of the largest ML research communities on Reddit with over 3 million subscribers. It was chosen because the discourse is genuinely varied — posts range from structured technical arguments backed by benchmarks and papers, to open-ended career questions, to project and tool announcements. The distinction between a well-argued technical post and a casual question is real and meaningful to people in this community. Members regularly upvote substantive analysis and push back on unsupported claims, making discourse quality a natural classification target.

---

## Label Taxonomy

### `analysis`
The post makes a structured argument backed by specific evidence — data, benchmarks, code, papers, or verifiable observations. The post reaches a conclusion rather than opening a question.

**Example 1:**
> *"AI-generated CUDA kernels silently break training and inference"* — The author describes a specific reproducible bug (bf16 accumulation error in an embedding gradient kernel), explains why it only manifests under certain conditions, and provides a root cause. Entirely evidence-driven.

**Example 2:**
> *"The famous METR AI time horizons graph contains numerous severe errors"* — The author walks through specific methodological flaws: unverified human baselines, incentivized benchmarkers, train-test contamination, and biased sampling. Concludes the graph is unreliable.

---

### `discussion`
The post asks a question, shares an opinion, or opens a conversation without a structured evidence-backed argument. The goal is to invite community input rather than demonstrate a conclusion.

**Example 1:**
> *"Where do you go for serious AI research discussion online?"* — The poster describes what they're looking for and asks for recommendations. No argument made — the post exists to gather opinions.

**Example 2:**
> *"How long does it realistically take to produce an ICML/NeurIPS/ICLR-level paper?"* — A genuine question about research timelines, inviting personal experiences from the community.

---

### `announcement`
The post shares a project, tool, paper, dataset, or news update. The goal is to inform the community about something new, not to argue a position or seek opinions.

**Example 1:**
> *"Introducing Papers Without Code"* — A Hugging Face team member announces the relaunch of paperswithcode.co and describes its features. Informational, not argumentative.

**Example 2:**
> *"Hi Reddit, I posted my Build Your Own LLM workshop to YouTube"* — The author shares a newly published educational resource. The post's purpose is awareness, not debate.

---

## Data Collection

**Source:** r/MachineLearning via Reddit's JSON API (`/top.json`), pulling from `month`, `year`, `all`, and `hot` feeds to ensure variety across time periods and post types.

**Collection process:** Fetched using Python's `requests` library in Google Colab. Posts with empty, removed, or deleted text were filtered out. Duplicate posts (appearing in multiple feeds) were removed by post ID.

**Labeling process:** Posts were pre-labeled in batches using Claude (Cowork) with explicit label definitions. Every pre-label was reviewed manually and corrected where needed. All AI-assisted labels are tracked in the `prelabeled_by_ai` column of the dataset.

**Label distribution:**
| Label | Count |
|---|---|
| analysis | 70 |
| discussion | 71 |
| announcement | 72 |
| **Total** | **213** |

**Train / Validation / Test split:** 149 / 32 / 32 (stratified by label)

---

### 3 Difficult-to-Label Examples

**1. "NeurIPS used uncalibrated AI detector for desk rejections"**
Could be `analysis` (cites specific detector scores, methodological critique) or `discussion` (invites community debate about scientific standards). Decision: **`analysis`** — the bulk of the post is a structured critique with verifiable claims. The closing invitation to discuss is secondary.

**2. "Anthropic walks back policy on silent nerfing for AI/ML"**
Could be `announcement` (shares news) or `discussion` (community reacts and debates). Decision: **`announcement`** — the primary move is sharing a news update, not arguing a position.

**3. "Should ArXiv backtrack endorsement?"**
Could be `analysis` (makes a structured argument about endorsement accountability) or `discussion` (ends with an open question to the community). Decision: **`discussion`** — the post's primary move is opening a question, even though it provides some supporting reasoning.

---

## Fine-Tuning Approach

**Base model:** `distilbert-base-uncased` — chosen for its small size (66M parameters), fast training on a T4 GPU, and strong performance on text classification tasks.

**Training setup:**
- Epochs: 6
- Learning rate: 2e-5
- Batch size: 16 (train), 32 (eval)
- Weight decay: 0.01
- Warmup steps: 50
- Max token length: 256

**Key hyperparameter decision:** The notebook default was 3 epochs, which produced only 46.9% validation accuracy. After increasing to 6 epochs, accuracy jumped to 75.0% on the validation set. The model appeared to hit a learning cliff between epochs 4 and 5, suggesting the task needed more passes through the data to generalize. This was the single most impactful change made to the defaults.

**Text preprocessing:** Post title and body text were concatenated into a single `text` field before tokenization, since titles alone often lack enough context to determine the post type.

---

## Baseline Description

**Model:** Groq `llama-3.3-70b-versatile` (zero-shot, no task-specific training)

**Prompt used:**

```
You are classifying posts from r/MachineLearning, a machine learning research community on Reddit.
Assign each post to exactly one of the following categories.

analysis: The post makes a structured argument backed by specific evidence — data, benchmarks, code, papers, or verifiable observations. The post reaches a conclusion.
Example: "AI-generated CUDA kernels silently break training — the embedding gradient kernel accumulates in bf16 instead of fp32, causing loss divergence only under real text distributions."

discussion: The post asks a question or opens a conversation without a structured evidence-backed argument. The goal is to invite community input.
Example: "How long does it realistically take to produce an ICML/NeurIPS paper? Looking for honest timelines from people in the field."

announcement: The post shares a project, tool, paper, dataset, or news update. The goal is to inform the community about something new.
Example: "We just released PapersWithCode revival — browse CVPR 2026 papers by task, with linked GitHub repos and evals."

Respond with ONLY the label name.
Do not explain your reasoning.

Valid labels:
analysis
discussion
announcement
```

**How results were collected:** Each of the 32 test posts was passed to Groq's API individually with `temperature=0` and `max_tokens=20`. Responses were matched to label strings. All 32/32 responses were parseable.

---

## Evaluation Report

### Overall Accuracy

| Model | Accuracy |
|---|---|
| Zero-shot baseline (Groq) | 62.5% |
| Fine-tuned DistilBERT | **68.8%** |
| **Improvement** | **+6.3%** |

---

### Per-Class Metrics

**Fine-tuned DistilBERT:**

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| analysis | 0.57 | 0.36 | 0.44 | 11 |
| discussion | 0.64 | 0.70 | 0.67 | 10 |
| announcement | 0.79 | 1.00 | 0.88 | 11 |
| **macro avg** | **0.66** | **0.69** | **0.66** | 32 |

**Zero-shot baseline (Groq):**

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| analysis | 0.00 | 0.00 | 0.00 | 11 |
| discussion | 0.48 | 1.00 | 0.65 | 10 |
| announcement | 0.91 | 0.91 | 0.91 | 11 |
| **macro avg** | **0.46** | **0.64** | **0.52** | 32 |

---

### Confusion Matrix

![Confusion Matrix](outputs/confusion_matrix.png)

**As a table (Fine-tuned DistilBERT):**

| True \ Predicted | analysis | discussion | announcement |
|---|---|---|---|
| analysis | 4 | 4 | 3 |
| discussion | 3 | 7 | 0 |
| announcement | 0 | 0 | 11 |

---

### 3 Wrong Predictions — Error Analysis

**Error #1 — `analysis` predicted as `announcement` (confidence: 0.42)**
> *"ICML rejects papers of reviewers who used LLMs despite agreeing not to"*

The post reports a specific policy event (ICML rejection) which resembles a news announcement. However the body makes a structured argument about fairness and process integrity. The model was misled by the news-event framing in the title. This is a surface-level signal error — the title pattern matched `announcement` but the body content was `analysis`.

**Error #2 — `discussion` predicted as `analysis` (confidence: 0.39)**
> *"How do ML practitioners select hyperparameters for self-supervised learning when loss is non-monotonic?"*

The post contains dense technical terminology (BYOL, JEPA, data2vec, RankMe) which typically appears in `analysis` posts. But the post is genuinely asking a question — the technical vocabulary misled the model into thinking it was a structured argument. This reveals the model learned technical jargon as a proxy signal for `analysis`.

**Error #3 — `analysis` predicted as `discussion` (confidence: 0.42)**
> *"Can we stop glazing big labs and universities?"*

The post opens with a rhetorical question, which is a strong `discussion` surface signal. But the body makes a structured argument with specific observations about attribution practices. The model was misled by the question framing in the title while missing the argumentative content in the body.

---

### Sample Classifications

| Post (truncated) | True Label | Predicted | Confidence |
|---|---|---|---|
| "AI-generated CUDA kernels silently break training..." | analysis | analysis | 0.71 |
| "Where do you go for serious AI research discussion?" | discussion | discussion | 0.68 |
| "Introducing Papers Without Code [P]" | announcement | announcement | 0.94 |
| "ICML rejects papers of reviewers who used LLMs..." | analysis | announcement | 0.42 |
| "How do ML practitioners select hyperparameters..." | discussion | analysis | 0.39 |

---

### One Correct Prediction Explained

**Post:** *"Introducing Papers Without Code"*
**True:** `announcement` → **Predicted:** `announcement` (confidence: 0.94)

This is the model's most confident correct prediction. The post contains strong `announcement` signals: a named product launch, feature descriptions, an author identifying their affiliation ("Niels here from the open-source team at Hugging Face"), and a call for feedback. No argument is made, no question is asked. The model correctly identified all these signals and assigned the label with high confidence.

---

## Reflection

### What the model learned vs. what I intended

I intended the model to learn the *communicative function* of each post — is it arguing, asking, or sharing? What it actually learned were **surface-level lexical signals**:

- `announcement` → product names, "we built", "check it out", author affiliations
- `discussion` → question marks, "what do you think", "anyone else"
- `analysis` → technical terminology, numbers, paper citations

This works well for `announcement` (F1: 0.88) because the surface signals are reliable. It works reasonably for `discussion` (F1: 0.67). It fails most for `analysis` (F1: 0.44) because `analysis` posts often *open with* a question or framing that looks like `discussion`, with the evidence appearing later in the body.

The Groq baseline's complete failure on `analysis` (F1: 0.00) is the most revealing finding — even a 70B parameter model with explicit label definitions couldn't reliably identify `analysis` zero-shot. This confirms that `analysis` vs `discussion` is a genuinely hard boundary that requires either more training data or a different labeling strategy.

---

## Spec Reflection

**One way the spec helped:** Writing the decision rule for the `analysis` vs `discussion` edge case in `planning.md` before annotating was the single most valuable planning step. When ambiguous posts came up during labeling, I had a clear rule to apply rather than making inconsistent judgment calls. The stress-testing exercise (generating 10 boundary posts and classifying them) also confirmed the rule was stable before I committed to labeling 200 examples.

**One way implementation diverged from the spec:** My `planning.md` targeted 70 examples per label, but the initial AI pre-labeling produced only 26 `analysis` posts (12%). I had to re-prompt Claude Cowork with stricter instructions emphasizing that evidence + conclusion = `analysis` regardless of whether the post invites follow-up debate. The final distribution (70/71/72) matched the target, but required an extra annotation iteration not planned for in the spec.

---

## AI Usage

This project used AI assistance in three specific ways:

**1. Label stress-testing (Claude)**
Before annotating any examples, I gave Claude my three label definitions and asked it to generate 10 boundary posts between `analysis` and `discussion`. I classified all 10 correctly using my decision rule, confirming the definitions were stable. No revisions were needed.

**2. Annotation pre-labeling (Claude Cowork)**
Claude Cowork pre-labeled all 213 posts in two passes. The first pass produced a skewed distribution (111 discussion / 76 announcement / 26 analysis). I reviewed the labels, identified the systematic error (too many `analysis` posts labeled as `discussion`), and re-prompted with stricter instructions. The second pass produced a balanced distribution (71/72/70). I personally reviewed all labels and made corrections where I disagreed with Claude's judgment — particularly on boundary cases between `analysis` and `discussion`.

**3. Failure analysis (Claude)**
After generating wrong predictions, I shared them with Claude and asked it to identify patterns. Claude identified three patterns: news-framing misleading the model away from `analysis`, technical vocabulary misleading the model toward `analysis`, and rhetorical questions misleading the model away from `analysis`. I verified each pattern by manually re-reading the relevant wrong predictions before including them in this report.

---

## Stretch Feature: Deployed Interface

### What was built

A Gradio-based web interface that accepts any Reddit post text, runs it through the fine-tuned DistilBERT classifier, and displays the predicted label with confidence scores for all three labels.

### How to run it

1. Open `notebooks/takemeter.ipynb` in Google Colab
2. Run all cells from Section 1 through Section 3 (loads and trains the model)
3. Run the final cell (Section 7 — Gradio Interface)
4. A public URL will appear — click it to open the interface in your browser

### Interface screenshot outcome

The interface was tested on all three example posts:

| Post | Predicted | Confidence |
|---|---|---|
| "AI-generated CUDA kernels silently break training..." | announcement | 40% |
| "Where do you go for serious AI research discussion online?" | discussion | ~60% |
| "Introducing PapersWithCode revival — browse CVPR 2026 papers..." | announcement | ~85% |

### Observations

The first example (CUDA kernels post) was predicted as `announcement` instead of `analysis` with only 40% confidence — consistent with the model's known weakness on `analysis` vs `announcement` boundary cases identified in the evaluation. The low confidence correctly signals uncertainty, which is meaningful behavior — the model knows it is unsure on hard cases.

The `discussion` and `announcement` examples classified correctly with higher confidence, consistent with the test set evaluation results (F1: 0.67 and 0.88 respectively).

### Stretch features considered but not implemented

- **Inter-annotator reliability** — would require recruiting another person to label 30+ examples independently. Cohen's kappa would be the appropriate metric. Not feasible given project timeline.
- **Confidence calibration** — would require bucketing predictions by confidence range and computing accuracy per bucket. Deferred due to time constraints.
- **Error pattern analysis** — partially completed via the AI-assisted failure analysis above. A more systematic version clustering wrong predictions by post length or label pair is deferred.
