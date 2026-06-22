# TakeMeter — Planning Document
## Milestone 1: Community, Labels, and Taxonomy

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

## Next Steps

- Collect 200+ labeled examples (posts from r/MachineLearning)
- Split into train / validation / test sets
- Fine-tune `distilbert-base-uncased` on labeled dataset
- Compare against Groq `llama-3.3-70b-versatile` zero-shot baseline
- Evaluate with accuracy, per-class F1, and confusion matrix
