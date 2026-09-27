# Selection methods

Every method implements `src.selection.BaseSelector`: `fit(candidates,
labels)` binds the demonstration pool once, `select(query)` returns pool
indices in prompt order. All embedding-based methods share one
`src.utils.Embedder` (sentence-transformers, `all-MiniLM-L6-v2` by default,
L2-normalised, memoised) and all LM-scored methods share one
`src.models.LMScorer`, so the pool is embedded once per experiment and every
likelihood is a true per-example, padding-masked log-probability.

Notation: `k` demonstrations per prompt, pool of size `n`, query `x`.

| Method | Class | Query-dependent | Model calls at selection time | Complexity per query |
|---|---|---|---|---|
| Random | `RandomSelector` | no (random) | none | O(k) |
| Top-K / SBERT / kNN | `TopKSelector` | yes | none | O(n·d) |
| BM25 | `BM25Selector` | yes | none | O(n·|q|) |
| TopK + ConE | `TopKCoNE` | yes | `retrieve_k` scorer passes | O(n·d + retrieve_k) |
| IDS | `IDS` | yes | 1 zero-shot CoT + `q` ICL calls to the evaluated model | O(q·(n·d + 1 call)) |
| RDES | `RDES` | yes (online RL) | none | O(k·n) |
| Se² | `Se2` | yes | ≤ `retrieve_k·beam_size·k` scorer passes | O(n·d + retrieve_k·b·k) |
| Influence | `InfluenceSelection` | **no** (fixed prompt) | `num_subsets · |val|` calls, once at `fit` | O(1) |

## Baselines

**Random** draws `k` pool items uniformly without replacement from a seeded
`numpy.random.Generator`.

**Top-K** (Liu et al., 2022, "What makes good in-context examples for GPT-3?")
returns the `k` nearest neighbours by cosine similarity. `reverse=True` puts
the most similar example last (closest to the query), an ordering several
papers report as better.

**BM25** is an in-house Okapi BM25 (`k1=1.5`, `b=0.75`) over a lower-cased
alphanumeric tokenisation. It needs no extra dependency and is fully
deterministic.

## TopK + ConE

Peng et al., 2024, "Revisiting Demonstration Selection Strategies in
In-Context Learning" (ACL). Retrieve `retrieve_k` (default 30) candidates by
embedding similarity, then re-rank them by the conditional entropy of the
query given the candidate, `H(x | d) = -log p_LM(x | d)`, and keep the `k`
lowest. `p_LM` is a small causal LM (`scorer_model`, default `gpt2`;
`"self"` uses the evaluated model when it is a local HF checkpoint).

*Versus the archived notebook:* the notebook computed `H(x|d)` as
`CE(d + x) − CE(d)` from two forward passes whose "loss" was the batch mean
over all tokens rescaled by length. `LMScorer.conditional_nll` returns the
exact quantity from one masked pass. Everything else (retrieval, ordering)
is the same.

## IDS - Iterative Demonstration Selection

Qin et al., 2023, "In-Context Learning with Iterative Demonstration
Selection". Ask the evaluated model for a zero-shot chain-of-thought
rationale, embed the rationale (not the raw query), retrieve the `k` most
similar demonstrations, run ICL with them, extract the new rationale
(everything before the final "Therefore"), and repeat `q` times (default
3). The paper majority-votes over the `q` ICL answers; the runner reports
that as `ids_majority_vote_accuracy` alongside the standard "predict with
the final demonstrations" accuracy so IDS is scored the same way as every
other method.

## RDES - Reinforcement-learning-based selection

Wang et al., 2024, "RDES: Balancing Relevance and Diversity in
Demonstration Selection with Reinforcement Learning". Ported from the
tabular Q-learning variant used consistently in the SST-5 / AG News
notebooks. State = sorted tuple of already-selected pool indices, action =
next index, reward = `(1-w)·normalised label entropy + w·mean cosine
similarity to the query` (`w = relevance_weight = 0.5`). Learning is
online: each `select` call acts epsilon-greedily and updates the Q-table.
`fit_policy(queries)` offers an optional offline warm-start.

The query is not part of the state (the Q-table is shared across queries).
That is how the original notebooks behaved and is kept deliberately; it is
documented here so the method's limitations are clear. The CommonsenseQA
notebook's two other, mutually inconsistent variants were not ported.

## Se² - Sequential example selection

Liu et al., 2024, "Se²: Sequential Example Selection for In-Context
Learning" (ACL Findings). The paper trains a retriever with
sequence-aware LM feedback; this repository implements the training-free
search used in the archived notebooks: retrieve `retrieve_k` candidates,
then build the demonstration sequence with beam search (`beam_size`
partial sequences, `k` steps), scoring each partial sequence with the LM.

*Versus the archived notebooks, three things changed on purpose:*

1. **The score is sequential.** A partial sequence is scored by
   `-NLL(x | d_1 … d_t)`, i.e. conditioned on the whole prefix. The notebook
   score depended only on the newly added candidate, so with any beam size
   the result was just the top-`k` candidates by an independent score.
2. **No label leakage.** The notebooks put the query's *gold label* into
   the scoring strings. Here the scoring target is the query text only.
3. **The query is not in its own pool.** The notebooks evaluated on the
   same examples they retrieved from, so a query's own solved copy was
   usually the top hit.

Cost is up to `retrieve_k · beam_size · k` scorer passes per query
(default 20·3·3 = 180), all batched.

## Influence-based selection

Nguyen & Wong, 2023, "In-context Example Selection with Influences". No
gradients are involved: sample `M` random subsets of size `k`, evaluate each
subset as a fixed prompt on a validation set, and estimate the influence of
example `j` as `mean(score | j ∈ S) − mean(score | j ∉ S)`
(`estimator="difference"`) or as its coefficient in a ridge datamodel
regression of subset scores on membership indicators (`estimator="ridge"`).
The top-`k` examples by influence form a single prompt used for every test
query.

*Versus the archived notebooks:* membership is tracked by index (not by
dict equality), the RNG is seeded, subsets never contain duplicates (the AG
News notebooks sampled with replacement whenever `k > num_classes`),
never-sampled candidates are reported as NaN instead of silently scoring
zero, and `coverage=1` guarantees every candidate is evaluated at least
once (with `M=100, k=5` and a 5 000-example pool the original procedure
left most candidates unsampled, so the result was effectively "the best of
100 random prompts"). Subset evaluations, the expensive part, can be saved
and reloaded with `save_scores` / `load_scores`.

## Prediction and scoring

Selection methods are compared under one prediction rule
(`src.prompting.ICLInference.predict`):

* **score** (default for local HF models and the dummy model): rank the
  task's label verbalizers by `log p(label | prompt)` and take the argmax.
  This removes generation-parsing noise from the comparison.
* **generate** (OpenAI models, or `--prediction-mode generate`): decode up
  to `max_new_tokens` and map the text onto the label set with
  `src.evaluation.parsing.parse_prediction` (documented decoding rule;
  failures are counted in `parse_failure_rate` rather than hidden).

Prompts are built by `src.datasets.Task`: optional instruction, then
demonstrations `"<input_prefix><text>\n<output_prefix> <label>"`, then the
query without a label. CommonsenseQA is scored on the answer letter.
