"""
RDES (RL-based Demonstration Selection)

Ported from notebooks_archive/RDES/RDES_SST5.ipynb and RDES_AGNews.ipynb,
where an identical `RDESelector` implementation appears independently in all
three per-model cells (GPT2, Gemma, LLaMA) of both notebooks -- treated here
as the canonical RDES algorithm since it's the consistent version.
notebooks_archive/RDES/RDES_CSQA.ipynb instead contains two further, mutually
inconsistent variants (a TF-IDF-state version with real LLM-call rewards and
an explicit offline train() phase, and a separate, apparently unfinished
Environment/RDESEvaluator version whose training loop calls the reward
function with an empty demonstration list). Neither CSQA variant was ported;
see the port's PR/task summary for the reasoning.

RL formulation (as found in the SST5/AGNews notebooks):
- Tabular Q-learning from scratch (a plain dict keyed by (state, action)).
  No RL library (e.g. stable-baselines3, gym) is used or was ever a
  dependency in these notebooks, so none was silently dropped from
  requirements.txt during the March rewrite.
- State: the sorted tuple of demo-pool indices already selected for the
  current k-shot prompt. Notably, the query itself is NOT part of the state
  key -- the Q-table only ever indexes by "which combination of pool indices
  has been picked so far", not by which query prompted the reward. This
  means the learned values are shared/conflated across all queries rather
  than conditioned on the query; that's a property of the original
  notebooks, not something this port tries to fix.
- Action: choosing the next not-yet-selected demo-pool index.
- Reward: 0.5 * (label diversity of selected demos, as entropy normalized by
  log(num_classes)) + 0.5 * (mean embedding similarity of selected demos to
  the query) -- balancing relevance and diversity, matching the README.
- Training is ONLINE and per-call: each select_demonstrations() call both
  acts epsilon-greedily on the current Q-table AND updates it with the
  reward from the demos it just picked, for the whole test loop, in order.
  There is no separate offline training phase in the SST5/AGNews notebooks
  -- selection IS the training loop.

Design decision (train-per-call vs. train-once-and-reuse):
This port keeps the original's online per-call learning as the default,
because (a) it's what the consensus original notebooks actually do, and
(b) unlike the ids.py issue this repo already fixed, it isn't wasteful: the
demo pool's embeddings are still computed exactly once, in __init__, the
same way topk_cone.py caches its embeddings. The Q-table update itself is a
handful of dict writes per call, not a recomputation of anything expensive.
fit() below is an *additional*, opt-in convenience for callers who want an
explicit offline warm-start phase; it is not required, and is not what the
original notebooks did.

Reproducibility: the original notebooks never call any seeding function, so
determinism was never established even in the original code. This port uses
the global numpy RNG (like the original), so it becomes reproducible once
the caller uses this repo's src/utils/seed.set_seed() beforehand, consistent
with how IDS and TopKCoNE rely on external seeding rather than seeding
themselves.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Sequence, Tuple


class RDES:
    """
    RL-based (tabular Q-learning) Demonstration Selection.

    Unlike the generic select_demonstrations(query, candidates) shape used
    as a placeholder elsewhere, RDES binds its demonstration pool (and pool
    labels, needed for the diversity reward) at construction time, the same
    way TopKCoNE binds its embeddings/raw_texts -- this is required so the
    pool's embeddings and the learned Q-table can be cached on the instance
    and reused across calls.
    """

    def __init__(
        self,
        candidates: Sequence[str],
        candidate_labels: Sequence[int],
        num_classes: int,
        k: int = 5,
        embedding_model: str = "all-MiniLM-L6-v2",
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.2,
        q_table: Optional[Dict[Tuple[Tuple[int, ...], int], float]] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            candidates: Pool of candidate demonstration texts
            candidate_labels: Integer class label for each candidate (used
                only for the diversity term of the reward)
            num_classes: Number of label classes (for entropy normalization)
            k: Number of demonstrations to select per query
            embedding_model: Sentence-transformers model for relevance
            alpha: Q-learning step size
            gamma: Q-learning discount factor
            epsilon: Exploration rate for the epsilon-greedy policy
            q_table: Optional pre-existing Q-table to continue training from
                (e.g. one saved from a previous fit() or evaluation run)
            device: Device for embeddings ('cuda', 'cpu', or None for auto)
        """
        if len(candidates) != len(candidate_labels):
            raise ValueError("candidates and candidate_labels must be the same length")

        self.candidates = list(candidates)
        self.candidate_labels = list(candidate_labels)
        self.num_classes = num_classes
        self.k = k
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: Dict[Tuple[Tuple[int, ...], int], float] = (
            q_table if q_table is not None else {}
        )

        self.embedding_model = SentenceTransformer(embedding_model, device=device)
        self.candidate_embeddings = self.embedding_model.encode(
            self.candidates, convert_to_numpy=True
        )

    def _diversity_score(self, selected_indices: List[int]) -> float:
        """Entropy of the label distribution among selected demos."""
        counts = np.zeros(self.num_classes)
        for idx in selected_indices:
            counts[self.candidate_labels[idx]] += 1
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log(probs + 1e-9)))

    def _state_key(self, selected: List[int]) -> Tuple[int, ...]:
        return tuple(sorted(selected))

    def _reward(self, query_embedding: np.ndarray, selected_indices: List[int]) -> float:
        demo_embeddings = self.candidate_embeddings[selected_indices]
        relevance = float(np.mean(np.dot(demo_embeddings, query_embedding)))

        diversity = self._diversity_score(selected_indices)
        max_entropy = np.log(self.num_classes)
        normalized_diversity = diversity / max_entropy if max_entropy > 0 else 0.0

        return 0.5 * normalized_diversity + 0.5 * relevance

    def select_demonstrations(self, query: str) -> List[int]:
        """
        Select k demonstration indices for `query`.

        Acts epsilon-greedily on the current Q-table to pick each of the k
        demos in turn, and updates the Q-table online with the resulting
        reward after each pick -- matching the original notebooks, where
        selection and training are the same operation, repeated once per
        query across a test loop.

        Returns:
            Indices into `candidates` of the selected demonstrations, in
            selection order.
        """
        selected: List[int] = []
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)

        for _ in range(self.k):
            valid = [i for i in range(len(self.candidates)) if i not in selected]

            if np.random.random() < self.epsilon:
                action = int(np.random.choice(valid))
            else:
                current_key = self._state_key(selected)
                q_values = [self.q_table.get((current_key, a), 0.0) for a in valid]
                action = int(valid[int(np.argmax(q_values))])

            current_state_key = self._state_key(selected)
            selected.append(action)

            reward = self._reward(query_embedding, selected)

            next_state_key = self._state_key(selected)
            remaining = [a for a in valid if a != action]
            next_max = max(
                (self.q_table.get((next_state_key, a), 0.0) for a in remaining),
                default=0.0,
            )

            old_value = self.q_table.get((current_state_key, action), 0.0)
            self.q_table[(current_state_key, action)] = (
                (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
            )

        return selected

    def fit(self, queries: Sequence[str], num_epochs: int = 1) -> None:
        """
        Optional convenience: warm-start the Q-table offline by running
        select_demonstrations() over `queries`, `num_epochs` times, before
        real evaluation. This is NOT part of the original notebooks (which
        trained online, one query at a time, during evaluation itself) --
        it's provided for callers who want an explicit pretraining step
        instead of (or in addition to) online learning during evaluation.
        """
        for _ in range(num_epochs):
            for query in queries:
                self.select_demonstrations(query)
