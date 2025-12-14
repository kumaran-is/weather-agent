"""
Retrieval and Generation Metrics Module.

Level 6c: Self-Evolving Platform

Provides advanced evaluation metrics:
1. MRR (Mean Reciprocal Rank) - Retrieval quality
2. NDCG (Normalized Discounted Cumulative Gain) - Ranking quality
3. BLEU (Bilingual Evaluation Understudy) - Generation quality
4. ROUGE (Recall-Oriented Understudy) - Summary quality

Target: Comprehensive evaluation for RAG and generation tasks
"""

from typing import Any
import logging
import math
from collections import Counter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


class RetrievalMetrics(BaseModel):
    """Metrics for retrieval evaluation."""

    mrr: float = Field(ge=0.0, le=1.0, description="Mean Reciprocal Rank")
    mrr_at_k: dict[int, float] = Field(default_factory=dict, description="MRR@k for various k")
    ndcg: float = Field(ge=0.0, le=1.0, description="Normalized DCG")
    ndcg_at_k: dict[int, float] = Field(default_factory=dict, description="NDCG@k for various k")
    precision_at_k: dict[int, float] = Field(default_factory=dict, description="Precision@k")
    recall_at_k: dict[int, float] = Field(default_factory=dict, description="Recall@k")
    map_score: float = Field(ge=0.0, le=1.0, default=0.0, description="Mean Average Precision")
    num_queries: int = Field(default=0, description="Number of queries evaluated")


class GenerationMetrics(BaseModel):
    """Metrics for text generation evaluation."""

    bleu_1: float = Field(ge=0.0, le=1.0, description="BLEU-1 (unigram)")
    bleu_2: float = Field(ge=0.0, le=1.0, description="BLEU-2 (bigram)")
    bleu_3: float = Field(ge=0.0, le=1.0, description="BLEU-3 (trigram)")
    bleu_4: float = Field(ge=0.0, le=1.0, description="BLEU-4 (4-gram)")
    bleu_combined: float = Field(ge=0.0, le=1.0, description="Combined BLEU score")
    rouge_1: float = Field(ge=0.0, le=1.0, description="ROUGE-1 (unigram)")
    rouge_2: float = Field(ge=0.0, le=1.0, description="ROUGE-2 (bigram)")
    rouge_l: float = Field(ge=0.0, le=1.0, description="ROUGE-L (longest common subsequence)")
    rouge_combined: float = Field(ge=0.0, le=1.0, description="Combined ROUGE score")
    num_samples: int = Field(default=0, description="Number of samples evaluated")


class CombinedMetrics(BaseModel):
    """Combined retrieval and generation metrics."""

    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    overall_score: float = Field(ge=0.0, le=1.0, description="Overall evaluation score")


# ============================================================================
# MRR (Mean Reciprocal Rank)
# ============================================================================


class MRRCalculator:
    """
    Calculate Mean Reciprocal Rank for retrieval evaluation.

    MRR = (1/|Q|) * sum(1/rank_i) for all queries Q
    where rank_i is the rank of the first relevant document for query i.
    """

    def __init__(self, k_values: list[int] | None = None):
        """
        Initialize MRR calculator.

        Args:
            k_values: List of k values for MRR@k calculation
        """
        self.k_values = k_values or [1, 3, 5, 10, 20]

    def calculate(
        self,
        retrieved_docs: list[list[str]],
        relevant_docs: list[set[str]],
    ) -> tuple[float, dict[int, float]]:
        """
        Calculate MRR and MRR@k.

        Args:
            retrieved_docs: List of retrieved document IDs per query
            relevant_docs: List of sets of relevant document IDs per query

        Returns:
            Tuple of (MRR, dict of MRR@k values)
        """
        if len(retrieved_docs) != len(relevant_docs):
            raise ValueError("Mismatched number of queries")

        if not retrieved_docs:
            return 0.0, {k: 0.0 for k in self.k_values}

        reciprocal_ranks: list[float] = []
        mrr_at_k_sums: dict[int, float] = {k: 0.0 for k in self.k_values}

        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            rr = self._reciprocal_rank(retrieved, relevant)
            reciprocal_ranks.append(rr)

            # Calculate MRR@k
            for k in self.k_values:
                rr_at_k = self._reciprocal_rank(retrieved[:k], relevant)
                mrr_at_k_sums[k] += rr_at_k

        num_queries = len(retrieved_docs)
        mrr = sum(reciprocal_ranks) / num_queries

        mrr_at_k = {k: v / num_queries for k, v in mrr_at_k_sums.items()}

        logger.debug(f"MRR calculated | mrr={mrr:.4f} | queries={num_queries}")

        return round(mrr, 4), {k: round(v, 4) for k, v in mrr_at_k.items()}

    def _reciprocal_rank(
        self,
        retrieved: list[str],
        relevant: set[str],
    ) -> float:
        """Calculate reciprocal rank for a single query."""
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / rank
        return 0.0


# ============================================================================
# NDCG (Normalized Discounted Cumulative Gain)
# ============================================================================


class NDCGCalculator:
    """
    Calculate Normalized Discounted Cumulative Gain.

    NDCG = DCG / IDCG where:
    - DCG = sum(rel_i / log2(i + 1)) for position i
    - IDCG = ideal DCG (sorted by relevance)
    """

    def __init__(self, k_values: list[int] | None = None):
        """
        Initialize NDCG calculator.

        Args:
            k_values: List of k values for NDCG@k calculation
        """
        self.k_values = k_values or [1, 3, 5, 10, 20]

    def calculate(
        self,
        retrieved_docs: list[list[str]],
        relevance_scores: list[dict[str, float]],
    ) -> tuple[float, dict[int, float]]:
        """
        Calculate NDCG and NDCG@k.

        Args:
            retrieved_docs: List of retrieved document IDs per query
            relevance_scores: List of dicts mapping doc_id -> relevance score

        Returns:
            Tuple of (NDCG, dict of NDCG@k values)
        """
        if len(retrieved_docs) != len(relevance_scores):
            raise ValueError("Mismatched number of queries")

        if not retrieved_docs:
            return 0.0, {k: 0.0 for k in self.k_values}

        ndcg_scores: list[float] = []
        ndcg_at_k_sums: dict[int, float] = {k: 0.0 for k in self.k_values}

        for retrieved, scores in zip(retrieved_docs, relevance_scores):
            ndcg = self._ndcg(retrieved, scores)
            ndcg_scores.append(ndcg)

            for k in self.k_values:
                ndcg_k = self._ndcg(retrieved[:k], scores, k)
                ndcg_at_k_sums[k] += ndcg_k

        num_queries = len(retrieved_docs)
        avg_ndcg = sum(ndcg_scores) / num_queries

        ndcg_at_k = {k: v / num_queries for k, v in ndcg_at_k_sums.items()}

        logger.debug(f"NDCG calculated | ndcg={avg_ndcg:.4f} | queries={num_queries}")

        return round(avg_ndcg, 4), {k: round(v, 4) for k, v in ndcg_at_k.items()}

    def _dcg(self, retrieved: list[str], scores: dict[str, float], k: int | None = None) -> float:
        """Calculate Discounted Cumulative Gain."""
        if k:
            retrieved = retrieved[:k]

        dcg = 0.0
        for i, doc_id in enumerate(retrieved, start=1):
            rel = scores.get(doc_id, 0.0)
            dcg += rel / math.log2(i + 1)

        return dcg

    def _idcg(self, scores: dict[str, float], k: int | None = None) -> float:
        """Calculate Ideal DCG (best possible ranking)."""
        sorted_scores = sorted(scores.values(), reverse=True)
        if k:
            sorted_scores = sorted_scores[:k]

        idcg = 0.0
        for i, rel in enumerate(sorted_scores, start=1):
            idcg += rel / math.log2(i + 1)

        return idcg

    def _ndcg(
        self,
        retrieved: list[str],
        scores: dict[str, float],
        k: int | None = None,
    ) -> float:
        """Calculate Normalized DCG."""
        dcg = self._dcg(retrieved, scores, k)
        idcg = self._idcg(scores, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg


# ============================================================================
# BLEU (Bilingual Evaluation Understudy)
# ============================================================================


class BLEUCalculator:
    """
    Calculate BLEU score for text generation evaluation.

    BLEU = BP * exp(sum(w_n * log(p_n))) where:
    - BP = brevity penalty
    - p_n = modified n-gram precision
    - w_n = weight for n-gram (typically uniform)
    """

    def __init__(
        self,
        max_n: int = 4,
        weights: list[float] | None = None,
    ):
        """
        Initialize BLEU calculator.

        Args:
            max_n: Maximum n-gram order
            weights: Weights for each n-gram (default: uniform)
        """
        self.max_n = max_n
        self.weights = weights or [1.0 / max_n] * max_n

    def calculate(
        self,
        candidates: list[str],
        references: list[list[str]],
    ) -> GenerationMetrics:
        """
        Calculate BLEU scores for candidate texts.

        Args:
            candidates: List of candidate (generated) texts
            references: List of reference text lists (multiple references per candidate)

        Returns:
            GenerationMetrics with BLEU scores
        """
        if len(candidates) != len(references):
            raise ValueError("Mismatched number of candidates and references")

        if not candidates:
            return GenerationMetrics(
                bleu_1=0.0, bleu_2=0.0, bleu_3=0.0, bleu_4=0.0,
                bleu_combined=0.0, rouge_1=0.0, rouge_2=0.0, rouge_l=0.0,
                rouge_combined=0.0, num_samples=0,
            )

        # Calculate per-n-gram precision
        n_gram_precisions: dict[int, list[float]] = {n: [] for n in range(1, self.max_n + 1)}

        total_candidate_length = 0
        total_ref_length = 0

        for candidate, refs in zip(candidates, references):
            cand_tokens = self._tokenize(candidate)
            ref_tokens_list = [self._tokenize(ref) for ref in refs]

            total_candidate_length += len(cand_tokens)
            # Use closest reference length
            total_ref_length += min(
                len(rt) for rt in ref_tokens_list
            ) if ref_tokens_list else 0

            for n in range(1, self.max_n + 1):
                precision = self._modified_precision(cand_tokens, ref_tokens_list, n)
                n_gram_precisions[n].append(precision)

        # Average precisions
        avg_precisions = {}
        for n, precisions in n_gram_precisions.items():
            avg_precisions[n] = sum(precisions) / len(precisions) if precisions else 0.0

        # Calculate brevity penalty
        bp = self._brevity_penalty(total_candidate_length, total_ref_length)

        # Combined BLEU
        if all(p > 0 for p in avg_precisions.values()):
            log_sum = sum(
                w * math.log(avg_precisions[n + 1])
                for n, w in enumerate(self.weights)
                if n < len(avg_precisions)
            )
            bleu_combined = bp * math.exp(log_sum)
        else:
            bleu_combined = 0.0

        return GenerationMetrics(
            bleu_1=round(avg_precisions.get(1, 0.0), 4),
            bleu_2=round(avg_precisions.get(2, 0.0), 4),
            bleu_3=round(avg_precisions.get(3, 0.0), 4),
            bleu_4=round(avg_precisions.get(4, 0.0), 4),
            bleu_combined=round(bleu_combined, 4),
            rouge_1=0.0,
            rouge_2=0.0,
            rouge_l=0.0,
            rouge_combined=0.0,
            num_samples=len(candidates),
        )

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization."""
        return text.lower().split()

    def _get_ngrams(self, tokens: list[str], n: int) -> Counter:
        """Get n-gram counts."""
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        return Counter(ngrams)

    def _modified_precision(
        self,
        candidate_tokens: list[str],
        reference_tokens_list: list[list[str]],
        n: int,
    ) -> float:
        """Calculate modified n-gram precision."""
        cand_ngrams = self._get_ngrams(candidate_tokens, n)

        if not cand_ngrams:
            return 0.0

        # Get max count for each n-gram across all references
        max_ref_counts: Counter = Counter()
        for ref_tokens in reference_tokens_list:
            ref_ngrams = self._get_ngrams(ref_tokens, n)
            for ngram, count in ref_ngrams.items():
                max_ref_counts[ngram] = max(max_ref_counts[ngram], count)

        # Calculate clipped counts
        clipped_count = 0
        total_count = 0
        for ngram, count in cand_ngrams.items():
            clipped_count += min(count, max_ref_counts[ngram])
            total_count += count

        return clipped_count / total_count if total_count > 0 else 0.0

    def _brevity_penalty(
        self,
        candidate_length: int,
        reference_length: int,
    ) -> float:
        """Calculate brevity penalty."""
        if candidate_length > reference_length:
            return 1.0
        if candidate_length == 0:
            return 0.0
        return math.exp(1 - reference_length / candidate_length)


# ============================================================================
# ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
# ============================================================================


class ROUGECalculator:
    """
    Calculate ROUGE scores for summarization evaluation.

    ROUGE-N = overlap of n-grams between candidate and reference
    ROUGE-L = longest common subsequence based score
    """

    def calculate(
        self,
        candidates: list[str],
        references: list[str],
    ) -> GenerationMetrics:
        """
        Calculate ROUGE scores.

        Args:
            candidates: List of candidate texts
            references: List of reference texts

        Returns:
            GenerationMetrics with ROUGE scores
        """
        if len(candidates) != len(references):
            raise ValueError("Mismatched number of candidates and references")

        if not candidates:
            return GenerationMetrics(
                bleu_1=0.0, bleu_2=0.0, bleu_3=0.0, bleu_4=0.0,
                bleu_combined=0.0, rouge_1=0.0, rouge_2=0.0, rouge_l=0.0,
                rouge_combined=0.0, num_samples=0,
            )

        rouge_1_scores: list[float] = []
        rouge_2_scores: list[float] = []
        rouge_l_scores: list[float] = []

        for candidate, reference in zip(candidates, references):
            cand_tokens = self._tokenize(candidate)
            ref_tokens = self._tokenize(reference)

            r1 = self._rouge_n(cand_tokens, ref_tokens, 1)
            r2 = self._rouge_n(cand_tokens, ref_tokens, 2)
            rl = self._rouge_l(cand_tokens, ref_tokens)

            rouge_1_scores.append(r1)
            rouge_2_scores.append(r2)
            rouge_l_scores.append(rl)

        avg_rouge_1 = sum(rouge_1_scores) / len(rouge_1_scores)
        avg_rouge_2 = sum(rouge_2_scores) / len(rouge_2_scores)
        avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

        # Combined ROUGE (average of all)
        rouge_combined = (avg_rouge_1 + avg_rouge_2 + avg_rouge_l) / 3

        return GenerationMetrics(
            bleu_1=0.0,
            bleu_2=0.0,
            bleu_3=0.0,
            bleu_4=0.0,
            bleu_combined=0.0,
            rouge_1=round(avg_rouge_1, 4),
            rouge_2=round(avg_rouge_2, 4),
            rouge_l=round(avg_rouge_l, 4),
            rouge_combined=round(rouge_combined, 4),
            num_samples=len(candidates),
        )

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization."""
        return text.lower().split()

    def _get_ngrams(self, tokens: list[str], n: int) -> Counter:
        """Get n-gram counts."""
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        return Counter(ngrams)

    def _rouge_n(
        self,
        candidate_tokens: list[str],
        reference_tokens: list[str],
        n: int,
    ) -> float:
        """Calculate ROUGE-N F1 score."""
        cand_ngrams = self._get_ngrams(candidate_tokens, n)
        ref_ngrams = self._get_ngrams(reference_tokens, n)

        if not cand_ngrams or not ref_ngrams:
            return 0.0

        # Overlap
        overlap = sum((cand_ngrams & ref_ngrams).values())

        # Precision and recall
        precision = overlap / sum(cand_ngrams.values()) if cand_ngrams else 0.0
        recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0.0

        # F1 score
        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    def _rouge_l(
        self,
        candidate_tokens: list[str],
        reference_tokens: list[str],
    ) -> float:
        """Calculate ROUGE-L F1 score using LCS."""
        lcs_length = self._lcs_length(candidate_tokens, reference_tokens)

        if not candidate_tokens or not reference_tokens:
            return 0.0

        precision = lcs_length / len(candidate_tokens)
        recall = lcs_length / len(reference_tokens)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    def _lcs_length(self, seq1: list[str], seq2: list[str]) -> int:
        """Calculate length of Longest Common Subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]


# ============================================================================
# Precision and Recall at K
# ============================================================================


class PrecisionRecallCalculator:
    """Calculate precision and recall at various k values."""

    def __init__(self, k_values: list[int] | None = None):
        """
        Initialize calculator.

        Args:
            k_values: List of k values for metrics
        """
        self.k_values = k_values or [1, 3, 5, 10, 20]

    def calculate(
        self,
        retrieved_docs: list[list[str]],
        relevant_docs: list[set[str]],
    ) -> tuple[dict[int, float], dict[int, float]]:
        """
        Calculate precision@k and recall@k.

        Args:
            retrieved_docs: List of retrieved document IDs per query
            relevant_docs: List of sets of relevant document IDs

        Returns:
            Tuple of (precision@k dict, recall@k dict)
        """
        if len(retrieved_docs) != len(relevant_docs):
            raise ValueError("Mismatched number of queries")

        if not retrieved_docs:
            return (
                {k: 0.0 for k in self.k_values},
                {k: 0.0 for k in self.k_values},
            )

        precision_sums: dict[int, float] = {k: 0.0 for k in self.k_values}
        recall_sums: dict[int, float] = {k: 0.0 for k in self.k_values}

        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            for k in self.k_values:
                top_k = set(retrieved[:k])
                relevant_in_k = len(top_k & relevant)

                precision_sums[k] += relevant_in_k / k if k > 0 else 0.0
                recall_sums[k] += (
                    relevant_in_k / len(relevant) if relevant else 0.0
                )

        num_queries = len(retrieved_docs)

        precision_at_k = {k: round(v / num_queries, 4) for k, v in precision_sums.items()}
        recall_at_k = {k: round(v / num_queries, 4) for k, v in recall_sums.items()}

        return precision_at_k, recall_at_k


# ============================================================================
# Mean Average Precision (MAP)
# ============================================================================


class MAPCalculator:
    """Calculate Mean Average Precision."""

    def calculate(
        self,
        retrieved_docs: list[list[str]],
        relevant_docs: list[set[str]],
    ) -> float:
        """
        Calculate Mean Average Precision.

        Args:
            retrieved_docs: List of retrieved document IDs per query
            relevant_docs: List of sets of relevant document IDs

        Returns:
            MAP score
        """
        if len(retrieved_docs) != len(relevant_docs):
            raise ValueError("Mismatched number of queries")

        if not retrieved_docs:
            return 0.0

        average_precisions: list[float] = []

        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            ap = self._average_precision(retrieved, relevant)
            average_precisions.append(ap)

        return round(sum(average_precisions) / len(average_precisions), 4)

    def _average_precision(
        self,
        retrieved: list[str],
        relevant: set[str],
    ) -> float:
        """Calculate Average Precision for a single query."""
        if not relevant:
            return 0.0

        num_relevant_found = 0
        precision_sum = 0.0

        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                num_relevant_found += 1
                precision_at_rank = num_relevant_found / rank
                precision_sum += precision_at_rank

        if num_relevant_found == 0:
            return 0.0

        return precision_sum / len(relevant)


# ============================================================================
# Combined Metrics Calculator
# ============================================================================


class MetricsCalculator:
    """
    Combined calculator for all retrieval and generation metrics.

    Provides a unified interface for evaluating RAG systems.
    """

    def __init__(self, k_values: list[int] | None = None):
        """
        Initialize combined metrics calculator.

        Args:
            k_values: List of k values for retrieval metrics
        """
        self.k_values = k_values or [1, 3, 5, 10, 20]

        self.mrr_calc = MRRCalculator(k_values=self.k_values)
        self.ndcg_calc = NDCGCalculator(k_values=self.k_values)
        self.pr_calc = PrecisionRecallCalculator(k_values=self.k_values)
        self.map_calc = MAPCalculator()
        self.bleu_calc = BLEUCalculator()
        self.rouge_calc = ROUGECalculator()

        logger.info(f"MetricsCalculator initialized | k_values={self.k_values}")

    def evaluate_retrieval(
        self,
        retrieved_docs: list[list[str]],
        relevant_docs: list[set[str]],
        relevance_scores: list[dict[str, float]] | None = None,
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval quality.

        Args:
            retrieved_docs: List of retrieved document IDs per query
            relevant_docs: List of sets of relevant document IDs
            relevance_scores: Optional dict of doc_id -> relevance score for NDCG

        Returns:
            RetrievalMetrics with all retrieval metrics
        """
        # MRR
        mrr, mrr_at_k = self.mrr_calc.calculate(retrieved_docs, relevant_docs)

        # NDCG (use binary relevance if no scores provided)
        if relevance_scores is None:
            relevance_scores = [
                {doc: 1.0 for doc in docs} for docs in relevant_docs
            ]
        ndcg, ndcg_at_k = self.ndcg_calc.calculate(retrieved_docs, relevance_scores)

        # Precision and Recall
        precision_at_k, recall_at_k = self.pr_calc.calculate(
            retrieved_docs, relevant_docs
        )

        # MAP
        map_score = self.map_calc.calculate(retrieved_docs, relevant_docs)

        return RetrievalMetrics(
            mrr=mrr,
            mrr_at_k=mrr_at_k,
            ndcg=ndcg,
            ndcg_at_k=ndcg_at_k,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            map_score=map_score,
            num_queries=len(retrieved_docs),
        )

    def evaluate_generation(
        self,
        candidates: list[str],
        references: list[str] | list[list[str]],
    ) -> GenerationMetrics:
        """
        Evaluate generation quality.

        Args:
            candidates: List of generated texts
            references: List of reference texts (or list of reference lists for BLEU)

        Returns:
            GenerationMetrics with BLEU and ROUGE scores
        """
        # Convert to list of lists for BLEU if needed
        if references and isinstance(references[0], str):
            bleu_references: list[list[str]] = [[ref] for ref in references]
            rouge_references: list[str] = list(references)  # type: ignore
        else:
            bleu_references = list(references)  # type: ignore
            rouge_references = [refs[0] if refs else "" for refs in references]  # type: ignore

        # Calculate BLEU
        bleu_metrics = self.bleu_calc.calculate(candidates, bleu_references)

        # Calculate ROUGE
        rouge_metrics = self.rouge_calc.calculate(candidates, rouge_references)

        # Combine
        return GenerationMetrics(
            bleu_1=bleu_metrics.bleu_1,
            bleu_2=bleu_metrics.bleu_2,
            bleu_3=bleu_metrics.bleu_3,
            bleu_4=bleu_metrics.bleu_4,
            bleu_combined=bleu_metrics.bleu_combined,
            rouge_1=rouge_metrics.rouge_1,
            rouge_2=rouge_metrics.rouge_2,
            rouge_l=rouge_metrics.rouge_l,
            rouge_combined=rouge_metrics.rouge_combined,
            num_samples=len(candidates),
        )

    def evaluate_combined(
        self,
        retrieved_docs: list[list[str]],
        relevant_docs: list[set[str]],
        candidates: list[str],
        references: list[str],
        relevance_scores: list[dict[str, float]] | None = None,
        retrieval_weight: float = 0.5,
    ) -> CombinedMetrics:
        """
        Evaluate both retrieval and generation.

        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document sets
            candidates: Generated texts
            references: Reference texts
            relevance_scores: Optional relevance scores for NDCG
            retrieval_weight: Weight for retrieval in overall score

        Returns:
            CombinedMetrics with all metrics and overall score
        """
        retrieval = self.evaluate_retrieval(
            retrieved_docs, relevant_docs, relevance_scores
        )
        generation = self.evaluate_generation(candidates, references)

        # Overall score (weighted average of MRR and ROUGE-L)
        overall = (
            retrieval_weight * retrieval.mrr +
            (1 - retrieval_weight) * generation.rouge_l
        )

        return CombinedMetrics(
            retrieval=retrieval,
            generation=generation,
            overall_score=round(overall, 4),
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def calculate_mrr(
    retrieved_docs: list[list[str]],
    relevant_docs: list[set[str]],
) -> float:
    """
    Convenience function to calculate MRR.

    Args:
        retrieved_docs: List of retrieved document IDs per query
        relevant_docs: List of sets of relevant document IDs

    Returns:
        MRR score
    """
    calc = MRRCalculator()
    mrr, _ = calc.calculate(retrieved_docs, relevant_docs)
    return mrr


def calculate_ndcg(
    retrieved_docs: list[list[str]],
    relevance_scores: list[dict[str, float]],
) -> float:
    """
    Convenience function to calculate NDCG.

    Args:
        retrieved_docs: List of retrieved document IDs per query
        relevance_scores: List of dicts mapping doc_id -> relevance score

    Returns:
        NDCG score
    """
    calc = NDCGCalculator()
    ndcg, _ = calc.calculate(retrieved_docs, relevance_scores)
    return ndcg


def calculate_bleu(
    candidates: list[str],
    references: list[list[str]],
) -> float:
    """
    Convenience function to calculate BLEU-4.

    Args:
        candidates: List of candidate texts
        references: List of reference text lists

    Returns:
        BLEU-4 score
    """
    calc = BLEUCalculator()
    metrics = calc.calculate(candidates, references)
    return metrics.bleu_combined


def calculate_rouge(
    candidates: list[str],
    references: list[str],
) -> float:
    """
    Convenience function to calculate ROUGE-L.

    Args:
        candidates: List of candidate texts
        references: List of reference texts

    Returns:
        ROUGE-L score
    """
    calc = ROUGECalculator()
    metrics = calc.calculate(candidates, references)
    return metrics.rouge_l
