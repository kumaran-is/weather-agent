"""
Tests for Retrieval and Generation Metrics Module.

Level 6c: Self-Evolving Platform

Tests:
1. MRR calculation
2. NDCG calculation
3. BLEU scores
4. ROUGE scores
5. Precision/Recall@k
6. MAP
7. Combined metrics
"""

import pytest

from backend.src.evaluation.retrieval_metrics import (
    BLEUCalculator,
    CombinedMetrics,
    GenerationMetrics,
    MAPCalculator,
    MetricsCalculator,
    MRRCalculator,
    NDCGCalculator,
    PrecisionRecallCalculator,
    RetrievalMetrics,
    ROUGECalculator,
    calculate_bleu,
    calculate_mrr,
    calculate_ndcg,
    calculate_rouge,
)


class TestMRRCalculator:
    """Test MRR calculation."""

    @pytest.fixture
    def calculator(self):
        return MRRCalculator(k_values=[1, 3, 5])

    def test_perfect_mrr(self, calculator):
        """Test MRR when first result is always relevant."""
        retrieved = [["doc1", "doc2", "doc3"], ["doc4", "doc5", "doc6"]]
        relevant = [{"doc1"}, {"doc4"}]

        mrr, mrr_at_k = calculator.calculate(retrieved, relevant)

        assert mrr == 1.0
        assert mrr_at_k[1] == 1.0

    def test_second_position_mrr(self, calculator):
        """Test MRR when relevant doc is in second position."""
        retrieved = [["doc2", "doc1", "doc3"]]
        relevant = [{"doc1"}]

        mrr, mrr_at_k = calculator.calculate(retrieved, relevant)

        assert mrr == 0.5  # 1/2

    def test_third_position_mrr(self, calculator):
        """Test MRR when relevant doc is in third position."""
        retrieved = [["doc2", "doc3", "doc1"]]
        relevant = [{"doc1"}]

        mrr, mrr_at_k = calculator.calculate(retrieved, relevant)

        assert abs(mrr - 0.3333) < 0.01  # 1/3

    def test_no_relevant_found(self, calculator):
        """Test MRR when no relevant docs are found."""
        retrieved = [["doc2", "doc3", "doc4"]]
        relevant = [{"doc1"}]

        mrr, _ = calculator.calculate(retrieved, relevant)

        assert mrr == 0.0

    def test_multiple_queries(self, calculator):
        """Test MRR with multiple queries."""
        retrieved = [
            ["doc1", "doc2"],  # rank 1
            ["doc3", "doc4", "doc5"],  # rank 3
        ]
        relevant = [{"doc1"}, {"doc5"}]

        mrr, _ = calculator.calculate(retrieved, relevant)

        # (1/1 + 1/3) / 2 = 0.667
        assert abs(mrr - 0.6667) < 0.01

    def test_empty_input(self, calculator):
        """Test with empty input."""
        mrr, mrr_at_k = calculator.calculate([], [])

        assert mrr == 0.0
        assert all(v == 0.0 for v in mrr_at_k.values())


class TestNDCGCalculator:
    """Test NDCG calculation."""

    @pytest.fixture
    def calculator(self):
        return NDCGCalculator(k_values=[1, 3, 5])

    def test_perfect_ndcg(self, calculator):
        """Test NDCG with perfect ranking."""
        retrieved = [["doc1", "doc2", "doc3"]]
        relevance = [{"doc1": 3.0, "doc2": 2.0, "doc3": 1.0}]

        ndcg, _ = calculator.calculate(retrieved, relevance)

        assert ndcg == 1.0

    def test_inverted_ranking(self, calculator):
        """Test NDCG with inverted ranking."""
        retrieved = [["doc3", "doc2", "doc1"]]
        relevance = [{"doc1": 3.0, "doc2": 2.0, "doc3": 1.0}]

        ndcg, _ = calculator.calculate(retrieved, relevance)

        assert ndcg < 1.0  # Not perfect

    def test_no_relevant_docs(self, calculator):
        """Test NDCG with no relevant docs in ranking."""
        retrieved = [["doc4", "doc5", "doc6"]]
        relevance = [{"doc1": 3.0, "doc2": 2.0}]

        ndcg, _ = calculator.calculate(retrieved, relevance)

        assert ndcg == 0.0

    def test_partial_relevance(self, calculator):
        """Test NDCG with partial relevance."""
        retrieved = [["doc1", "doc4", "doc2"]]  # doc4 is not relevant
        relevance = [{"doc1": 3.0, "doc2": 2.0, "doc3": 1.0}]

        ndcg, _ = calculator.calculate(retrieved, relevance)

        assert 0.0 < ndcg < 1.0

    def test_ndcg_at_k(self, calculator):
        """Test NDCG@k calculation."""
        retrieved = [["doc1", "doc2", "doc3", "doc4", "doc5"]]
        relevance = [{"doc1": 1.0, "doc3": 1.0, "doc5": 1.0}]

        _, ndcg_at_k = calculator.calculate(retrieved, relevance)

        assert 1 in ndcg_at_k
        assert 3 in ndcg_at_k
        assert 5 in ndcg_at_k


class TestBLEUCalculator:
    """Test BLEU score calculation."""

    @pytest.fixture
    def calculator(self):
        return BLEUCalculator()

    def test_identical_text(self, calculator):
        """Test BLEU with identical text."""
        candidates = ["the weather is sunny today"]
        references = [["the weather is sunny today"]]

        metrics = calculator.calculate(candidates, references)

        assert metrics.bleu_1 == 1.0
        assert metrics.bleu_2 == 1.0

    def test_partial_match(self, calculator):
        """Test BLEU with partial match."""
        candidates = ["the weather is sunny"]
        references = [["the weather is cloudy"]]

        metrics = calculator.calculate(candidates, references)

        # Should have some match (3/4 unigrams match)
        assert 0.0 < metrics.bleu_1 < 1.0

    def test_no_match(self, calculator):
        """Test BLEU with no matching words."""
        candidates = ["hello world"]
        references = [["goodbye universe"]]

        metrics = calculator.calculate(candidates, references)

        assert metrics.bleu_1 == 0.0

    def test_multiple_references(self, calculator):
        """Test BLEU with multiple references."""
        candidates = ["the cat sat on the mat"]
        references = [["the cat is on the mat", "a cat sat on the rug"]]

        metrics = calculator.calculate(candidates, references)

        assert metrics.bleu_1 > 0.5

    def test_brevity_penalty(self, calculator):
        """Test brevity penalty for short candidates."""
        candidates = ["sunny"]
        references = [["the weather is sunny today"]]

        metrics = calculator.calculate(candidates, references)

        # Brevity penalty should reduce score
        assert metrics.bleu_combined < 0.5

    def test_empty_input(self, calculator):
        """Test with empty input."""
        metrics = calculator.calculate([], [])

        assert metrics.bleu_1 == 0.0
        assert metrics.num_samples == 0


class TestROUGECalculator:
    """Test ROUGE score calculation."""

    @pytest.fixture
    def calculator(self):
        return ROUGECalculator()

    def test_identical_text(self, calculator):
        """Test ROUGE with identical text."""
        candidates = ["the weather is sunny today"]
        references = ["the weather is sunny today"]

        metrics = calculator.calculate(candidates, references)

        assert metrics.rouge_1 == 1.0
        assert metrics.rouge_l == 1.0

    def test_partial_overlap(self, calculator):
        """Test ROUGE with partial overlap."""
        candidates = ["the weather is sunny today"]
        references = ["the weather is cloudy today"]

        metrics = calculator.calculate(candidates, references)

        # 4/5 unigram overlap
        assert 0.7 < metrics.rouge_1 < 1.0

    def test_no_overlap(self, calculator):
        """Test ROUGE with no overlap."""
        candidates = ["hello world"]
        references = ["goodbye universe"]

        metrics = calculator.calculate(candidates, references)

        assert metrics.rouge_1 == 0.0
        assert metrics.rouge_l == 0.0

    def test_rouge_l_subsequence(self, calculator):
        """Test ROUGE-L with longer common subsequence."""
        candidates = ["the big red cat"]
        references = ["the big red dog"]

        metrics = calculator.calculate(candidates, references)

        # LCS is "the big red" = 3 words
        assert metrics.rouge_l > 0.5

    def test_empty_input(self, calculator):
        """Test with empty input."""
        metrics = calculator.calculate([], [])

        assert metrics.rouge_1 == 0.0
        assert metrics.num_samples == 0


class TestPrecisionRecallCalculator:
    """Test Precision and Recall@k calculation."""

    @pytest.fixture
    def calculator(self):
        return PrecisionRecallCalculator(k_values=[1, 3, 5])

    def test_perfect_precision(self, calculator):
        """Test perfect precision."""
        retrieved = [["doc1", "doc2", "doc3"]]
        relevant = [{"doc1", "doc2", "doc3"}]

        precision, recall = calculator.calculate(retrieved, relevant)

        assert precision[3] == 1.0
        assert recall[3] == 1.0

    def test_low_precision(self, calculator):
        """Test low precision (many irrelevant docs)."""
        retrieved = [["doc1", "doc2", "doc3", "doc4", "doc5"]]
        relevant = [{"doc1"}]

        precision, recall = calculator.calculate(retrieved, relevant)

        assert precision[5] == 0.2  # 1/5
        assert recall[5] == 1.0  # Found the 1 relevant doc

    def test_low_recall(self, calculator):
        """Test low recall (missing relevant docs)."""
        retrieved = [["doc1"]]
        relevant = [{"doc1", "doc2", "doc3", "doc4", "doc5"}]

        precision, recall = calculator.calculate(retrieved, relevant)

        assert precision[1] == 1.0  # 1/1
        assert recall[1] == 0.2  # Found 1/5

    def test_varying_k(self, calculator):
        """Test precision/recall at different k values."""
        retrieved = [["doc1", "doc2", "doc3", "doc4", "doc5"]]
        relevant = [{"doc1", "doc2", "doc3"}]

        precision, recall = calculator.calculate(retrieved, relevant)

        # Precision decreases as k increases (more irrelevant docs)
        assert precision[1] >= precision[3] >= precision[5]
        # Recall increases as k increases (more chance to find relevant)
        assert recall[1] <= recall[3] <= recall[5]


class TestMAPCalculator:
    """Test Mean Average Precision calculation."""

    @pytest.fixture
    def calculator(self):
        return MAPCalculator()

    def test_perfect_map(self, calculator):
        """Test perfect MAP."""
        retrieved = [["doc1", "doc2", "doc3"]]
        relevant = [{"doc1", "doc2", "doc3"}]

        map_score = calculator.calculate(retrieved, relevant)

        assert map_score == 1.0

    def test_partial_map(self, calculator):
        """Test partial MAP."""
        retrieved = [["doc1", "doc4", "doc2", "doc5", "doc3"]]
        relevant = [{"doc1", "doc2", "doc3"}]

        map_score = calculator.calculate(retrieved, relevant)

        # AP = (1/1 + 2/3 + 3/5) / 3 = 0.756
        assert 0.7 < map_score < 0.8

    def test_no_relevant_found(self, calculator):
        """Test MAP when no relevant docs found."""
        retrieved = [["doc4", "doc5", "doc6"]]
        relevant = [{"doc1", "doc2", "doc3"}]

        map_score = calculator.calculate(retrieved, relevant)

        assert map_score == 0.0

    def test_multiple_queries(self, calculator):
        """Test MAP with multiple queries."""
        retrieved = [
            ["doc1", "doc2"],  # Perfect
            ["doc3", "doc4", "doc5"],  # 1 relevant at rank 3
        ]
        relevant = [{"doc1", "doc2"}, {"doc5"}]

        map_score = calculator.calculate(retrieved, relevant)

        # Query 1: (1/1 + 2/2) / 2 = 1.0
        # Query 2: (1/3) / 1 = 0.333
        # Average: 0.667
        assert 0.6 < map_score < 0.7


class TestMetricsCalculator:
    """Test combined MetricsCalculator."""

    @pytest.fixture
    def calculator(self):
        return MetricsCalculator(k_values=[1, 3, 5])

    def test_evaluate_retrieval(self, calculator):
        """Test retrieval evaluation."""
        retrieved = [
            ["doc1", "doc2", "doc3"],
            ["doc4", "doc5", "doc6"],
        ]
        relevant = [{"doc1", "doc2"}, {"doc5", "doc6"}]

        result = calculator.evaluate_retrieval(retrieved, relevant)

        assert isinstance(result, RetrievalMetrics)
        assert 0.0 <= result.mrr <= 1.0
        assert 0.0 <= result.ndcg <= 1.0
        assert result.num_queries == 2

    def test_evaluate_generation(self, calculator):
        """Test generation evaluation."""
        candidates = [
            "the weather is sunny today",
            "expect rain tomorrow",
        ]
        references = [
            "the weather is sunny today",
            "rain is expected tomorrow",
        ]

        result = calculator.evaluate_generation(candidates, references)

        assert isinstance(result, GenerationMetrics)
        assert 0.0 <= result.bleu_combined <= 1.0
        assert 0.0 <= result.rouge_l <= 1.0
        assert result.num_samples == 2

    def test_evaluate_combined(self, calculator):
        """Test combined evaluation."""
        retrieved = [["doc1", "doc2"]]
        relevant = [{"doc1"}]
        candidates = ["sunny weather today"]
        references = ["sunny weather today"]

        result = calculator.evaluate_combined(
            retrieved, relevant, candidates, references
        )

        assert isinstance(result, CombinedMetrics)
        assert 0.0 <= result.overall_score <= 1.0

    def test_evaluate_with_relevance_scores(self, calculator):
        """Test retrieval with custom relevance scores."""
        retrieved = [["doc1", "doc2", "doc3"]]
        relevant = [{"doc1", "doc2"}]
        relevance_scores = [{"doc1": 3.0, "doc2": 2.0, "doc3": 1.0}]

        result = calculator.evaluate_retrieval(
            retrieved, relevant, relevance_scores
        )

        assert result.ndcg > 0.0


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_calculate_mrr(self):
        """Test calculate_mrr convenience function."""
        retrieved = [["doc1", "doc2"]]
        relevant = [{"doc1"}]

        mrr = calculate_mrr(retrieved, relevant)

        assert mrr == 1.0

    def test_calculate_ndcg(self):
        """Test calculate_ndcg convenience function."""
        retrieved = [["doc1", "doc2"]]
        relevance = [{"doc1": 2.0, "doc2": 1.0}]

        ndcg = calculate_ndcg(retrieved, relevance)

        assert ndcg == 1.0

    def test_calculate_bleu(self):
        """Test calculate_bleu convenience function."""
        candidates = ["the weather is nice"]
        references = [["the weather is nice"]]

        bleu = calculate_bleu(candidates, references)

        assert bleu > 0.9

    def test_calculate_rouge(self):
        """Test calculate_rouge convenience function."""
        candidates = ["the weather is nice"]
        references = ["the weather is nice"]

        rouge = calculate_rouge(candidates, references)

        assert rouge == 1.0


class TestMetricsModels:
    """Test Pydantic models."""

    def test_retrieval_metrics_model(self):
        """Test RetrievalMetrics model."""
        metrics = RetrievalMetrics(
            mrr=0.8,
            ndcg=0.75,
            mrr_at_k={1: 0.9, 3: 0.85},
            ndcg_at_k={1: 0.8, 3: 0.78},
            precision_at_k={1: 0.9, 3: 0.8},
            recall_at_k={1: 0.3, 3: 0.6},
            map_score=0.82,
            num_queries=100,
        )

        assert metrics.mrr == 0.8
        assert metrics.ndcg == 0.75
        assert metrics.num_queries == 100

    def test_generation_metrics_model(self):
        """Test GenerationMetrics model."""
        metrics = GenerationMetrics(
            bleu_1=0.9,
            bleu_2=0.85,
            bleu_3=0.8,
            bleu_4=0.75,
            bleu_combined=0.82,
            rouge_1=0.88,
            rouge_2=0.82,
            rouge_l=0.85,
            rouge_combined=0.85,
            num_samples=50,
        )

        assert metrics.bleu_combined == 0.82
        assert metrics.rouge_l == 0.85
        assert metrics.num_samples == 50

    def test_combined_metrics_model(self):
        """Test CombinedMetrics model."""
        retrieval = RetrievalMetrics(mrr=0.8, ndcg=0.75)
        generation = GenerationMetrics(
            bleu_1=0.9, bleu_2=0.85, bleu_3=0.8, bleu_4=0.75,
            bleu_combined=0.82, rouge_1=0.88, rouge_2=0.82,
            rouge_l=0.85, rouge_combined=0.85,
        )

        combined = CombinedMetrics(
            retrieval=retrieval,
            generation=generation,
            overall_score=0.825,
        )

        assert combined.retrieval.mrr == 0.8
        assert combined.generation.rouge_l == 0.85
        assert combined.overall_score == 0.825
