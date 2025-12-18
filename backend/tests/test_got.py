"""Comprehensive tests for Graph of Thoughts (GoT) implementation.

Tests cover:
- Graph building and expansion
- Node merging (shared sub-problems)
- Path extraction and scoring
- LLM error handling
- Edge cases and failure modes
"""

from unittest.mock import AsyncMock, Mock

import pytest

from backend.src.models.reasoning import ThoughtGraph, ThoughtNode, ThoughtType
from backend.src.reasoning.got import GraphOfThoughts


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = AsyncMock()
    # Default response for root generation
    llm.ainvoke.return_value = Mock(content="Sub-problem 1: X\nSub-problem 2: Y")
    return llm


@pytest.fixture
def got_reasoner(mock_llm):
    """Create a GoT reasoner with mocked LLM."""
    return GraphOfThoughts(llm=mock_llm, max_iterations=2, similarity_threshold=0.7)


class TestGraphOfThoughtsInitialization:
    """Test GoT initialization and configuration."""

    def test_initialization_defaults(self, mock_llm):
        """Test GoT initializes with correct defaults."""
        got = GraphOfThoughts(llm=mock_llm)
        assert got.llm == mock_llm
        assert got.max_iterations == 5
        assert got.similarity_threshold == 0.7

    def test_initialization_custom_params(self, mock_llm):
        """Test GoT initializes with custom parameters."""
        got = GraphOfThoughts(llm=mock_llm, max_iterations=3, similarity_threshold=0.8)
        assert got.max_iterations == 3
        assert got.similarity_threshold == 0.8

    def test_initialization_clamps_max_iterations(self, mock_llm):
        """Test max_iterations is clamped to [2, 10]."""
        got_low = GraphOfThoughts(llm=mock_llm, max_iterations=1)
        assert got_low.max_iterations == 2

        got_high = GraphOfThoughts(llm=mock_llm, max_iterations=100)
        assert got_high.max_iterations == 10

    def test_initialization_clamps_similarity_threshold(self, mock_llm):
        """Test similarity_threshold is clamped to [0.0, 1.0]."""
        got_low = GraphOfThoughts(llm=mock_llm, similarity_threshold=-0.5)
        assert got_low.similarity_threshold == 0.0

        got_high = GraphOfThoughts(llm=mock_llm, similarity_threshold=1.5)
        assert got_high.similarity_threshold == 1.0


class TestRootGeneration:
    """Test root node generation."""

    @pytest.mark.asyncio
    async def test_generate_root_success(self, got_reasoner, mock_llm):
        """Test successful root node generation."""
        mock_llm.ainvoke.return_value = Mock(content="Decompose into steps A, B, C")

        root = await got_reasoner._generate_root("Test query")

        assert root.node_id == "root"
        assert root.parent_id is None
        assert root.thought_type == ThoughtType.DECOMPOSITION
        assert "Decompose" in root.content
        assert root.confidence_score == 1.0
        assert root.depth == 0

    @pytest.mark.asyncio
    async def test_generate_root_llm_failure(self, got_reasoner, mock_llm):
        """Test root generation handles LLM failures gracefully."""
        mock_llm.ainvoke.side_effect = Exception("LLM API error")

        with pytest.raises(RuntimeError, match="LLM root generation failed"):
            await got_reasoner._generate_root("Test query")


class TestThoughtGeneration:
    """Test child thought generation."""

    @pytest.mark.asyncio
    async def test_generate_thoughts_success(self, got_reasoner, mock_llm):
        """Test successful thought generation."""
        parent = ThoughtNode(
            node_id="parent_1",
            parent_id="root",
            thought_type=ThoughtType.ANALYSIS,
            content="Parent thought",
            depth=1,
        )

        mock_llm.ainvoke.return_value = Mock(
            content="1. First thought\n2. Second thought\n3. Third thought"
        )

        thoughts = await got_reasoner._generate_thoughts(parent, num=3)

        assert len(thoughts) == 3
        assert all(t.parent_id == "parent_1" for t in thoughts)
        assert all(t.depth == 2 for t in thoughts)
        assert thoughts[0].content == "First thought"
        assert thoughts[1].content == "Second thought"

    @pytest.mark.asyncio
    async def test_generate_thoughts_llm_failure(self, got_reasoner, mock_llm):
        """Test thought generation handles LLM failures."""
        parent = ThoughtNode(
            node_id="parent_1",
            parent_id="root",
            thought_type=ThoughtType.ANALYSIS,
            content="Parent thought",
            depth=1,
        )

        mock_llm.ainvoke.side_effect = Exception("LLM API error")

        with pytest.raises(RuntimeError, match="LLM thought generation failed"):
            await got_reasoner._generate_thoughts(parent, num=3)


class TestSimilarityCalculation:
    """Test text similarity calculation."""

    def test_calculate_similarity_identical(self, got_reasoner):
        """Test similarity for identical texts."""
        text = "The quick brown fox jumps over the lazy dog"
        similarity = got_reasoner._calculate_similarity(text, text)
        assert similarity == 1.0

    def test_calculate_similarity_completely_different(self, got_reasoner):
        """Test similarity for completely different texts."""
        text1 = "weather forecast rain"
        text2 = "mathematics algebra equations"
        similarity = got_reasoner._calculate_similarity(text1, text2)
        assert similarity == 0.0

    def test_calculate_similarity_partial_overlap(self, got_reasoner):
        """Test similarity for partially overlapping texts."""
        text1 = "weather forecast rain wind"
        text2 = "weather rain today"
        similarity = got_reasoner._calculate_similarity(text1, text2)
        assert 0.0 < similarity < 1.0

    def test_calculate_similarity_case_insensitive(self, got_reasoner):
        """Test similarity is case-insensitive."""
        text1 = "WEATHER FORECAST"
        text2 = "weather forecast"
        similarity = got_reasoner._calculate_similarity(text1, text2)
        assert similarity == 1.0

    def test_calculate_similarity_empty_strings(self, got_reasoner):
        """Test similarity with empty strings."""
        similarity = got_reasoner._calculate_similarity("", "test")
        assert similarity == 0.0


class TestNodeMerging:
    """Test node merging (shared sub-problems)."""

    def test_find_similar_node_found(self, got_reasoner):
        """Test finding similar node in graph."""
        graph = ThoughtGraph(
            nodes=[
                ThoughtNode(node_id="node_1", thought_type=ThoughtType.ANALYSIS, content="analyze weather patterns", depth=1),
                ThoughtNode(node_id="node_2", thought_type=ThoughtType.ANALYSIS, content="study temperature trends", depth=1),
            ],
            edges=[],
            merge_nodes=[],
        )

        new_thought = ThoughtNode(
            node_id="node_3",
            thought_type=ThoughtType.ANALYSIS,
            content="analyze weather patterns and trends",  # Very similar to node_1
            depth=1,
        )

        similar_id = got_reasoner._find_similar_node(graph, new_thought)
        assert similar_id == "node_1"

    def test_find_similar_node_not_found(self, got_reasoner):
        """Test when no similar node exists."""
        graph = ThoughtGraph(
            nodes=[
                ThoughtNode(node_id="node_1", thought_type=ThoughtType.ANALYSIS, content="mathematics algebra", depth=1),
            ],
            edges=[],
            merge_nodes=[],
        )

        new_thought = ThoughtNode(
            node_id="node_2",
            thought_type=ThoughtType.ANALYSIS,
            content="weather forecast rain",  # Completely different
            depth=1,
        )

        similar_id = got_reasoner._find_similar_node(graph, new_thought)
        assert similar_id is None


class TestPathExtraction:
    """Test path extraction and scoring."""

    def test_get_leaf_node_ids(self, got_reasoner):
        """Test identifying leaf nodes (no outgoing edges)."""
        graph = ThoughtGraph(
            nodes=[
                ThoughtNode(node_id="root", thought_type=ThoughtType.DECOMPOSITION, content="Root node", depth=0),
                ThoughtNode(node_id="node_1", thought_type=ThoughtType.ANALYSIS, content="Node 1", depth=1),
                ThoughtNode(node_id="node_2", thought_type=ThoughtType.ANALYSIS, content="Node 2", depth=1),
                ThoughtNode(node_id="leaf_1", thought_type=ThoughtType.SYNTHESIS, content="Leaf 1", depth=2),
                ThoughtNode(node_id="leaf_2", thought_type=ThoughtType.SYNTHESIS, content="Leaf 2", depth=2),
            ],
            edges=[
                {"source": "root", "target": "node_1"},
                {"source": "root", "target": "node_2"},
                {"source": "node_1", "target": "leaf_1"},
                {"source": "node_2", "target": "leaf_2"},
            ],
            merge_nodes=[],
        )

        leaf_ids = got_reasoner._get_leaf_node_ids(graph)
        assert set(leaf_ids) == {"leaf_1", "leaf_2"}

    def test_score_path(self, got_reasoner):
        """Test path scoring algorithm."""
        path = [
            ThoughtNode(node_id="root", thought_type=ThoughtType.DECOMPOSITION, content="Root", confidence_score=1.0, depth=0),
            ThoughtNode(node_id="node_1", thought_type=ThoughtType.ANALYSIS, content="Node 1", confidence_score=0.8, depth=1),
            ThoughtNode(node_id="node_2", thought_type=ThoughtType.SYNTHESIS, content="Node 2", confidence_score=0.9, depth=2),
        ]

        score = got_reasoner._score_path(path)
        assert 0.0 <= score <= 1.0

    def test_score_path_empty(self, got_reasoner):
        """Test scoring empty path."""
        score = got_reasoner._score_path([])
        assert score == 0.0


class TestGraphBuilding:
    """Test complete graph building process."""

    @pytest.mark.asyncio
    async def test_build_graph_basic(self, got_reasoner, mock_llm):
        """Test basic graph building."""
        # Mock responses for root, thoughts, and synthesis
        mock_llm.ainvoke.side_effect = [
            Mock(content="Root: Break down problem"),  # Root
            Mock(content="1. Step A\n2. Step B"),  # First expansion
            Mock(content="1. Step C\n2. Step D"),  # Second expansion
            Mock(content="Final answer based on reasoning"),  # Synthesis
        ]

        result = await got_reasoner.build_graph("Test query")

        assert result.reasoning_type == "got"
        assert len(result.best_path) > 0
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.final_answer is not None
        assert "Final answer" in result.final_answer


class TestSynthesis:
    """Test answer synthesis."""

    @pytest.mark.asyncio
    async def test_synthesize_answer_success(self, got_reasoner, mock_llm):
        """Test successful answer synthesis."""
        path = [
            ThoughtNode(node_id="root", thought_type=ThoughtType.DECOMPOSITION, content="Decompose problem", depth=0),
            ThoughtNode(node_id="node_1", thought_type=ThoughtType.ANALYSIS, content="Analyze component A", depth=1),
            ThoughtNode(node_id="node_2", thought_type=ThoughtType.SYNTHESIS, content="Analyze component B", depth=2),
        ]

        graph = ThoughtGraph(nodes=path, edges=[], merge_nodes=[])

        mock_llm.ainvoke.return_value = Mock(content="Synthesized final answer")

        answer = await got_reasoner._synthesize_answer(path, graph)

        assert answer == "Synthesized final answer"

    @pytest.mark.asyncio
    async def test_synthesize_answer_llm_failure(self, got_reasoner, mock_llm):
        """Test synthesis handles LLM failures."""
        path = [ThoughtNode(node_id="root", thought_type=ThoughtType.DECOMPOSITION, content="Test", depth=0)]
        graph = ThoughtGraph(nodes=path, edges=[], merge_nodes=[])

        mock_llm.ainvoke.side_effect = Exception("LLM API error")

        with pytest.raises(RuntimeError, match="LLM synthesis failed"):
            await got_reasoner._synthesize_answer(path, graph)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_build_graph_no_children(self, got_reasoner, mock_llm):
        """Test graph building when no child thoughts are generated."""
        mock_llm.ainvoke.side_effect = [
            Mock(content="Root node"),  # Root
            Mock(content=""),  # Empty thoughts
            Mock(content="Final answer"),  # Synthesis
        ]

        result = await got_reasoner.build_graph("Test query")

        assert result.reasoning_type == "got"
        assert len(result.best_path) >= 1  # At least root

    def test_get_node_by_id_found(self, got_reasoner):
        """Test getting node by ID when it exists."""
        graph = ThoughtGraph(
            nodes=[
                ThoughtNode(node_id="test_id", thought_type=ThoughtType.ANALYSIS, content="Test node", depth=0),
            ],
            edges=[],
            merge_nodes=[],
        )

        node = got_reasoner._get_node_by_id(graph, "test_id")
        assert node is not None
        assert node.node_id == "test_id"

    def test_get_node_by_id_not_found(self, got_reasoner):
        """Test getting node by ID when it doesn't exist."""
        graph = ThoughtGraph(nodes=[], edges=[], merge_nodes=[])

        node = got_reasoner._get_node_by_id(graph, "nonexistent_id")
        assert node is None
