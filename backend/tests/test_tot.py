"""Comprehensive tests for Tree of Thoughts (ToT) implementation.

Tests cover:
- Tree exploration (BFS)
- Thought generation and evaluation
- Path extraction and scoring
- LLM error handling
- Edge cases and boundary conditions
"""

import pytest
from unittest.mock import AsyncMock, Mock
from backend.src.reasoning.tot import TreeOfThoughts
from backend.src.models.reasoning import ThoughtNode, ThoughtTree, ThoughtType


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = AsyncMock()
    # Default response for root generation
    llm.ainvoke.return_value = Mock(content="Decompose into sub-tasks A, B, C")
    return llm


@pytest.fixture
def tot_reasoner(mock_llm):
    """Create a ToT reasoner with mocked LLM."""
    return TreeOfThoughts(llm=mock_llm, max_depth=2, breadth=2)


class TestTreeOfThoughtsInitialization:
    """Test ToT initialization and configuration."""

    def test_initialization_defaults(self, mock_llm):
        """Test ToT initializes with correct defaults."""
        tot = TreeOfThoughts(llm=mock_llm)
        assert tot.llm == mock_llm
        assert tot.max_depth == 3
        assert tot.breadth == 3

    def test_initialization_custom_params(self, mock_llm):
        """Test ToT initializes with custom parameters."""
        tot = TreeOfThoughts(llm=mock_llm, max_depth=4, breadth=4)
        assert tot.max_depth == 4
        assert tot.breadth == 4

    def test_initialization_clamps_max_depth(self, mock_llm):
        """Test max_depth is clamped to [1, 5]."""
        tot_low = TreeOfThoughts(llm=mock_llm, max_depth=0)
        assert tot_low.max_depth == 1

        tot_high = TreeOfThoughts(llm=mock_llm, max_depth=100)
        assert tot_high.max_depth == 5

    def test_initialization_clamps_breadth(self, mock_llm):
        """Test breadth is clamped to [2, 5]."""
        tot_low = TreeOfThoughts(llm=mock_llm, breadth=1)
        assert tot_low.breadth == 2

        tot_high = TreeOfThoughts(llm=mock_llm, breadth=100)
        assert tot_high.breadth == 5


class TestRootGeneration:
    """Test root thought generation."""

    @pytest.mark.asyncio
    async def test_generate_root_success(self, tot_reasoner, mock_llm):
        """Test successful root thought generation."""
        mock_llm.ainvoke.return_value = Mock(content="Break into steps: A, B, C")

        root = await tot_reasoner._generate_root("Test query")

        assert root.node_id == "root"
        assert root.parent_id is None
        assert root.thought_type == ThoughtType.DECOMPOSITION
        assert "Break into steps" in root.content
        assert root.confidence_score == 1.0
        assert root.depth == 0

    @pytest.mark.asyncio
    async def test_generate_root_llm_failure(self, tot_reasoner, mock_llm):
        """Test root generation handles LLM failures gracefully."""
        mock_llm.ainvoke.side_effect = Exception("LLM API error")

        with pytest.raises(RuntimeError, match="LLM root generation failed"):
            await tot_reasoner._generate_root("Test query")


class TestThoughtGeneration:
    """Test alternative thought generation."""

    @pytest.mark.asyncio
    async def test_generate_thoughts_success(self, tot_reasoner, mock_llm):
        """Test successful alternative thought generation."""
        parent = ThoughtNode(
            node_id="parent_1",
            parent_id="root",
            thought_type=ThoughtType.ANALYSIS,
            content="Parent thought",
            depth=1,
        )

        mock_llm.ainvoke.return_value = Mock(
            content="1. Alternative A\n2. Alternative B\n3. Alternative C"
        )

        thoughts = await tot_reasoner._generate_thoughts(parent, num=2)

        assert len(thoughts) == 2
        assert all(t.parent_id == "parent_1" for t in thoughts)
        assert all(t.depth == 2 for t in thoughts)
        assert thoughts[0].content == "Alternative A"
        assert thoughts[1].content == "Alternative B"

    @pytest.mark.asyncio
    async def test_generate_thoughts_llm_failure(self, tot_reasoner, mock_llm):
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
            await tot_reasoner._generate_thoughts(parent, num=2)


class TestThoughtEvaluation:
    """Test thought evaluation and scoring."""

    @pytest.mark.asyncio
    async def test_evaluate_thought_success(self, tot_reasoner, mock_llm):
        """Test successful thought evaluation."""
        thought = ThoughtNode(
            node_id="test_node",
            thought_type=ThoughtType.ANALYSIS,
            content="This is a good reasoning step",
            depth=1,
        )

        mock_llm.ainvoke.return_value = Mock(content="8")  # Score 0-10

        score = await tot_reasoner._evaluate_thought(thought)

        assert 0.0 <= score <= 1.0
        assert score == 0.8  # 8/10 = 0.8

    @pytest.mark.asyncio
    async def test_evaluate_thought_llm_failure(self, tot_reasoner, mock_llm):
        """Test evaluation handles LLM failures gracefully (returns 0.5)."""
        thought = ThoughtNode(node_id="test", thought_type=ThoughtType.ANALYSIS, content="Test", depth=1)

        mock_llm.ainvoke.side_effect = Exception("LLM API error")

        score = await tot_reasoner._evaluate_thought(thought)

        assert score == 0.5  # Default score on failure

    @pytest.mark.asyncio
    async def test_evaluate_thought_parsing_failure(self, tot_reasoner, mock_llm):
        """Test evaluation handles parsing failures gracefully."""
        thought = ThoughtNode(node_id="test", thought_type=ThoughtType.ANALYSIS, content="Test", depth=1)

        mock_llm.ainvoke.return_value = Mock(content="not a number")

        score = await tot_reasoner._evaluate_thought(thought)

        assert score == 0.5  # Default score on parsing failure

    @pytest.mark.asyncio
    async def test_evaluate_thought_out_of_range(self, tot_reasoner, mock_llm):
        """Test evaluation clamps scores to [0, 1]."""
        thought = ThoughtNode(node_id="test", thought_type=ThoughtType.ANALYSIS, content="Test", depth=1)

        mock_llm.ainvoke.return_value = Mock(content="15")  # Out of range

        score = await tot_reasoner._evaluate_thought(thought)

        assert score == 1.0  # Clamped to 1.0


class TestPathExtraction:
    """Test path extraction from thought tree."""

    def test_extract_all_paths_single_branch(self, tot_reasoner):
        """Test extracting paths from a single-branch tree."""
        root = ThoughtNode(node_id="root", thought_type=ThoughtType.DECOMPOSITION, content="Root", depth=0)
        child1 = ThoughtNode(node_id="child_1", thought_type=ThoughtType.ANALYSIS, content="Child 1", parent_id="root", depth=1)
        child2 = ThoughtNode(node_id="child_2", thought_type=ThoughtType.SYNTHESIS, content="Child 2", parent_id="child_1", depth=2)

        tree = ThoughtTree(root=root, nodes=[root, child1, child2])

        paths = tot_reasoner._extract_all_paths(tree)

        assert len(paths) == 1
        assert len(paths[0]) == 3  # root -> child1 -> child2

    def test_extract_all_paths_multiple_branches(self, tot_reasoner):
        """Test extracting paths from a multi-branch tree."""
        root = ThoughtNode(node_id="root", thought_type=ThoughtType.DECOMPOSITION, content="Root", depth=0)
        child1 = ThoughtNode(node_id="child_1", thought_type=ThoughtType.ANALYSIS, content="Child 1", parent_id="root", depth=1)
        child2 = ThoughtNode(node_id="child_2", thought_type=ThoughtType.ANALYSIS, content="Child 2", parent_id="root", depth=1)

        tree = ThoughtTree(root=root, nodes=[root, child1, child2])

        paths = tot_reasoner._extract_all_paths(tree)

        assert len(paths) == 2  # Two branches


class TestPathScoring:
    """Test path scoring algorithm."""

    def test_score_path_weighted_average(self, tot_reasoner):
        """Test path scoring uses weighted average (later thoughts weighted higher)."""
        path = [
            ThoughtNode(node_id="root", thought_type=ThoughtType.DECOMPOSITION, content="Root", confidence_score=1.0, depth=0),
            ThoughtNode(node_id="node_1", thought_type=ThoughtType.ANALYSIS, content="Node 1", confidence_score=0.5, depth=1),
            ThoughtNode(node_id="node_2", thought_type=ThoughtType.SYNTHESIS, content="Node 2", confidence_score=0.8, depth=2),
        ]

        score = tot_reasoner._score_path(path)

        assert 0.0 <= score <= 1.0
        # Later thoughts should be weighted more heavily
        assert score > (1.0 + 0.5 + 0.8) / 3  # Higher than simple average

    def test_score_path_empty(self, tot_reasoner):
        """Test scoring empty path."""
        score = tot_reasoner._score_path([])
        assert score == 0.0

    def test_score_path_single_node(self, tot_reasoner):
        """Test scoring single-node path."""
        path = [ThoughtNode(node_id="root", thought_type=ThoughtType.DECOMPOSITION, content="Root", confidence_score=0.7, depth=0)]
        score = tot_reasoner._score_path(path)
        assert score == 0.7


class TestTreeExploration:
    """Test complete tree exploration process."""

    @pytest.mark.asyncio
    async def test_explore_basic(self, tot_reasoner, mock_llm):
        """Test basic tree exploration."""
        # Mock responses for root, thoughts, evaluation, and synthesis
        mock_llm.ainvoke.side_effect = [
            Mock(content="Root: Decompose problem"),  # Root
            Mock(content="1. Branch A\n2. Branch B"),  # First level thoughts
            Mock(content="8"),  # Evaluation for Branch A
            Mock(content="7"),  # Evaluation for Branch B
            Mock(content="1. SubA1\n2. SubA2"),  # Second level from Branch A
            Mock(content="9"),  # Evaluation
            Mock(content="8"),  # Evaluation
            Mock(content="1. SubB1\n2. SubB2"),  # Second level from Branch B
            Mock(content="7"),  # Evaluation
            Mock(content="6"),  # Evaluation
            Mock(content="Final answer based on best path"),  # Synthesis
        ]

        result = await tot_reasoner.explore("Test query")

        assert result.reasoning_type == "tot"
        assert len(result.best_path) > 0
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.final_answer is not None


class TestSynthesis:
    """Test answer synthesis."""

    @pytest.mark.asyncio
    async def test_synthesize_answer_success(self, tot_reasoner, mock_llm):
        """Test successful answer synthesis."""
        path = [
            ThoughtNode(node_id="root", thought_type=ThoughtType.DECOMPOSITION, content="Decompose problem", depth=0),
            ThoughtNode(node_id="node_1", thought_type=ThoughtType.ANALYSIS, content="Step 1", depth=1),
            ThoughtNode(node_id="node_2", thought_type=ThoughtType.SYNTHESIS, content="Step 2", depth=2),
        ]

        mock_llm.ainvoke.return_value = Mock(content="Synthesized final answer")

        answer = await tot_reasoner._synthesize_answer(path)

        assert answer == "Synthesized final answer"

    @pytest.mark.asyncio
    async def test_synthesize_answer_llm_failure(self, tot_reasoner, mock_llm):
        """Test synthesis handles LLM failures."""
        path = [ThoughtNode(node_id="root", thought_type=ThoughtType.DECOMPOSITION, content="Test", depth=0)]

        mock_llm.ainvoke.side_effect = Exception("LLM API error")

        with pytest.raises(RuntimeError, match="LLM synthesis failed"):
            await tot_reasoner._synthesize_answer(path)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_explore_no_children_generated(self, tot_reasoner, mock_llm):
        """Test exploration when no child thoughts are generated."""
        mock_llm.ainvoke.side_effect = [
            Mock(content="Root node"),  # Root
            Mock(content=""),  # Empty thoughts
            Mock(content="Final answer"),  # Synthesis
        ]

        result = await tot_reasoner.explore("Test query")

        assert result.reasoning_type == "tot"
        assert len(result.best_path) >= 1  # At least root

    @pytest.mark.asyncio
    async def test_explore_early_termination(self, tot_reasoner, mock_llm):
        """Test early termination when no thoughts to explore."""
        # Set max_depth=3 but generate empty thoughts at depth 1
        tot_reasoner.max_depth = 3

        mock_llm.ainvoke.side_effect = [
            Mock(content="Root"),  # Root
            Mock(content=""),  # No thoughts generated
            Mock(content="Final answer"),  # Synthesis
        ]

        result = await tot_reasoner.explore("Test query")

        assert result.reasoning_type == "tot"
        # Should terminate early, not reach max_depth


class TestThoughtTreeMethods:
    """Test ThoughtTree helper methods."""

    def test_get_leaf_nodes(self):
        """Test getting leaf nodes from tree."""
        root = ThoughtNode(node_id="root", thought_type=ThoughtType.DECOMPOSITION, content="Root", depth=0)
        child1 = ThoughtNode(node_id="child_1", thought_type=ThoughtType.ANALYSIS, content="Child 1", parent_id="root", depth=1)
        leaf1 = ThoughtNode(node_id="leaf_1", thought_type=ThoughtType.SYNTHESIS, content="Leaf 1", parent_id="child_1", depth=2)
        leaf2 = ThoughtNode(node_id="leaf_2", thought_type=ThoughtType.SYNTHESIS, content="Leaf 2", parent_id="child_1", depth=2)

        tree = ThoughtTree(root=root, nodes=[root, child1, leaf1, leaf2])

        leaf_nodes = tree.get_leaf_nodes()

        assert len(leaf_nodes) == 2
        assert all(node.node_id in ["leaf_1", "leaf_2"] for node in leaf_nodes)

    def test_get_path_to_root(self):
        """Test getting path from node to root."""
        root = ThoughtNode(node_id="root", thought_type=ThoughtType.DECOMPOSITION, content="Root", depth=0)
        child = ThoughtNode(node_id="child", thought_type=ThoughtType.ANALYSIS, content="Child", parent_id="root", depth=1)
        leaf = ThoughtNode(node_id="leaf", thought_type=ThoughtType.SYNTHESIS, content="Leaf", parent_id="child", depth=2)

        tree = ThoughtTree(root=root, nodes=[root, child, leaf])

        path = tree.get_path_to_root(leaf)

        assert len(path) == 3
        assert path[0].node_id == "root"
        assert path[1].node_id == "child"
        assert path[2].node_id == "leaf"
