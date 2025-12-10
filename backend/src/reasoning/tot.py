"""Tree of Thoughts implementation for Level 3b.

Tree of Thoughts (ToT) is an advanced reasoning technique that:
1. Explores multiple reasoning paths simultaneously (breadth-first)
2. Evaluates each thought/reasoning step
3. Prunes less promising paths
4. Selects the best path to the final answer

Key Benefits:
- Better decision-making through exploration
- Handles complex multi-step problems
- Self-evaluating and self-correcting

Reference: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
https://arxiv.org/abs/2305.10601
"""

import uuid

from langchain_core.language_models import BaseChatModel

from backend.src.models.reasoning import (
    ReasoningResult,
    ThoughtNode,
    ThoughtTree,
    ThoughtType,
)


class TreeOfThoughts:
    """BFS-based Tree of Thoughts reasoning.

    Algorithm:
    1. Generate root thought from query
    2. For each depth level:
       a. For each thought in current level:
          - Generate N alternative next thoughts
          - Evaluate each thought (score 0-1)
          - Prune: Keep only top breadth thoughts
       b. Add surviving thoughts to next level
    3. Extract all paths from root to leaves
    4. Select path with highest cumulative score

    Parameters:
        llm: Language model for thought generation and evaluation
        max_depth: Maximum tree depth (default: 3)
        breadth: Number of thoughts to keep at each level (default: 3)
    """

    def __init__(
        self, llm: BaseChatModel, max_depth: int = 3, breadth: int = 3
    ) -> None:
        """Initialize Tree of Thoughts reasoner.

        Args:
            llm: Language model instance
            max_depth: Maximum depth to explore (1-5)
            breadth: Number of branches to keep per level (2-5)
        """
        self.llm = llm
        self.max_depth = max(1, min(5, max_depth))  # Clamp to [1, 5]
        self.breadth = max(2, min(5, breadth))  # Clamp to [2, 5]

    async def explore(self, query: str) -> ReasoningResult:
        """Explore multiple reasoning paths and select best.

        Args:
            query: User query to reason about

        Returns:
            ReasoningResult with best path, confidence, and all paths explored
        """
        # Generate root thought
        root = await self._generate_root(query)

        # Build tree structure
        tree = ThoughtTree(
            root=root, nodes=[root], max_depth=self.max_depth, breadth=self.breadth
        )

        # BFS exploration
        current_level = [root]

        for level in range(self.max_depth):
            next_level = []

            for thought in current_level:
                # Generate N alternative next thoughts
                children = await self._generate_thoughts(thought, num=self.breadth)

                # Evaluate each thought
                scored_children = []
                for child in children:
                    score = await self._evaluate_thought(child)
                    child.confidence_score = score
                    scored_children.append(child)
                    tree.nodes.append(child)

                # Prune: Keep top breadth thoughts
                best_children = sorted(
                    scored_children, key=lambda x: x.confidence_score, reverse=True
                )[: self.breadth]

                next_level.extend(best_children)

            # Move to next level
            current_level = next_level

            # Early termination if no more thoughts to explore
            if not current_level:
                break

        # Extract all paths from root to leaves
        paths = self._extract_all_paths(tree)

        # Select best path
        if not paths:
            # Fallback: Return root only
            paths = [[root]]

        best_path = max(paths, key=lambda p: self._score_path(p))

        # Generate final answer from best path
        final_answer = await self._synthesize_answer(best_path)

        return ReasoningResult(
            reasoning_type="tot",
            best_path=best_path,
            confidence_score=self._score_path(best_path),
            all_paths=paths,
            final_answer=final_answer,
        )

    async def _generate_root(self, query: str) -> ThoughtNode:
        """Generate root thought (problem decomposition).

        Args:
            query: Original user query

        Returns:
            Root ThoughtNode with problem decomposition

        Raises:
            RuntimeError: If LLM generation fails
        """
        prompt = f"""
        Problem: {query}

        Decompose this problem into its key components.
        What are the main sub-questions or sub-tasks needed to answer this?

        Provide a brief decomposition (2-3 sentences).
        """

        try:
            response = await self.llm.ainvoke(prompt)
        except Exception as e:
            raise RuntimeError(f"LLM root generation failed: {e}")

        return ThoughtNode(
            node_id="root",
            parent_id=None,
            thought_type=ThoughtType.DECOMPOSITION,
            content=response.content.strip(),
            confidence_score=1.0,
            depth=0,
        )

    async def _generate_thoughts(
        self, parent: ThoughtNode, num: int
    ) -> list[ThoughtNode]:
        """Generate N alternative next thoughts.

        Args:
            parent: Parent thought node
            num: Number of alternative thoughts to generate

        Returns:
            List of ThoughtNode instances (up to num nodes)

        Raises:
            RuntimeError: If LLM generation fails
        """
        prompt = f"""
        Current reasoning step: {parent.content}

        Generate {num} different next reasoning steps.
        Each should explore a different approach or angle.

        Requirements:
        - Be specific and concrete
        - Focus on actionable steps
        - Explore diverse approaches

        Format each thought on a new line starting with a number:
        1. [First alternative thought]
        2. [Second alternative thought]
        ...
        """

        try:
            response = await self.llm.ainvoke(prompt)
        except Exception as e:
            raise RuntimeError(f"LLM thought generation failed: {e}")

        # Parse thoughts from response
        thoughts = []
        lines = response.content.strip().split("\n")

        for i, line in enumerate(lines):
            line = line.strip()
            if not line or not line[0].isdigit():
                continue

            # Extract content after number and period
            content = line.split(". ", 1)[1] if ". " in line else line

            thoughts.append(
                ThoughtNode(
                    node_id=f"{parent.node_id}_child_{str(uuid.uuid4())[:8]}",
                    parent_id=parent.node_id,
                    thought_type=ThoughtType.ANALYSIS,
                    content=content,
                    depth=parent.depth + 1,
                    confidence_score=0.5,  # Will be evaluated separately
                )
            )

        return thoughts[:num]

    async def _evaluate_thought(self, thought: ThoughtNode) -> float:
        """Score thought quality (0-1).

        Evaluation criteria:
        - Logical coherence
        - Progress toward solution
        - Specificity and actionability

        Args:
            thought: Thought node to evaluate

        Returns:
            Score between 0.0 and 1.0 (defaults to 0.5 on error)

        Note:
            If LLM generation or parsing fails, gracefully degrades to 0.5
        """
        prompt = f"""
        Evaluate the promise of this reasoning step (0-10 scale):

        Reasoning step: {thought.content}

        Criteria:
        1. Does it make logical sense?
        2. Does it move toward solving the problem?
        3. Is it specific and actionable (not vague)?

        Return ONLY a single number between 0-10 (no explanation).
        """

        try:
            response = await self.llm.ainvoke(prompt)
        except Exception as e:
            # If LLM fails, return medium score (graceful degradation)
            return 0.5

        try:
            # Parse score from response
            score_text = response.content.strip()
            score = float(score_text)
            # Normalize to 0-1 range
            normalized_score = max(0.0, min(1.0, score / 10.0))
            return normalized_score
        except (ValueError, AttributeError):
            # Default to medium score if parsing fails
            return 0.5

    def _extract_all_paths(self, tree: ThoughtTree) -> list[list[ThoughtNode]]:
        """Extract all paths from root to leaf nodes.

        Args:
            tree: ThoughtTree instance

        Returns:
            List of paths, where each path is a list of ThoughtNodes
        """
        leaf_nodes = tree.get_leaf_nodes()

        if not leaf_nodes:
            # No leaves found, return root only
            return [[tree.root]]

        paths = []
        for leaf in leaf_nodes:
            path = tree.get_path_to_root(leaf)
            if path:
                paths.append(path)

        return paths

    def _score_path(self, path: list[ThoughtNode]) -> float:
        """Calculate cumulative score for a path.

        Uses weighted average favoring later thoughts (they're more refined).

        Args:
            path: List of ThoughtNodes representing a path

        Returns:
            Path score between 0.0 and 1.0
        """
        if not path:
            return 0.0

        # Weight later thoughts more heavily (they're closer to solution)
        weights = [i + 1 for i in range(len(path))]
        total_weight = sum(weights)

        weighted_sum = sum(
            node.confidence_score * weight for node, weight in zip(path, weights)
        )

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    async def _synthesize_answer(self, path: list[ThoughtNode]) -> str:
        """Synthesize final answer from best reasoning path.

        Args:
            path: Best reasoning path

        Returns:
            Final synthesized answer

        Raises:
            RuntimeError: If LLM synthesis fails
        """
        # Build reasoning chain summary
        reasoning_chain = "\n".join(
            [f"{i+1}. {node.content}" for i, node in enumerate(path)]
        )

        prompt = f"""
        Based on this chain of reasoning, provide a final answer:

        Reasoning chain:
        {reasoning_chain}

        Synthesize a clear, concise final answer that incorporates all the reasoning steps.
        Be specific and actionable.
        """

        try:
            response = await self.llm.ainvoke(prompt)
        except Exception as e:
            raise RuntimeError(f"LLM synthesis failed: {e}")

        return response.content.strip()
