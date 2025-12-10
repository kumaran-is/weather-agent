"""Graph of Thoughts implementation for Level 3b.

Graph of Thoughts (GoT) extends Tree of Thoughts by:
1. Representing reasoning as a DAG (Directed Acyclic Graph)
2. Detecting and merging similar thoughts (shared sub-problems)
3. Reusing solutions to common sub-problems
4. More efficient than ToT for problems with overlapping sub-tasks

Key Benefits:
- Efficient problem-solving through reuse
- Handles complex multi-city or multi-factor queries
- Natural representation of real-world problem structures

Reference: "Graph of Thoughts: Solving Elaborate Problems with Large Language Models"
https://arxiv.org/abs/2308.09687
"""

import uuid

from langchain_core.language_models import BaseChatModel

from backend.src.models.reasoning import (
    ReasoningResult,
    ThoughtGraph,
    ThoughtNode,
    ThoughtType,
)


class GraphOfThoughts:
    """DAG-based Graph of Thoughts reasoning.

    Algorithm:
    1. Create root node from query
    2. For each iteration:
       a. Find all leaf nodes (no children)
       b. For each leaf:
          - Generate new child thoughts
          - Check if similar thought already exists (merge)
          - If exists: Create edge to existing node
          - If not: Create new node
    3. Extract all paths from root to leaves
    4. Select path with highest score

    Parameters:
        llm: Language model for thought generation
        max_iterations: Maximum number of expansion iterations (default: 5)
        similarity_threshold: Threshold for merging nodes (default: 0.7)
    """

    def __init__(
        self,
        llm: BaseChatModel,
        max_iterations: int = 5,
        similarity_threshold: float = 0.7,
    ) -> None:
        """Initialize Graph of Thoughts reasoner.

        Args:
            llm: Language model instance
            max_iterations: Number of graph expansion iterations (2-10)
            similarity_threshold: Similarity threshold for node merging (0.0-1.0)
        """
        self.llm = llm
        self.max_iterations = max(2, min(10, max_iterations))
        self.similarity_threshold = max(0.0, min(1.0, similarity_threshold))

    async def build_graph(self, query: str) -> ReasoningResult:
        """Build thought graph with merging and reuse.

        Args:
            query: User query to reason about

        Returns:
            ReasoningResult with best path through the graph
        """
        # Initialize graph with root node
        root = await self._generate_root(query)
        graph = ThoughtGraph(nodes=[root], edges=[], merge_nodes=[])

        # Iteratively expand graph
        for _ in range(self.max_iterations):
            # Find leaf nodes (nodes with no outgoing edges)
            leaf_ids = self._get_leaf_node_ids(graph)

            if not leaf_ids:
                break  # No more nodes to expand

            # Expand each leaf
            for node_id in leaf_ids:
                parent = self._get_node_by_id(graph, node_id)
                if not parent:
                    continue

                # Generate new thoughts from this parent
                new_thoughts = await self._generate_thoughts(parent, num=3)

                for thought in new_thoughts:
                    # Check if similar thought already exists (merge opportunity)
                    existing_id = self._find_similar_node(graph, thought)

                    if existing_id:
                        # Merge: Create edge to existing node
                        graph.add_edge(node_id, existing_id)

                        # Mark as merge node if not already
                        if existing_id not in graph.merge_nodes:
                            graph.merge_nodes.append(existing_id)
                    else:
                        # Create new node
                        graph.nodes.append(thought)
                        graph.add_edge(node_id, thought.node_id)

        # Extract all paths from root to leaves
        paths = self._extract_all_paths(graph)

        # Select best path
        if not paths:
            paths = [[root]]

        best_path = max(paths, key=lambda p: self._score_path(p))

        # Generate final answer
        final_answer = await self._synthesize_answer(best_path, graph)

        return ReasoningResult(
            reasoning_type="got",
            best_path=best_path,
            confidence_score=self._score_path(best_path),
            all_paths=paths,
            final_answer=final_answer,
        )

    async def _generate_root(self, query: str) -> ThoughtNode:
        """Generate root node (problem decomposition).

        Args:
            query: Original user query

        Returns:
            Root ThoughtNode

        Raises:
            RuntimeError: If LLM generation fails
        """
        prompt = f"""
        Problem: {query}

        Break down this problem into independent sub-problems.
        What are the key components that need to be solved?

        Provide a structured decomposition (2-3 points).
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
        self, parent: ThoughtNode, num: int = 3
    ) -> list[ThoughtNode]:
        """Generate child thoughts from parent.

        Args:
            parent: Parent thought node
            num: Number of thoughts to generate

        Returns:
            List of ThoughtNode instances (may be fewer than num if generation fails)

        Raises:
            RuntimeError: If LLM generation fails
        """
        prompt = f"""
        Current reasoning: {parent.content}

        Generate {num} next reasoning steps to address this thought.
        Focus on sub-problems or analysis angles.

        Format:
        1. [First step]
        2. [Second step]
        ...
        """

        try:
            response = await self.llm.ainvoke(prompt)
        except Exception as e:
            raise RuntimeError(f"LLM thought generation failed: {e}")

        thoughts = []
        lines = response.content.strip().split("\n")

        for _, line in enumerate(lines):
            line = line.strip()
            if not line or not line[0].isdigit():
                continue

            content = line.split(". ", 1)[1] if ". " in line else line

            thoughts.append(
                ThoughtNode(
                    node_id=f"node_{str(uuid.uuid4())[:8]}",
                    parent_id=parent.node_id,
                    thought_type=ThoughtType.ANALYSIS,
                    content=content,
                    depth=parent.depth + 1,
                    confidence_score=0.7,  # Default score
                )
            )

        return thoughts[:num]

    def _find_similar_node(
        self, graph: ThoughtGraph, thought: ThoughtNode
    ) -> str | None:
        """Find existing node with similar content.

        Args:
            graph: Current thought graph
            thought: Thought to check for similarity

        Returns:
            Node ID of similar node, or None if no match
        """
        for node in graph.nodes:
            if node.node_id == thought.node_id:
                continue  # Skip self

            similarity = self._calculate_similarity(thought.content, node.content)

            if similarity >= self.similarity_threshold:
                return node.node_id

        return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using word overlap (Jaccard similarity).

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0.0 to 1.0)
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def _get_leaf_node_ids(self, graph: ThoughtGraph) -> list[str]:
        """Get IDs of all leaf nodes (no outgoing edges).

        Args:
            graph: Thought graph

        Returns:
            List of node IDs
        """
        # Find all node IDs that are NOT sources of any edge
        all_node_ids = {node.node_id for node in graph.nodes}
        source_ids = {edge["source"] for edge in graph.edges}

        leaf_ids = all_node_ids - source_ids
        return list(leaf_ids)

    def _get_node_by_id(self, graph: ThoughtGraph, node_id: str) -> ThoughtNode | None:
        """Get node by ID.

        Args:
            graph: Thought graph
            node_id: Node ID to find

        Returns:
            ThoughtNode or None if not found
        """
        for node in graph.nodes:
            if node.node_id == node_id:
                return node
        return None

    def _extract_all_paths(self, graph: ThoughtGraph) -> list[list[ThoughtNode]]:
        """Extract all paths from root to leaf nodes.

        Args:
            graph: Thought graph

        Returns:
            List of paths (each path is a list of ThoughtNodes)
        """
        root = self._get_node_by_id(graph, "root")
        if not root:
            return []

        leaf_ids = self._get_leaf_node_ids(graph)

        paths = []
        for leaf_id in leaf_ids:
            path = self._find_path(graph, "root", leaf_id)
            if path:
                paths.append(path)

        return paths if paths else [[root]]

    def _find_path(
        self, graph: ThoughtGraph, start_id: str, end_id: str
    ) -> list[ThoughtNode] | None:
        """Find a path from start to end node using DFS.

        Args:
            graph: Thought graph
            start_id: Start node ID
            end_id: End node ID

        Returns:
            Path as list of ThoughtNodes, or None if no path exists
        """
        if start_id == end_id:
            node = self._get_node_by_id(graph, start_id)
            return [node] if node else None

        visited = set()
        stack = [(start_id, [start_id])]

        while stack:
            current_id, path = stack.pop()

            if current_id in visited:
                continue

            visited.add(current_id)

            if current_id == end_id:
                # Convert path IDs to nodes
                return [
                    self._get_node_by_id(graph, node_id)
                    for node_id in path
                    if self._get_node_by_id(graph, node_id)
                ]

            # Add children to stack
            children = graph.get_children(current_id)
            for child_id in children:
                if child_id not in visited:
                    stack.append((child_id, path + [child_id]))

        return None

    def _score_path(self, path: list[ThoughtNode]) -> float:
        """Calculate path score.

        Considers:
        - Node confidence scores
        - Path length (prefer shorter paths)
        - Merge node bonus (reused sub-problems are valuable)

        Args:
            path: List of ThoughtNodes

        Returns:
            Path score (0.0 to 1.0)
        """
        if not path:
            return 0.0

        # Average confidence score
        avg_confidence = sum(node.confidence_score for node in path) / len(path)

        # Length penalty (prefer concise reasoning)
        length_penalty = 1.0 / (1.0 + 0.1 * len(path))

        # Combined score
        return avg_confidence * length_penalty

    async def _synthesize_answer(
        self, path: list[ThoughtNode], graph: ThoughtGraph
    ) -> str:
        """Synthesize final answer from best path and graph.

        Args:
            path: Best reasoning path
            graph: Complete thought graph (for context)

        Returns:
            Final answer

        Raises:
            RuntimeError: If LLM synthesis fails
        """
        # Build reasoning summary
        reasoning_chain = "\n".join(
            [f"{i+1}. {node.content}" for i, node in enumerate(path)]
        )

        # Note merge points if any exist in path
        merge_info = ""
        path_ids = {node.node_id for node in path}
        merge_nodes_in_path = [mid for mid in graph.merge_nodes if mid in path_ids]

        if merge_nodes_in_path:
            merge_info = f"\nNote: This solution reuses {len(merge_nodes_in_path)} shared sub-problem(s)."

        prompt = f"""
        Synthesize a final answer from this reasoning chain:

        {reasoning_chain}{merge_info}

        Provide a clear, actionable final answer that incorporates all reasoning steps.
        """

        try:
            response = await self.llm.ainvoke(prompt)
        except Exception as e:
            raise RuntimeError(f"LLM synthesis failed: {e}")

        return response.content.strip()
