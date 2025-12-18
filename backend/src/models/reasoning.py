"""Reasoning models for Level 3b: ToT + GoT.

CRITICAL: These are INTERNAL models (not API-facing)

Models:
- ThoughtType: Enumeration of thought types
- ThoughtNode: Single reasoning step
- ThoughtTree: Tree of Thoughts structure
- ThoughtGraph: Graph of Thoughts structure
- ReasoningResult: Final reasoning output

Level 3b Enhancements:
- Tree of Thoughts (ToT) for multi-path exploration
- Graph of Thoughts (GoT) for DAG-based reasoning
- Step-Back Prompting for principle-based reasoning
"""

from enum import Enum

from pydantic import BaseModel, Field


class ThoughtType(str, Enum):
    """Type of reasoning thought."""

    DECOMPOSITION = "decomposition"  # Breaking down the problem
    ANALYSIS = "analysis"  # Analyzing a sub-problem
    SYNTHESIS = "synthesis"  # Combining multiple insights
    EVALUATION = "evaluation"  # Evaluating a solution


class ThoughtNode(BaseModel):
    """Single reasoning step in ToT/GoT.

    Represents one node in a thought tree or graph, containing:
    - Content of the thought
    - Type of reasoning used
    - Confidence score
    - Position in the reasoning structure
    """

    node_id: str = Field(description="Unique identifier for this thought node")
    parent_id: str | None = Field(
        default=None, description="ID of parent node (None for root)"
    )
    thought_type: ThoughtType = Field(description="Type of reasoning step")
    content: str = Field(description="The actual thought/reasoning content")
    confidence_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in this thought (0.0 to 1.0)",
    )
    depth: int = Field(default=0, ge=0, description="Depth in the tree (0 = root)")


class ThoughtTree(BaseModel):
    """Tree of Thoughts structure for multi-path exploration.

    Used in Tree of Thoughts (ToT) reasoning:
    - Explores multiple reasoning paths simultaneously
    - Prunes less promising paths
    - Selects best path to final answer
    """

    root: ThoughtNode = Field(description="Root node of the thought tree")
    nodes: list[ThoughtNode] = Field(
        default_factory=list, description="All nodes in the tree"
    )
    max_depth: int = Field(default=3, ge=1, le=5, description="Maximum tree depth")
    breadth: int = Field(
        default=3, ge=2, le=5, description="Number of branches per level"
    )

    def get_leaf_nodes(self) -> list[ThoughtNode]:
        """Get all leaf nodes (nodes with no children)."""
        # Find nodes that are not parents
        parent_ids = {node.parent_id for node in self.nodes if node.parent_id}
        node_ids = {node.node_id for node in self.nodes}

        leaf_ids = node_ids - parent_ids
        return [node for node in self.nodes if node.node_id in leaf_ids]

    def get_path_to_root(self, node: ThoughtNode) -> list[ThoughtNode]:
        """Get path from given node back to root."""
        path = [node]
        current = node

        while current.parent_id:
            # Find parent node
            parent = next(
                (n for n in self.nodes if n.node_id == current.parent_id), None
            )
            if not parent:
                break

            path.insert(0, parent)
            current = parent

        return path


class ThoughtGraph(BaseModel):
    """Graph of Thoughts structure for interconnected reasoning.

    Used in Graph of Thoughts (GoT) reasoning:
    - Represents reasoning as a DAG (Directed Acyclic Graph)
    - Allows thought merging (shared sub-problems)
    - Enables more efficient exploration than ToT
    """

    nodes: list[ThoughtNode] = Field(
        default_factory=list, description="All nodes in the graph"
    )
    edges: list[dict[str, str]] = Field(
        default_factory=list,
        description="Edges between nodes: [{'source': 'node_1', 'target': 'node_2'}]",
    )
    merge_nodes: list[str] = Field(
        default_factory=list,
        description="Node IDs where multiple paths merge (shared sub-problems)",
    )

    def add_edge(self, source_id: str, target_id: str) -> None:
        """Add an edge between two nodes."""
        self.edges.append({"source": source_id, "target": target_id})

    def get_children(self, node_id: str) -> list[str]:
        """Get all children of a node."""
        return [edge["target"] for edge in self.edges if edge["source"] == node_id]

    def get_parents(self, node_id: str) -> list[str]:
        """Get all parents of a node."""
        return [edge["source"] for edge in self.edges if edge["target"] == node_id]

    def is_merge_node(self, node_id: str) -> bool:
        """Check if a node is a merge point (multiple parents)."""
        return len(self.get_parents(node_id)) > 1


class ReasoningResult(BaseModel):
    """Final reasoning output with confidence.

    Contains:
    - Best reasoning path found
    - Confidence score for the path
    - All explored paths (for debugging/analysis)
    - Type of reasoning used
    """

    reasoning_type: str = Field(
        description="Type of reasoning: 'tot', 'got', or 'stepback'"
    )
    best_path: list[ThoughtNode] = Field(
        description="Best reasoning path from root to conclusion"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Overall confidence in the result (0.0 to 1.0)"
    )
    all_paths: list[list[ThoughtNode]] = Field(
        default_factory=list,
        description="All explored paths (for debugging and analysis)",
    )
    final_answer: str | None = Field(
        default=None, description="Final synthesized answer from best path"
    )

    def get_best_path_summary(self) -> str:
        """Get a summary of the best reasoning path."""
        steps = []
        for i, node in enumerate(self.best_path, 1):
            steps.append(f"{i}. [{node.thought_type.value}] {node.content}")

        return "\n".join(steps)
