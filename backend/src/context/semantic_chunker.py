"""
Semantic Chunking for Context Optimization.

Phase 2 of the 5-phase context optimization pipeline.

Provides intelligent text chunking that preserves semantic boundaries,
ensuring that meaningful units of information are not split across chunks.

Key Features:
- Recursive character splitting with configurable separators
- Semantic boundary preservation (sentences, paragraphs)
- Configurable chunk size and overlap
- Weather-domain specific separators
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Phase 2: Semantic chunking with boundary preservation.

    Splits text into semantically meaningful chunks while preserving
    context boundaries. Uses a hierarchy of separators to find the
    best split points.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        separators: list[str] | None = None,
    ):
        """
        Initialize the semantic chunker.

        Args:
            chunk_size: Target size for each chunk (in characters)
            overlap: Number of characters to overlap between chunks
            separators: Custom separators (default: paragraph, sentence, word)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or [
            "\n\n\n",  # Triple newline (major sections)
            "\n\n",  # Double newline (paragraphs)
            "\n",  # Single newline
            ". ",  # Sentence boundary
            "? ",  # Question boundary
            "! ",  # Exclamation boundary
            "; ",  # Semicolon boundary
            ", ",  # Comma boundary
            " ",  # Word boundary
            "",  # Character boundary (last resort)
        ]

        logger.debug(
            f"SemanticChunker initialized | chunk_size={chunk_size} | overlap={overlap}"
        )

    def chunk(self, text: str) -> list[str]:
        """
        Split text into semantic chunks.

        Uses recursive splitting with the separator hierarchy to preserve
        semantic boundaries while staying within chunk size limits.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if not text:
            return []

        # Clean text before chunking
        text = self._clean_text(text)

        # If text is already small enough, return as single chunk
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Recursively split using separators
        chunks = self._recursive_split(text, self.separators)

        # Merge small chunks and ensure overlap
        merged_chunks = self._merge_small_chunks(chunks)
        final_chunks = self._add_overlap(merged_chunks)

        logger.debug(
            f"Chunked text | original_length={len(text)} | "
            f"chunks={len(final_chunks)} | avg_size={len(text)//max(len(final_chunks),1)}"
        )

        return final_chunks

    def _clean_text(self, text: str) -> str:
        """Clean text by removing excessive whitespace."""
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        # Restore paragraph breaks
        text = re.sub(r"(\. )", r".\n", text)
        # Remove trailing/leading whitespace
        return text.strip()

    def _recursive_split(
        self, text: str, separators: list[str]
    ) -> list[str]:
        """Recursively split text using separator hierarchy."""
        if not separators:
            # Base case: no more separators, split by character
            return self._split_by_size(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        # Try to split by current separator
        if separator:
            splits = text.split(separator)
        else:
            # Empty separator means character-level split
            splits = list(text)

        # If we only got one piece, try next separator
        if len(splits) == 1:
            return self._recursive_split(text, remaining_separators)

        # Process each split
        chunks: list[str] = []
        current_chunk = ""

        for split in splits:
            # Add separator back (except for character splits)
            piece = split + separator if separator else split

            # If adding this piece exceeds chunk size
            if len(current_chunk) + len(piece) > self.chunk_size:
                # Save current chunk if not empty
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                # Start new chunk
                current_chunk = piece
            else:
                current_chunk += piece

        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Recursively process any oversized chunks
        final_chunks: list[str] = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size and remaining_separators:
                final_chunks.extend(
                    self._recursive_split(chunk, remaining_separators)
                )
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _split_by_size(self, text: str) -> list[str]:
        """Split text into fixed-size chunks (last resort)."""
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i : i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """Merge chunks that are too small."""
        min_chunk_size = self.chunk_size // 4  # Minimum 25% of target size
        merged: list[str] = []
        buffer = ""

        for chunk in chunks:
            if len(chunk) < min_chunk_size:
                # Accumulate small chunks
                buffer += " " + chunk if buffer else chunk
            else:
                # Flush buffer if it exists
                if buffer:
                    if len(buffer) + len(chunk) <= self.chunk_size:
                        chunk = buffer + " " + chunk
                        buffer = ""
                    else:
                        merged.append(buffer.strip())
                        buffer = ""
                merged.append(chunk.strip())

        # Flush remaining buffer
        if buffer.strip():
            if merged and len(merged[-1]) + len(buffer) <= self.chunk_size:
                merged[-1] = merged[-1] + " " + buffer.strip()
            else:
                merged.append(buffer.strip())

        return merged

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap between consecutive chunks for context continuity."""
        if len(chunks) <= 1 or self.overlap <= 0:
            return chunks

        overlapped: list[str] = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk: no prefix overlap
                overlapped.append(chunk)
            else:
                # Get overlap from previous chunk
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-self.overlap :] if len(prev_chunk) > self.overlap else prev_chunk

                # Find word boundary for cleaner overlap
                word_boundary = overlap_text.rfind(" ")
                if word_boundary > 0:
                    overlap_text = overlap_text[word_boundary + 1 :]

                # Prepend overlap
                overlapped.append(f"...{overlap_text} {chunk}")

        return overlapped

    def chunk_with_metadata(
        self, text: str
    ) -> list[dict[str, Any]]:
        """
        Chunk text and return with metadata.

        Returns chunks with additional information like position,
        length, and semantic type.
        """
        chunks = self.chunk(text)

        result = []
        position = 0

        for i, chunk in enumerate(chunks):
            # Determine semantic type
            semantic_type = self._determine_semantic_type(chunk)

            result.append(
                {
                    "chunk": chunk,
                    "index": i,
                    "length": len(chunk),
                    "start_position": position,
                    "semantic_type": semantic_type,
                    "is_first": i == 0,
                    "is_last": i == len(chunks) - 1,
                }
            )

            # Update position (approximate due to overlap)
            position += len(chunk) - self.overlap

        return result

    def _determine_semantic_type(self, chunk: str) -> str:
        """Determine the semantic type of a chunk."""
        lower_chunk = chunk.lower()

        # Weather-specific types
        if any(
            word in lower_chunk
            for word in ["hurricane", "storm", "tropical", "cyclone"]
        ):
            return "hurricane"
        if any(
            word in lower_chunk for word in ["forecast", "prediction", "outlook"]
        ):
            return "forecast"
        if any(word in lower_chunk for word in ["warning", "alert", "advisory"]):
            return "alert"
        if any(
            word in lower_chunk for word in ["temperature", "temp", "°f", "°c"]
        ):
            return "temperature"
        if any(word in lower_chunk for word in ["wind", "mph", "knots"]):
            return "wind"
        if any(
            word in lower_chunk for word in ["evacuation", "evacuate", "shelter"]
        ):
            return "evacuation"

        # General types
        if chunk.startswith("#") or chunk.startswith("##"):
            return "heading"
        if "|" in chunk and chunk.count("|") >= 2:
            return "table"
        if chunk.startswith("{") or chunk.startswith("["):
            return "json"
        if any(chunk.startswith(f"{i}.") for i in range(1, 20)):
            return "list"

        return "general"
