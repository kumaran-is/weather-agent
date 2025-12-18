"""Curated knowledge loader for weather knowledge text files.

Loads text documents from backend/data/raw/curated/ directory into LangChain
Document objects for RAG indexing.

Directory structure:
    backend/data/raw/curated/
    ├── hurricanes/
    │   ├── saffir_simpson_scale.txt
    │   ├── evacuation_zones.txt
    │   ├── hurricane_katrina.txt
    │   └── ...
    ├── safety_guidelines/
    │   ├── heatwave_safety.txt
    │   ├── cold_weather_safety.txt
    │   └── ...
    ├── weather_terminology/
    │   ├── heat_index.txt
    │   ├── wind_chill.txt
    │   └── ...
    └── climate_patterns/
        ├── el_nino.txt
        ├── la_nina.txt
        └── ...

Usage:
    >>> from backend.src.rag.loaders.curated_knowledge_loader import load_all_curated_documents
    >>> docs = load_all_curated_documents()
    >>> print(f"Loaded {len(docs)} curated documents")
    Loaded 23 curated documents

    >>> # Load specific category
    >>> hurricane_docs = load_curated_category("hurricanes")
    >>> print(hurricane_docs[0].page_content[:100])
    Saffir-Simpson Hurricane Wind Scale...
"""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document


def load_curated_category(category: str) -> list[Document]:
    """Load all text documents from a specific curated knowledge category.

    Args:
        category: Category directory name
            - "hurricanes"
            - "safety_guidelines"
            - "weather_terminology"
            - "climate_patterns"

    Returns:
        list[Document]: Loaded documents with metadata

    Example:
        >>> docs = load_curated_category("hurricanes")
        >>> print(docs[0].metadata)
        {'source': 'backend/data/raw/curated/hurricanes/saffir_simpson_scale.txt',
         'category': 'hurricanes'}
    """
    # Get category directory path
    base_path = Path("backend/data/raw/curated")
    category_path = base_path / category

    if not category_path.exists():
        print(f"⚠️  Category not found: {category_path}")
        return []

    # Load all .txt files from category directory
    try:
        loader = DirectoryLoader(
            str(category_path),
            glob="*.txt",
            loader_cls=TextLoader,
            show_progress=True,
        )
        documents = loader.load()

        # Add category metadata
        for doc in documents:
            doc.metadata["category"] = category
            # Extract filename without extension for better metadata
            filename = Path(doc.metadata["source"]).stem
            doc.metadata["topic"] = filename.replace("_", " ").title()

        print(f"✅ Loaded {len(documents)} documents from {category}/")
        return documents

    except Exception as e:
        print(f"❌ Error loading {category}: {e}")
        return []


def load_all_curated_documents() -> list[Document]:
    """Load all curated knowledge documents from all categories.

    Loads from:
        - hurricanes/
        - safety_guidelines/
        - weather_terminology/
        - climate_patterns/

    Returns:
        list[Document]: All curated documents with category metadata

    Example:
        >>> docs = load_all_curated_documents()
        >>> print(f"Total documents: {len(docs)}")
        Total documents: 23

        >>> # Group by category
        >>> from collections import Counter
        >>> categories = [doc.metadata['category'] for doc in docs]
        >>> print(Counter(categories))
        Counter({'hurricanes': 6, 'safety_guidelines': 5,
                 'weather_terminology': 5, 'climate_patterns': 7})
    """
    print("=" * 70)
    print("Loading Curated Weather Knowledge Documents")
    print("=" * 70)

    all_documents = []
    categories = [
        "hurricanes",
        "safety_guidelines",
        "weather_terminology",
        "climate_patterns",
    ]

    for category in categories:
        docs = load_curated_category(category)
        all_documents.extend(docs)

    print("\n" + "=" * 70)
    print(f"✅ Total curated documents loaded: {len(all_documents)}")
    print("=" * 70)

    # Print summary by category
    from collections import Counter

    category_counts = Counter(doc.metadata["category"] for doc in all_documents)
    for category, count in sorted(category_counts.items()):
        print(f"   {category}: {count} documents")

    return all_documents


def load_curated_document(filepath: str) -> Document:
    """Load a single curated document by filepath.

    Args:
        filepath: Path to .txt file (relative or absolute)

    Returns:
        Document: Loaded document with metadata

    Example:
        >>> doc = load_curated_document("backend/data/raw/curated/hurricanes/saffir_simpson_scale.txt")
        >>> print(len(doc.page_content))
        5432
    """
    try:
        loader = TextLoader(filepath)
        docs = loader.load()

        if docs:
            doc = docs[0]
            # Add category metadata if in curated directory
            if "curated/" in filepath:
                parts = Path(filepath).parts
                if "curated" in parts:
                    curated_idx = parts.index("curated")
                    if curated_idx + 1 < len(parts):
                        doc.metadata["category"] = parts[curated_idx + 1]

            return doc
        else:
            raise ValueError(f"No content in {filepath}")

    except Exception as e:
        raise ValueError(f"Failed to load {filepath}: {e}")


if __name__ == "__main__":
    # Test loading
    docs = load_all_curated_documents()

    # Show sample
    if docs:
        print("\n" + "=" * 70)
        print("Sample Document")
        print("=" * 70)
        sample = docs[0]
        print(f"Source: {sample.metadata['source']}")
        print(f"Category: {sample.metadata['category']}")
        print(f"Topic: {sample.metadata.get('topic', 'N/A')}")
        print(f"Content length: {len(sample.page_content)} characters")
        print("\nFirst 200 characters:")
        print(sample.page_content[:200] + "...")
