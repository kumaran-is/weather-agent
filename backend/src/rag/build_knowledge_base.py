"""Build and load weather knowledge base into Qdrant vector store.

Main orchestrator for RAG knowledge base construction:
1. Validate Kaggle datasets
2. Load curated weather knowledge documents
3. Load Kaggle CSV documents (with narrative conversion)
4. Generate embeddings
5. Load into Qdrant vector database

Level 2 Target:
- ~500-600 documents total
- Curated knowledge: ~20-30 curated text files
- Kaggle data: ~500-600 narrative documents (sampled)

Usage:
    python -m backend.src.rag.build_knowledge_base

    # OR via Makefile:
    make rag-load
"""

from backend.src.rag.vector_store import get_vector_store
from backend.src.rag.embeddings import create_embeddings
from backend.src.rag.loaders.curated_knowledge_loader import load_all_curated_documents
from backend.src.rag.loaders.kaggle_loader import load_all_kaggle_documents
from backend.src.rag.loaders.validate_kaggle_datasets import (
    validate_all_kaggle_datasets,
    DatasetValidationError,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_documents(
    documents: list[Document], chunk_size: int = 800, chunk_overlap: int = 200
) -> list[Document]:
    """Split documents into chunks for better retrieval.

    Variable chunking strategy:
    - Default: 800 characters with 200 overlap (Level 2)
    - Level 5a will add: Category-specific chunking (hurricanes: 1000, safety: 600)

    Args:
        documents: list of documents to chunk
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks

    Returns:
        list[Document]: Chunked documents with preserved metadata

    Example:
        >>> docs = [Document(page_content="Long text..." * 100)]
        >>> chunks = chunk_documents(docs, chunk_size=500)
        >>> print(f"Split into {len(chunks)} chunks")
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # Try to split on paragraphs first
    )

    chunked = text_splitter.split_documents(documents)

    print(f"📄 Split {len(documents)} documents into {len(chunked)} chunks")
    print(f"   Avg chunk size: {sum(len(c.page_content) for c in chunked) // len(chunked)} chars")

    return chunked


def build_and_load_knowledge_base(
    skip_validation: bool = False,
    curated_only: bool = False,
    kaggle_sample_size: int = 500,
    chunk_size: int = 800,
) -> dict:
    """Build complete weather knowledge base and load into Qdrant.

    Complete pipeline:
    1. Validate Kaggle datasets (unless skipped)
    2. Load curated knowledge documents
    3. Load Kaggle documents (if enabled)
    4. Chunk documents
    5. Generate embeddings
    6. Load into Qdrant

    Args:
        skip_validation: Skip Kaggle dataset validation (default: False)
        curated_only: Load only curated knowledge, skip Kaggle (default: False)
        kaggle_sample_size: Number of Kaggle rows to sample (default: 500)
        chunk_size: Chunk size for text splitting (default: 800)

    Returns:
        dict: Statistics about the loaded knowledge base

    Example:
        >>> stats = build_and_load_knowledge_base()
        >>> print(f"Loaded {stats['total_documents']} documents")
    """
    print("=" * 70)
    print("🚀 Weather AI Agent - Knowledge Base Builder")
    print("=" * 70)
    print("Level 2: Basic RAG with Qdrant Vector Store")
    print("=" * 70)

    # Step 0: Validate Kaggle datasets (unless skipped)
    if not skip_validation and not curated_only:
        print("\n[Step 0/5] Validating Kaggle Datasets")
        print("-" * 70)
        try:
            validate_all_kaggle_datasets()
        except DatasetValidationError as e:
            print(f"\n❌ ABORT: {e}")
            print("Fix validation errors before continuing.")
            print("Or run with skip_validation=True to skip validation.")
            return {"error": str(e)}
    else:
        print("\n[Step 0/5] Skipping validation")
        if curated_only:
            print("   Reason: curated_only=True")
        else:
            print("   Reason: skip_validation=True")

    # Step 1: Load curated knowledge documents
    print("\n[Step 1/5] Loading Curated Weather Knowledge Documents")
    print("-" * 70)
    curated_docs = load_all_curated_documents()

    # Step 2: Load Kaggle documents (if enabled)
    kaggle_docs = []
    if not curated_only:
        print("\n[Step 2/5] Loading Kaggle Weather Datasets")
        print("-" * 70)
        kaggle_docs = load_all_kaggle_documents(
            daily_temp_sample_size=kaggle_sample_size, city_profile_count=100
        )
    else:
        print("\n[Step 2/5] Skipping Kaggle datasets (curated_only=True)")

    # Combine all documents
    all_documents = curated_docs + kaggle_docs

    if len(all_documents) == 0:
        print("\n❌ No documents loaded! Aborting.")
        return {"error": "No documents loaded"}

    # Step 3: Chunk documents
    print("\n[Step 3/5] Chunking Documents")
    print("-" * 70)
    chunked_docs = chunk_documents(all_documents, chunk_size=chunk_size)

    # Step 4: Initialize embeddings and vector store
    print("\n[Step 4/5] Initializing Vector Store")
    print("-" * 70)
    try:
        embeddings = create_embeddings()
        print("✅ OpenAI embeddings initialized (text-embedding-3-small)")
    except Exception as e:
        print(f"❌ Failed to initialize embeddings: {e}")
        print("   Make sure OPENAI_API_KEY is set in .env")
        return {"error": f"Embeddings initialization failed: {e}"}

    try:
        vectorstore = get_vector_store(embeddings=embeddings)
        print("✅ Qdrant vector store initialized")
    except Exception as e:
        print(f"❌ Failed to initialize vector store: {e}")
        print("   Make sure Qdrant is running: docker-compose up -d qdrant")
        return {"error": f"Vector store initialization failed: {e}"}

    # Step 5: Load documents into Qdrant
    print("\n[Step 5/5] Loading Documents into Qdrant")
    print("-" * 70)
    try:
        # Add documents in batches to avoid memory issues
        batch_size = 100
        total_loaded = 0

        for i in range(0, len(chunked_docs), batch_size):
            batch = chunked_docs[i : i + batch_size]
            vectorstore.add_documents(batch)
            total_loaded += len(batch)
            print(f"   Loaded batch {i // batch_size + 1}: {total_loaded}/{len(chunked_docs)} documents")

        print(f"\n✅ Successfully loaded {total_loaded} document chunks into Qdrant!")

    except Exception as e:
        print(f"❌ Failed to load documents: {e}")
        import traceback

        traceback.print_exc()
        return {"error": f"Document loading failed: {e}"}

    # Final statistics
    stats = {
        "total_documents": len(all_documents),
        "curated_documents": len(curated_docs),
        "kaggle_documents": len(kaggle_docs),
        "total_chunks": len(chunked_docs),
        "avg_chunk_size": sum(len(c.page_content) for c in chunked_docs)
        // len(chunked_docs),
        "collection_name": "weather_knowledge",
        "embedding_model": "text-embedding-3-small",
        "embedding_dimensions": 1536,
    }

    print("\n" + "=" * 70)
    print("✅ Knowledge Base Build Complete!")
    print("=" * 70)
    print(f"📊 Statistics:")
    print(f"   Total source documents: {stats['total_documents']}")
    print(f"   - Curated knowledge documents: {stats['curated_documents']}")
    print(f"   - Kaggle documents: {stats['kaggle_documents']}")
    print(f"   Total chunks loaded: {stats['total_chunks']}")
    print(f"   Average chunk size: {stats['avg_chunk_size']} characters")
    print(f"   Collection: {stats['collection_name']}")
    print(f"   Embedding model: {stats['embedding_model']} ({stats['embedding_dimensions']}-d)")
    print("=" * 70)
    print("\n🎉 Ready for RAG queries! Test with:")
    print('   vectorstore.similarity_search("What is a Category 5 hurricane?")')
    print("=" * 70)

    return stats


if __name__ == "__main__":
    # Run the build
    import sys

    # Parse command line args (simple version for Level 2)
    curated_only = "--curated-only" in sys.argv
    skip_validation = "--skip-validation" in sys.argv

    stats = build_and_load_knowledge_base(
        skip_validation=skip_validation, curated_only=curated_only
    )

    # Exit with error code if build failed
    if "error" in stats:
        sys.exit(1)
    else:
        sys.exit(0)
