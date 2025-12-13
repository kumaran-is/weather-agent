"""Document loaders for RAG knowledge base.

This module provides loaders for different data sources:
- Kaggle CSV datasets (numerical weather data → narrative documents)
- Curated text files (expert weather knowledge)
- Validation scripts (ensure data quality before loading)

Level 2 Loaders:
- kaggle_loader.py: Convert CSV rows to narrative documents
- curated_knowledge_loader.py: Load curated text documents
- validate_kaggle_datasets.py: Schema and quality validation
- csv_to_narrative.py: CSV row → narrative text conversion

Future Loaders (L5a):
- pdf_loader.py: Extract from weather PDFs
- web_scraper.py: Scrape weather websites
- api_loader.py: Fetch from weather APIs
"""

__all__ = []
