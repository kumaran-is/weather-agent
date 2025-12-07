"""Kaggle weather dataset loader with CSV-to-narrative conversion.

Loads Kaggle weather CSV files and converts rows to narrative documents
suitable for RAG semantic search.

Level 2 Implementation (MVP):
- Load sample of rows (not all 2.9M - too large for L2)
- Convert to narrative text using csv_to_narrative
- Create documents with metadata
- Target: ~500-1000 documents from Kaggle data

Future Enhancements (L5a):
- Full dataset loading with batching
- Aggregation and summarization
- Multi-resolution indexing (daily, monthly, yearly)
- City-specific climate profiles

Supported datasets:
1. daily_temperature_major_cities.csv (long format)
   - Schema: Region, Country, State, City, Month, Day, Year, AvgTemperature
2. city_temperature_1980_2020.csv (wide format)
   - Schema: Cities as columns, time periods as rows
"""

from langchain_core.documents import Document
from pathlib import Path
import pandas as pd
import numpy as np
from backend.src.rag.loaders.csv_to_narrative import csv_row_to_narrative


def load_daily_temperature_sample(
    max_rows: int = 1000, random_sample: bool = True
) -> list[Document]:
    """Load sample from daily_temperature_major_cities.csv.

    Level 2: Loads a sample (not full 2.9M rows) to keep RAG knowledge base
    manageable for MVP.

    Args:
        max_rows: Maximum number of rows to load (default: 1000)
        random_sample: If True, random sample; if False, first N rows

    Returns:
        list[Document]: Narrative documents with metadata

    Example:
        >>> docs = load_daily_temperature_sample(max_rows=100)
        >>> print(f"Loaded {len(docs)} documents")
        Loaded 100 documents
        >>> print(docs[0].page_content[:100])
        In Tokyo, Japan, the average temperature in January 2020 was 45.2°F...
    """
    csv_path = Path("backend/data/raw/kaggle/daily_temperature_major_cities.csv")

    if not csv_path.exists():
        print(f"⚠️  File not found: {csv_path}")
        return []

    print(f"📂 Loading daily_temperature_major_cities.csv...")

    try:
        # Load sample of data
        if random_sample:
            # Read just enough to get total rows
            total_rows = sum(1 for _ in open(csv_path)) - 1  # Minus header

            # Calculate skip rows for random sampling
            skip = sorted(
                np.random.choice(range(1, total_rows + 1), total_rows - max_rows, replace=False)
            )
            df = pd.read_csv(csv_path, skiprows=skip)
            print(f"   Random sample: {len(df)} rows from {total_rows:,} total")
        else:
            # Just read first N rows
            df = pd.read_csv(csv_path, nrows=max_rows)
            print(f"   First {len(df)} rows")

        # Convert rows to narrative documents
        documents = []
        for idx, row in df.iterrows():
            # Convert row to dictionary
            row_dict = row.to_dict()

            # Create narrative text
            narrative = csv_row_to_narrative(row_dict)

            # Create document with metadata
            doc = Document(
                page_content=narrative,
                metadata={
                    "source": "daily_temperature_major_cities.csv",
                    "city": row_dict.get("City", "Unknown"),
                    "country": row_dict.get("Country", "Unknown"),
                    "year": row_dict.get("Year"),
                    "month": row_dict.get("Month"),
                    "type": "daily_temperature",
                },
            )
            documents.append(doc)

        print(f"✅ Converted {len(documents)} rows to narrative documents")
        return documents

    except Exception as e:
        print(f"❌ Error loading daily_temperature_major_cities.csv: {e}")
        return []


def load_city_temperature_1980_2020_sample(max_cities: int = 50) -> list[Document]:
    """Load sample from city_temperature_1980_2020.csv (wide format).

    Creates city climate profile documents from the wide-format dataset.

    Level 2: Loads subset of cities to keep RAG knowledge base manageable.

    Args:
        max_cities: Maximum number of cities to include (default: 50)

    Returns:
        list[Document]: City climate profile documents

    Example:
        >>> docs = load_city_temperature_1980_2020_sample(max_cities=20)
        >>> print(f"Loaded {len(docs)} city profiles")
        Loaded 20 city profiles
    """
    csv_path = Path("backend/data/raw/kaggle/city_temperature_1980_2020.csv")

    if not csv_path.exists():
        print(f"⚠️  File not found: {csv_path}")
        return []

    print(f"📂 Loading city_temperature_1980_2020.csv (wide format)...")

    try:
        # Read CSV with more rows for temperature data
        # CSV structure: 12 metadata rows (city, city_ascii, lat, lng, country, iso2, iso3,
        # admin_name, capital, population, id, datetime) then temperature data from 1980-01-01
        df = pd.read_csv(csv_path, nrows=500)  # Read more rows for better statistics

        # Extract city names from columns (skip first column which is row labels)
        city_columns = df.columns[1 : min(max_cities + 1, len(df.columns))]

        print(
            f"   Processing {len(city_columns)} cities from {len(df.columns) - 1} total"
        )

        documents = []

        for city_col in city_columns:
            try:
                # Get city name from first row
                city_name = df.iloc[0][city_col]

                # Skip if not a valid city name
                if pd.isna(city_name) or not isinstance(city_name, str):
                    continue

                # Get temperature data for this city
                # Skip first 12 rows (all metadata: city, city_ascii, lat, lng, country,
                # iso2, iso3, admin_name, capital, population, id, datetime)
                # Row 12 onwards contains actual temperature data starting from 1980-01-01
                temps = pd.to_numeric(df.iloc[12:][city_col], errors='coerce').dropna()

                if len(temps) == 0:
                    continue

                # Calculate basic statistics (temperatures are in Celsius in the CSV)
                # Convert Celsius to Fahrenheit for consistency
                avg_temp_c = temps.mean()
                min_temp_c = temps.min()
                max_temp_c = temps.max()

                # Convert to Fahrenheit: F = C * 9/5 + 32
                avg_temp = avg_temp_c * 9/5 + 32
                min_temp = min_temp_c * 9/5 + 32
                max_temp = max_temp_c * 9/5 + 32
            except Exception as e:
                # Skip problematic cities
                print(f"   ⚠️  Skipping city {city_col}: {e}")
                continue

            # Create narrative
            narrative = f"Climate profile for {city_name}: "
            narrative += f"Over the historical period (1980-2020), {city_name} had an average temperature of {avg_temp:.1f}°F ({avg_temp_c:.1f}°C). "
            narrative += f"The coldest recorded temperature was {min_temp:.1f}°F ({min_temp_c:.1f}°C), "
            narrative += f"and the hottest was {max_temp:.1f}°F ({max_temp_c:.1f}°C), "
            narrative += f"showing a temperature range of {max_temp - min_temp:.1f}°F. "

            # Add climate classification (using Fahrenheit thresholds)
            if avg_temp < 40:
                climate = "cold continental climate"
            elif avg_temp < 55:
                climate = "temperate climate"
            elif avg_temp < 70:
                climate = "warm temperate climate"
            elif avg_temp < 80:
                climate = "subtropical climate"
            else:
                climate = "tropical or hot desert climate"

            narrative += f"This indicates a {climate}."

            # Create document
            doc = Document(
                page_content=narrative,
                metadata={
                    "source": "city_temperature_1980_2020.csv",
                    "city": city_name,
                    "avg_temp": float(avg_temp),
                    "min_temp": float(min_temp),
                    "max_temp": float(max_temp),
                    "type": "city_climate_profile",
                },
            )
            documents.append(doc)

        print(f"✅ Created {len(documents)} city climate profiles")
        return documents

    except Exception as e:
        print(f"❌ Error loading city_temperature_1980_2020.csv: {e}")
        import traceback

        traceback.print_exc()
        return []


def load_all_kaggle_documents(
    daily_temp_sample_size: int = 500, city_profile_count: int = 100
) -> list[Document]:
    """Load all Kaggle datasets with sampling.

    Level 2 Configuration:
        - daily_temp_sample_size: 500 rows from daily temperatures
        - city_profile_count: 100 city climate profiles
        - Total: ~600 documents from Kaggle data

    Args:
        daily_temp_sample_size: Number of daily temperature records to sample
        city_profile_count: Number of city profiles to create

    Returns:
        list[Document]: Combined Kaggle documents

    Example:
        >>> docs = load_all_kaggle_documents()
        >>> print(f"Total Kaggle documents: {len(docs)}")
        Total Kaggle documents: 600
    """
    print("=" * 70)
    print("Loading Kaggle Weather Datasets")
    print("=" * 70)

    all_documents = []

    # Load daily temperature sample
    print("\n[1/2] Daily Temperature Records")
    daily_docs = load_daily_temperature_sample(
        max_rows=daily_temp_sample_size, random_sample=True
    )
    all_documents.extend(daily_docs)

    # Load city climate profiles
    print("\n[2/2] City Climate Profiles")
    city_docs = load_city_temperature_1980_2020_sample(max_cities=city_profile_count)
    all_documents.extend(city_docs)

    print("\n" + "=" * 70)
    print(f"✅ Total Kaggle documents loaded: {len(all_documents)}")
    print("=" * 70)
    print(f"   Daily temperature records: {len(daily_docs)}")
    print(f"   City climate profiles: {len(city_docs)}")

    return all_documents


if __name__ == "__main__":
    # Test loading
    docs = load_all_kaggle_documents(
        daily_temp_sample_size=10, city_profile_count=5  # Small sample for testing
    )

    # Show samples
    if docs:
        print("\n" + "=" * 70)
        print("Sample Documents")
        print("=" * 70)

        # Show daily temperature sample
        daily_sample = [d for d in docs if d.metadata["type"] == "daily_temperature"][
            :2
        ]
        for doc in daily_sample:
            print(f"\nDaily Temperature Record:")
            print(f"  City: {doc.metadata['city']}")
            print(f"  Content: {doc.page_content[:150]}...")

        # Show city profile sample
        city_sample = [
            d for d in docs if d.metadata["type"] == "city_climate_profile"
        ][:2]
        for doc in city_sample:
            print(f"\nCity Climate Profile:")
            print(f"  City: {doc.metadata['city']}")
            print(f"  Avg Temp: {doc.metadata['avg_temp']:.1f}°F")
            print(f"  Content: {doc.page_content[:150]}...")
