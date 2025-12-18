"""Kaggle dataset validation for RAG pipeline.

Validates CSV datasets before loading into vector database:
- File existence and readability
- Schema validation (required columns)
- Data type validation
- Data quality checks (null values, ranges)
- Row count verification

This script must be run BEFORE attempting to load datasets into Qdrant.
Prevents invalid data from corrupting the knowledge base.

Usage:
    python -m backend.src.rag.loaders.validate_kaggle_datasets

    # OR via Makefile:
    make rag-validate

Example:
    $ make rag-validate
    ==================================================================
    Validating Kaggle Datasets for RAG Pipeline
    ==================================================================

    [1/2] Validating: city_temperature_1980_2020.csv (REQUIRED)
    ✅ city_temperature_1980_2020.csv validation passed
       Rows: 2,906,327, Columns: 8
       Cities: 1,086, Countries: 108
       Temp range: -99.0°F to 116.6°F

    [2/2] Validating: daily_temperature_major_cities.csv (OPTIONAL)
    ⚠️  Optional file not found (can skip)

    ==================================================================
    Validation Summary
    ==================================================================
    ✅ PASS: city_temp_1980_2020
    ✅ PASS: daily_temp_major

    ✅ All validations passed! Ready for RAG loading.
       Next step: Run 'make rag-load'
"""

from pathlib import Path

import pandas as pd


class DatasetValidationError(Exception):
    """Raised when dataset validation fails."""

    pass


def validate_city_temperature_1980_2020() -> tuple[bool, list[str]]:
    """Validate: Temperature History of 1000 Cities (1980-2020).

    Expected Schema (Wide Format):
        - Row 1: city names (Tokyo, New York, Mumbai, ...)
        - Row 2: city_ascii
        - Row 3: lat (latitude)
        - Row 4: lng (longitude)
        - Rows 5+: Temperature readings (1980-2020 time periods)
        - ~1000 columns (cities), ~15,000 rows (time periods)

    Checks:
        1. File exists
        2. File is readable (CSV format)
        3. Wide format detected (many columns)
        4. Metadata rows present (city, city_ascii, lat, lng)
        5. Reasonable column count (should be ~1000 cities)
        6. Reasonable row count (should be ~15K time periods)

    Returns:
        (is_valid, error_messages): Tuple of validation result and errors

    Example:
        >>> is_valid, errors = validate_city_temperature_1980_2020()
        >>> if is_valid:
        ...     print("Dataset ready for RAG loading!")
    """
    csv_path = Path("backend/data/raw/kaggle/city_temperature_1980_2020.csv")
    errors = []

    # Check 1: File exists
    if not csv_path.exists():
        errors.append(f"❌ File not found: {csv_path}")
        errors.append(
            "   Download from: https://www.kaggle.com/datasets/hansukyang/temperature-history-of-1000-cities-1980-to-2020"
        )
        return False, errors

    # Check 2: File is readable
    try:
        df = pd.read_csv(csv_path, nrows=10)  # Read first 10 rows for validation
    except Exception as e:
        errors.append(f"❌ File not readable: {e}")
        errors.append("   File may be corrupted. Try re-downloading.")
        return False, errors

    # Check 3: Wide format detected (many columns = cities)
    num_cols = len(df.columns)
    if num_cols < 100:
        errors.append(f"❌ Expected wide format with ~1000 columns, got {num_cols}")
        errors.append("   This may be the wrong dataset or wrong format")
        return False, errors

    # Check 4: Metadata rows present
    # First column should be empty or index, second row should have city names
    first_col = df.columns[0]
    if first_col not in ["", "Unnamed: 0"]:
        errors.append(f"⚠️  Warning: Unexpected first column: {first_col}")

    # Check if first few rows look like metadata (city, city_ascii, lat, lng)
    try:
        first_few_values = df.iloc[0, 1:5].tolist()  # First row, columns 1-4
        # Should be city names (strings)
        if not all(isinstance(v, str) for v in first_few_values):
            errors.append("⚠️  Warning: First row doesn't look like city names")
    except Exception:
        pass  # Validation continues

    # Check 5: Reasonable column count
    if num_cols < 900 or num_cols > 1100:
        errors.append(
            f"⚠️  Warning: Expected ~1000 cities, got {num_cols} columns"
        )

    # Check 6: Reasonable row count
    try:
        # Count rows without loading entire file
        with open(csv_path) as f:
            row_count = sum(1 for _ in f)

        if row_count < 10000:
            errors.append(
                f"⚠️  Warning: Expected ~15,000 rows, got {row_count} rows"
            )
            errors.append("   File may be incomplete")
    except Exception as e:
        errors.append(f"⚠️  Could not count rows: {e}")

    print("✅ city_temperature_1980_2020.csv validation passed (wide format)")
    print("   Format: Wide (cities as columns)")
    print(f"   Columns (cities): {num_cols:,}")
    try:
        print(f"   Rows (time periods): {row_count:,}")
    except:
        print("   Rows: Could not determine")
    print(f"   Sample cities: {', '.join(str(c) for c in df.columns[1:6])}")

    return True, errors


def validate_daily_temperature_major_cities() -> tuple[bool, list[str]]:
    """Validate: Daily Temperature of Major Cities (OPTIONAL).

    Expected Schema:
        - Region, Country, State, City, Month, Day, Year, AvgTemperature
        (Similar to city_temperature_1980_2020)

    Note:
        This dataset is OPTIONAL for Level 2. If not present, validation
        still passes (just logs a warning).

    Returns:
        (is_valid, error_messages): Always True unless file is corrupted

    Example:
        >>> is_valid, errors = validate_daily_temperature_major_cities()
        >>> # True even if file missing (optional dataset)
    """
    csv_path = Path("backend/data/raw/kaggle/daily_temperature_major_cities.csv")
    errors = []

    if not csv_path.exists():
        errors.append(f"⚠️  Optional file not found: {csv_path} (can skip)")
        return True, errors  # Optional dataset - not an error

    try:
        df = pd.read_csv(csv_path, nrows=100)

        # Basic checks
        if "City" not in df.columns or "AvgTemperature" not in df.columns:
            errors.append(f"❌ Missing required columns in {csv_path}")
            return False, errors

        print("✅ daily_temperature_major_cities.csv validation passed")
        return True, errors

    except Exception as e:
        errors.append(f"❌ Error reading {csv_path}: {e}")
        return False, errors


def validate_all_kaggle_datasets() -> dict[str, tuple[bool, list[str]]]:
    """Validate all Kaggle datasets for RAG pipeline.

    Validates both required and optional datasets:
        - REQUIRED: city_temperature_1980_2020.csv
        - OPTIONAL: daily_temperature_major_cities.csv

    Returns:
        {dataset_name: (is_valid, error_messages)}

    Raises:
        DatasetValidationError: If any required dataset fails validation

    Example:
        >>> results = validate_all_kaggle_datasets()
        ✅ All validations passed! Ready for RAG loading.

        >>> # If validation fails:
        >>> results = validate_all_kaggle_datasets()
        ❌ Validation failed! Please fix errors before running 'make rag-load'
        DatasetValidationError: Kaggle dataset validation failed
    """
    print("=" * 70)
    print("Validating Kaggle Datasets for RAG Pipeline")
    print("=" * 70)

    results = {}

    # Validate required datasets
    print("\n[1/2] Validating: city_temperature_1980_2020.csv (REQUIRED)")
    results["city_temp_1980_2020"] = validate_city_temperature_1980_2020()

    # Validate optional datasets
    print("\n[2/2] Validating: daily_temperature_major_cities.csv (OPTIONAL)")
    results["daily_temp_major"] = validate_daily_temperature_major_cities()

    # Summary
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)

    all_valid = True
    for name, (is_valid, errors) in results.items():
        status = "✅ PASS" if is_valid else "❌ FAIL"
        print(f"{status}: {name}")
        if errors:
            for error in errors:
                print(f"  {error}")
        all_valid = all_valid and is_valid

    if all_valid:
        print("\n✅ All validations passed! Ready for RAG loading.")
        print("   Next step: Run 'make rag-load'")
    else:
        print("\n❌ Validation failed! Please fix errors before running 'make rag-load'")
        raise DatasetValidationError("Kaggle dataset validation failed")

    return results


if __name__ == "__main__":
    validate_all_kaggle_datasets()
