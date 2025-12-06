# RAG Datasets Guide

**Weather AI Agent Service Knowledge Base**

Complete guide to mock and Kaggle datasets used for RAG semantic search.

---

## Table of Contents

- [Overview](#overview)
- [Mock Data](#mock-data)
- [Kaggle Datasets](#kaggle-datasets)
- [Dataset Statistics](#dataset-statistics)
- [How to Download](#how-to-download)

---

## Overview

The Weather AI Agent uses a **dual-source knowledge base** combining curated mock data with real-world Kaggle datasets for comprehensive weather intelligence:

| Source | Files | Documents | Purpose |
|--------|-------|-----------|---------|
| **Mock Data** | 3 | 3 | Hurricane safety, weather terminology |
| **Kaggle Data** | 2 CSVs | 600 | Historical temperature data (1980-2020) |
| **Total** | 5 | **603** | Embedded into **632 chunks** in Qdrant |

---

## Mock Data

**Location**: `backend/data/raw/mock/`

Hand-curated text documents providing authoritative weather safety information.

### Files

| File | Category | Content |
|------|----------|---------|
| `saffir_simpson_scale.txt` | Hurricanes | Complete Saffir-Simpson hurricane categories (Cat 1-5), wind speeds, damage descriptions, storm surge, evacuation requirements, historical examples (Katrina, Michael, Milton) |
| `evacuation_zones.txt` | Hurricanes | Evacuation zones A-E with risk levels, timing (24-72 hours before landfall), special categories (mobile homes, high-rises), shelter information |
| `heat_index.txt` | Weather Terminology | Heat index calculations, temperature + humidity tables, health risk categories (Caution, Extreme Danger), safety tips, heat illness recognition |

**Purpose**: Provide accurate, life-safety information for hurricane alerts and weather terminology queries.

**Format**: Plain text narratives optimized for semantic search.

---

## Kaggle Datasets

**Location**: `backend/data/raw/kaggle/`

Real-world historical weather data from major cities worldwide (1980-2020).

### Dataset 1: Daily Temperature Major Cities

**File**: `daily_temperature_major_cities.csv`
**Source**: [Kaggle - Daily Temperature of Major Cities](https://www.kaggle.com/datasets/sudalairajkumar/daily-temperature-of-major-cities)
**Size**: 2.9M rows → **500 sampled** (random sampling)

**Schema (Long Format)**:
```
Region, Country, State, City, Month, Day, Year, AvgTemperature
```

**Sample Row**:
```
North America, USA, California, Los Angeles, 7, 15, 2020, 75.2
```

**Narrative Conversion**:
```
"In Los Angeles, USA, the average temperature in July 2020 was 75.2°F,
which is mild to warm. This is typical summer weather for the region."
```

**Purpose**: Answer queries about specific cities, dates, and seasonal patterns.

### Dataset 2: City Temperature 1980-2020

**File**: `city_temperature_1980_2020.csv`
**Source**: [Kaggle - Temperature History of 1000 Cities](https://www.kaggle.com/datasets/hansukyang/temperature-history-of-1000-cities-1980-to-2020)
**Size**: 1000 cities × 14,897 time periods → **100 city profiles created**

**Schema (Wide Format)**:
```
Row 1: City names (Tokyo, New York, Mumbai, ...)
Row 2: city_ascii
Row 3: lat (latitude)
Row 4: lng (longitude)
Rows 5+: Temperature readings (1980-2020)
```

**Climate Profile Generation**:
```
"Climate profile for Tokyo: Over the historical period (1980-2020),
Tokyo had an average temperature of 59.2°F. The coldest recorded
temperature was 28.4°F, and the hottest was 95.7°F, showing a
temperature range of 67.3°F. This indicates a temperate climate."
```

**Purpose**: Provide long-term climate profiles and temperature statistics for cities worldwide.

---

## Dataset Statistics

### Loading Summary

```
📊 Source Documents: 603
   ├─ Mock documents: 3 (hurricanes, weather terminology)
   └─ Kaggle documents: 600
      ├─ Daily temperature records: 500 (random sample)
      └─ City climate profiles: 100

📄 Total Chunks: 632 (avg 189 characters/chunk)

🔢 Embeddings: OpenAI text-embedding-3-small (1536 dimensions)

💾 Storage: Qdrant collection "weather_knowledge" (Cosine distance)
```

### Coverage

| Data Type | Count | Coverage |
|-----------|-------|----------|
| Cities Covered | 100+ | Global (Tokyo, NYC, Mumbai, London, etc.) |
| Time Period | 40 years | 1980-2020 |
| Daily Records | 500 | Random sample from 2.9M total |
| Hurricane Categories | 5 | Saffir-Simpson Cat 1-5 |
| Evacuation Zones | 5 | Zones A-E with detailed guidance |

---

## How to Download

> **Why aren't the CSV files in the repository?**
> The Kaggle datasets (216 MB total) are excluded from Git for several reasons:
> - **GitHub limits**: Files exceed GitHub's 100 MB file size limit
> - **Kaggle terms**: Users must download directly from Kaggle (proper attribution & licensing)
> - **Repository size**: Keeps the repo lightweight (<10 MB vs 216 MB)
> - **Latest data**: Ensures users get the most recent data from Kaggle
>
> **Don't worry!** The download process below takes ~2-3 minutes.

### Option 1: Kaggle CLI (Recommended)

**1. Install Kaggle CLI**:
```bash
pip install kaggle
```

**2. Get API Token**:
- Go to https://www.kaggle.com/settings
- Click "Create New API Token"
- Download `kaggle.json` to `~/.kaggle/`

**3. Download Datasets**:
```bash
# Daily Temperature Major Cities
kaggle datasets download -d sudalairajkumar/daily-temperature-of-major-cities
unzip daily-temperature-of-major-cities.zip -d backend/data/raw/kaggle/

# Temperature History 1000 Cities
kaggle datasets download -d hansukyang/temperature-history-of-1000-cities-1980-to-2020
unzip temperature-history-of-1000-cities-1980-to-2020.zip -d backend/data/raw/kaggle/
```

### Option 2: Manual Download

1. **Daily Temperature Major Cities**:
   - Visit: https://www.kaggle.com/datasets/sudalairajkumar/daily-temperature-of-major-cities
   - Download CSV
   - Rename to: `daily_temperature_major_cities.csv`
   - Place in: `backend/data/raw/kaggle/`

2. **Temperature History 1000 Cities**:
   - Visit: https://www.kaggle.com/datasets/hansukyang/temperature-history-of-1000-cities-1980-to-2020
   - Download CSV
   - Rename to: `city_temperature_1980_2020.csv`
   - Place in: `backend/data/raw/kaggle/`

### Verify Downloads

```bash
# Validate datasets
make rag-validate

# Expected output:
# ✅ PASS: city_temp_1980_2020
# ✅ PASS: daily_temp_major
# ✅ All validations passed!
```

---

## Next Steps

After downloading datasets:

```bash
# 1. Validate
make rag-validate

# 2. Load into Qdrant
make rag-load

# 3. Test retrieval
make rag-test
```

See [RAG Commands in README](../../README.md#rag-commands-level-2-qdrant-vector-database) for complete workflow.

---

**Last Updated**: 2025-12-04
**Level**: 2 (RAG with Qdrant)
