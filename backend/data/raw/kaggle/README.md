# Kaggle Datasets for Weather AI Agent - RAG Knowledge Base

This directory contains Kaggle weather datasets for the RAG knowledge base.

## Required Datasets

### 1. Temperature History of 1000 Cities (1980-2020)
**URL**: https://www.kaggle.com/datasets/hansukyang/temperature-history-of-1000-cities-1980-to-2020

**File**: `city_temperature_1980_2020.csv` (2.8 MB, ~2.9M rows)

**Schema**:
- Region, Country, State, City, Month, Day, Year, AvgTemperature

**RAG Use**: City climate profiles, seasonal patterns

**Download Instructions**:
1. Go to https://www.kaggle.com/datasets/hansukyang/temperature-history-of-1000-cities-1980-to-2020
2. Click "Download" button (requires Kaggle account)
3. Rename file to `city_temperature_1980_2020.csv`
4. Place in this directory (`backend/data/raw/kaggle/`)

---

### 2. Daily Temperature of Major Cities (OPTIONAL)
**URL**: https://www.kaggle.com/datasets/sudalairajkumar/daily-temperature-of-major-cities

**File**: `daily_temperature_major_cities.csv` (22 MB)

**Schema**: Similar to #1

**RAG Use**: Recent weather trends

**Download Instructions**:
1. Go to URL above
2. Download and rename file
3. Place in this directory

---

### 3. Global Daily Climate Data (OPTIONAL)
**URL**: https://www.kaggle.com/datasets/guillemservera/global-daily-climate-data

**File**: `global_daily_climate.csv` (50+ MB)

**RAG Use**: Global weather patterns

---

## Validation

After downloading datasets, validate them before loading:

```bash
make rag-validate
```

Expected output:
```
✅ city_temperature_1980_2020.csv validation passed
   Rows: 2,906,327, Columns: 8
   Cities: 1,086, Countries: 108
   Temp range: -99.0°F to 116.6°F
```

## Loading into Qdrant

Once validated, load into Qdrant vector store:

```bash
make rag-load
```

This will:
1. Convert CSV rows to narrative documents
2. Apply variable chunking (400-1000 chars by category)
3. Generate embeddings (text-embedding-3-small)
4. Load into Qdrant `weather_knowledge` collection

## File Size Expectations

- `city_temperature_1980_2020.csv`: ~2.8 MB (REQUIRED)
- `daily_temperature_major_cities.csv`: ~22 MB (optional)
- `global_daily_climate.csv`: ~50 MB (optional)

**Total**: Minimum 2.8 MB (required only), up to ~75 MB (all datasets)

## Notes

- **Level 2**: Only `city_temperature_1980_2020.csv` is required
- **Level 5a**: Additional datasets for expanded knowledge base
- CSV files are NOT committed to git (see `.gitignore`)
- Datasets must be manually downloaded (Kaggle ToS)
