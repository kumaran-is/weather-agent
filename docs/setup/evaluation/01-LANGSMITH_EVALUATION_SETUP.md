# Step 1: LangSmith Evaluation Setup

**Purpose**: Set up LangSmith for evaluation dataset management and result tracking
**Time Required**: 15-20 minutes

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [LangSmith Account Setup](#langsmith-account-setup)
3. [API Key Configuration](#api-key-configuration)
4. [Upload Golden Dataset](#upload-golden-dataset)
5. [Verify Setup](#verify-setup)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, ensure you have:

- [ ] Python 3.13+ installed
- [ ] Project dependencies installed (`uv sync`)
- [ ] Access to Weather AI Agent codebase
- [ ] Internet connection (for LangSmith API)

---

## LangSmith Account Setup

### Step 1.1: Create LangSmith Account

1. Visit [smith.langchain.com](https://smith.langchain.com)
2. Sign up with:
   - GitHub account (recommended)
   - Google account
   - Email/password

3. Verify your email if required

### Step 1.2: Create Organization & Project

1. After login, create an organization (or use personal workspace)
2. Create a new project:
   - Name: `weather-ai-agent`
   - Description: `Weather AI Agent Service - Level 5 Evaluation`

### Step 1.3: Generate API Key

1. Click your profile icon (top-right)
2. Go to **Settings** → **API Keys**
3. Click **Create API Key**
4. Name: `weather-ai-agent-dev`
5. Copy the key (format: `lsv2_pt_xxxxxxxxxxxx`)

> ⚠️ **Security**: Never commit API keys to git. Use `.env` files only.

---

## API Key Configuration

### Step 2.1: Update .env File

```bash
# In project root, edit .env file
cd /home/user/weather-ai-agent-service

# Add/update these environment variables
cat >> .env << 'EOF'

# ============================================================================
# LangSmith Configuration (Level 5 Evaluation)
# ============================================================================
LANGCHAIN_API_KEY=lsv2_pt_your_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=weather-ai-agent
LANGSMITH_DATASET_NAME=weather-ai-golden-dataset

# Optional: Enable detailed tracing
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
EOF
```

### Step 2.2: Verify Environment Variables

```bash
# Verify configuration (without exposing full key)
grep LANGCHAIN .env | sed 's/=.*/=***/'

# Expected output:
# LANGCHAIN_API_KEY=***
# LANGCHAIN_TRACING_V2=***
# LANGCHAIN_PROJECT=***
```

### Step 2.3: Test Connection

```python
# tests/evaluation/test_langsmith_connection.py
"""Quick test to verify LangSmith connection."""

import os
from langsmith import Client

def test_langsmith_connection():
    """Verify LangSmith API connectivity."""
    client = Client()

    # Test connection by listing datasets
    datasets = list(client.list_datasets(limit=1))
    print(f"✅ LangSmith connected. Found {len(datasets)} datasets.")

    return True

if __name__ == "__main__":
    test_langsmith_connection()
```

Run the test:

```bash
uv run python tests/evaluation/test_langsmith_connection.py
# Expected: ✅ LangSmith connected. Found X datasets.
```

---

## Upload Golden Dataset

### Step 3.1: Understand Dataset Structure

The golden dataset is located at:
```
tests/evaluation/golden_dataset.yaml
```

Contains **105 test cases** across 4 categories:
- **Simple** (40 cases): Basic weather queries
- **Complex** (30 cases): Multi-location, analysis queries
- **Hurricane** (20 cases): Safety-critical hurricane queries
- **Edge** (15 cases): Error handling, invalid inputs

### Step 3.2: Create Dataset Upload Script

```python
# scripts/upload_golden_dataset.py
"""Upload golden dataset to LangSmith for evaluation tracking."""

import yaml
from pathlib import Path
from langsmith import Client
from datetime import datetime

# Configuration
DATASET_PATH = Path("tests/evaluation/golden_dataset.yaml")
DATASET_NAME = "weather-ai-golden-dataset"
DATASET_DESCRIPTION = """
Weather AI Agent Golden Dataset (Level 5b)
- 105 test cases for regression testing
- Categories: simple, complex, hurricane, edge
- Used for 4-pillar evaluation (effectiveness, efficiency, robustness, safety)
"""

def load_golden_dataset(path: Path) -> list[dict]:
    """Load test cases from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("golden_dataset", [])

def upload_to_langsmith(test_cases: list[dict]) -> str:
    """Upload dataset to LangSmith."""
    client = Client()

    # Check if dataset exists
    existing = list(client.list_datasets(dataset_name=DATASET_NAME))

    if existing:
        print(f"⚠️ Dataset '{DATASET_NAME}' already exists. Updating...")
        dataset = existing[0]
        # Delete existing examples for clean upload
        examples = list(client.list_examples(dataset_id=dataset.id))
        for ex in examples:
            client.delete_example(ex.id)
        print(f"   Deleted {len(examples)} existing examples")
    else:
        print(f"📦 Creating new dataset: {DATASET_NAME}")
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=DATASET_DESCRIPTION,
        )

    # Upload each test case as an example
    examples_created = 0
    for case in test_cases:
        # Create input/output structure
        inputs = {
            "query": case["query"],
            "test_id": case["id"],
            "category": case.get("category", "unknown"),
        }

        outputs = {
            "expected_tools": case.get("expected_tools", []),
            "expected_answer_contains": case.get("expected_answer_contains", []),
            "expected_answer": case.get("expected_answer"),
        }

        metadata = {
            "success_criteria": case.get("success_criteria", {}),
            "is_safety_critical": case.get("is_safety_critical", False),
            "is_edge_case": case.get("is_edge_case", False),
        }

        client.create_example(
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            dataset_id=dataset.id,
        )
        examples_created += 1

    print(f"✅ Uploaded {examples_created} test cases to LangSmith")
    print(f"   Dataset URL: https://smith.langchain.com/datasets/{dataset.id}")

    return str(dataset.id)

def main():
    """Main upload workflow."""
    print("=" * 60)
    print("GOLDEN DATASET UPLOAD TO LANGSMITH")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Load dataset
    print(f"📄 Loading dataset from: {DATASET_PATH}")
    test_cases = load_golden_dataset(DATASET_PATH)
    print(f"   Found {len(test_cases)} test cases")

    # Categorize
    categories = {}
    for case in test_cases:
        cat = case.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print("\n📊 Category Breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"   - {cat}: {count}")

    # Upload
    print()
    dataset_id = upload_to_langsmith(test_cases)

    print()
    print("=" * 60)
    print("UPLOAD COMPLETE")
    print("=" * 60)
    print(f"Dataset ID: {dataset_id}")
    print(f"View at: https://smith.langchain.com/datasets")

if __name__ == "__main__":
    main()
```

### Step 3.3: Run Upload Script

```bash
# Ensure virtual environment is activated
cd /home/user/weather-ai-agent-service

# Run upload script
uv run python scripts/upload_golden_dataset.py

# Expected output:
# ============================================================
# GOLDEN DATASET UPLOAD TO LANGSMITH
# ============================================================
# Timestamp: 2025-12-11T...
#
# 📄 Loading dataset from: tests/evaluation/golden_dataset.yaml
#    Found 105 test cases
#
# 📊 Category Breakdown:
#    - complex: 30
#    - edge: 15
#    - hurricane: 20
#    - simple: 40
#
# 📦 Creating new dataset: weather-ai-golden-dataset
# ✅ Uploaded 105 test cases to LangSmith
#    Dataset URL: https://smith.langchain.com/datasets/xxx
#
# ============================================================
# UPLOAD COMPLETE
# ============================================================
```

### Step 3.4: Verify in LangSmith UI

1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Navigate to **Datasets** in left sidebar
3. Find `weather-ai-golden-dataset`
4. Verify:
   - 105 examples present
   - Categories visible in metadata
   - Input/output structure correct

---

## Verify Setup

### Step 4.1: Run Verification Checklist

```bash
# Create verification script
cat > scripts/verify_langsmith_setup.py << 'EOF'
"""Verify LangSmith setup is complete and working."""

import os
from langsmith import Client

def verify_setup():
    """Run all verification checks."""
    results = {}

    # 1. API Key present
    api_key = os.getenv("LANGCHAIN_API_KEY")
    results["api_key_present"] = bool(api_key and api_key.startswith("lsv2_"))

    # 2. Tracing enabled
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    results["tracing_enabled"] = tracing

    # 3. Project configured
    project = os.getenv("LANGCHAIN_PROJECT")
    results["project_configured"] = bool(project)

    # 4. Client connection
    try:
        client = Client()
        datasets = list(client.list_datasets(limit=1))
        results["client_connected"] = True
    except Exception as e:
        results["client_connected"] = False
        results["connection_error"] = str(e)

    # 5. Golden dataset exists
    try:
        datasets = list(client.list_datasets(dataset_name="weather-ai-golden-dataset"))
        if datasets:
            examples = list(client.list_examples(dataset_id=datasets[0].id))
            results["golden_dataset_exists"] = True
            results["golden_dataset_size"] = len(examples)
        else:
            results["golden_dataset_exists"] = False
    except Exception as e:
        results["golden_dataset_exists"] = False
        results["dataset_error"] = str(e)

    return results

def print_results(results: dict):
    """Print verification results."""
    print("\n" + "=" * 60)
    print("LANGSMITH SETUP VERIFICATION")
    print("=" * 60 + "\n")

    checks = [
        ("API Key Present", "api_key_present"),
        ("Tracing Enabled", "tracing_enabled"),
        ("Project Configured", "project_configured"),
        ("Client Connected", "client_connected"),
        ("Golden Dataset Exists", "golden_dataset_exists"),
    ]

    all_passed = True
    for label, key in checks:
        status = results.get(key, False)
        icon = "✅" if status else "❌"
        print(f"  {icon} {label}")
        if not status:
            all_passed = False

    if "golden_dataset_size" in results:
        print(f"\n  📊 Dataset Size: {results['golden_dataset_size']} examples")

    print("\n" + "-" * 60)
    if all_passed:
        print("🎉 All checks passed! LangSmith is ready for evaluation.")
    else:
        print("⚠️ Some checks failed. Review the issues above.")
    print("-" * 60 + "\n")

    return all_passed

if __name__ == "__main__":
    results = verify_setup()
    success = print_results(results)
    exit(0 if success else 1)
EOF

# Run verification
uv run python scripts/verify_langsmith_setup.py
```

### Step 4.2: Expected Output

```
============================================================
LANGSMITH SETUP VERIFICATION
============================================================

  ✅ API Key Present
  ✅ Tracing Enabled
  ✅ Project Configured
  ✅ Client Connected
  ✅ Golden Dataset Exists

  📊 Dataset Size: 105 examples

------------------------------------------------------------
🎉 All checks passed! LangSmith is ready for evaluation.
------------------------------------------------------------
```

---

## Troubleshooting

### Issue 1: "LANGCHAIN_API_KEY not found"

**Symptom**: `ValidationError: LANGCHAIN_API_KEY environment variable not set`

**Solution**:
```bash
# Check .env file
cat .env | grep LANGCHAIN_API_KEY

# If missing, add it
echo "LANGCHAIN_API_KEY=lsv2_pt_your_key_here" >> .env

# Reload environment (or restart terminal)
source .env
```

### Issue 2: "Connection refused" or timeout

**Symptom**: Client cannot connect to LangSmith API

**Solution**:
```bash
# Check network connectivity
curl -s https://api.smith.langchain.com/health

# If blocked by firewall/proxy, set proxy
export HTTPS_PROXY=http://your-proxy:port

# Verify API endpoint
echo $LANGCHAIN_ENDPOINT
# Should be: https://api.smith.langchain.com (or blank for default)
```

### Issue 3: "Invalid API key"

**Symptom**: `AuthenticationError: Invalid API key`

**Solution**:
1. Verify key format: Must start with `lsv2_pt_`
2. Check for trailing whitespace in `.env`
3. Regenerate key in LangSmith UI → Settings → API Keys

### Issue 4: Dataset upload fails

**Symptom**: `Error creating dataset` or `Permission denied`

**Solution**:
```bash
# Check API key permissions
curl -H "Authorization: Bearer $LANGCHAIN_API_KEY" \
     https://api.smith.langchain.com/api/v1/datasets

# If 403, regenerate API key with full permissions
# If 401, verify key is correct
```

### Issue 5: YAML parsing error

**Symptom**: `yaml.scanner.ScannerError`

**Solution**:
```bash
# Validate YAML syntax
uv run python -c "import yaml; yaml.safe_load(open('tests/evaluation/golden_dataset.yaml'))"

# If error, check for:
# - Tabs (use spaces only)
# - Unclosed quotes
# - Invalid characters
```

---

## Next Steps

After completing this setup:

1. ✅ **This Document**: LangSmith setup complete
2. 🔜 **Next**: [02-GOLDEN_DATASET_TESTING.md](./02-GOLDEN_DATASET_TESTING.md) - Run golden dataset tests
3. 📅 **Later**: [03-GUARDRAILS_TESTING.md](./03-GUARDRAILS_TESTING.md) - Test guardrails system
4. 📅 **Later**: [04-MONITORING_RESULTS.md](./04-MONITORING_RESULTS.md) - Monitor and interpret results

---

## Quick Reference Commands

```bash
# Verify LangSmith connection
uv run python -c "from langsmith import Client; print(Client().list_datasets())"

# Upload golden dataset
uv run python scripts/upload_golden_dataset.py

# Check dataset in LangSmith
# Visit: https://smith.langchain.com/datasets

# Run verification
uv run python scripts/verify_langsmith_setup.py
```

---

**Next Document**: [02-GOLDEN_DATASET_TESTING.md](./02-GOLDEN_DATASET_TESTING.md)
