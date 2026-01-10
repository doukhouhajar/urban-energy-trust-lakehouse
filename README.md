# Urban Energy Data Lakehouse with Data Quality & Governance
## London Pilot — Smart City Decision-Making

A production-grade data lakehouse architecture implementing Bronze/Silver/Gold layers with comprehensive data quality validation, governance, and ML-powered quality risk prediction for smart meter electricity consumption data.

---

## 🏗️ Architecture Overview

### Lakehouse Layers

1. **Bronze Layer** (`lakehouse/bronze/`)
   - Raw data ingestion with minimal transformation
   - Preservation of original data with ingestion metadata
   - Supports both batch and streaming ingestion

2. **Silver Layer** (`lakehouse/silver/`)
   - Cleaned and validated data
   - Standardized schemas
   - Deduplication and temporal coherence fixes
   - Missing value imputation/repair

3. **Gold Layer** (`lakehouse/gold/`)
   - Analytics-ready trusted tables
   - `quality_scores` - Dataset quality scores (0-100) per partition
   - `quality_incidents` - Detailed quality incident logs
   - `quality_risk_predictions` - ML predictions for future quality risk

### Tech Stack

- **PySpark** - Distributed data processing
- **Delta Lake** - ACID transactions, time travel, schema evolution
- **Great Expectations** - Data quality validation framework
- **Apache Sedona** - Geospatial processing (GADM, OSM)
- **LightGBM** - Quality risk prediction ML model
- **Spark Structured Streaming** - Real-time data ingestion
- **Hive Metastore** - Table metadata management

---

## 📋 Prerequisites

- Python 3.9+
- Java 8 or 11
- Docker & Docker Compose (for Spark cluster)
- 16GB+ RAM recommended

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Infrastructure (Docker Compose)

```bash
docker-compose up -d
```

This starts:
- Spark Master (localhost:8080)
- Spark Worker
- Hive Metastore (PostgreSQL backend)
- Jupyter (optional, localhost:8888)

### 3. Run Batch Pipeline

```bash
# Full end-to-end batch pipeline
python -m src.pipelines.batch_pipeline

# Or using Make
make batch-run
```

### 4. Run Streaming Pipeline

```bash
# Start streaming ingestion (simulates new data arrival)
python -m src.pipelines.streaming_pipeline

# Or using Make
make streaming-run
```

### 5. Train Quality Risk ML Model

```bash
python -m src.ml.train_quality_model

# Or using Make
make ml-train
```

---

## 📁 Project Structure

```
urban-energy-trust-lakehouse/
├── config/
│   ├── config.yaml                 # Main configuration
│   ├── great_expectations/         # GE expectations suites
│   └── schemas/                    # JSON schema definitions
├── src/
│   ├── ingestion/
│   │   ├── batch_ingestion.py      # Batch data ingestion
│   │   └── streaming_ingestion.py  # Streaming data ingestion
│   ├── quality/
│   │   ├── great_expectations_suite.py  # GE validations
│   │   ├── custom_checks.py        # Custom Spark quality checks
│   │   └── quality_scoring.py      # Quality score computation
│   ├── governance/
│   │   ├── schema_registry.py      # Schema evolution handling
│   │   ├── audit_log.py            # Pipeline audit logging
│   │   └── versioning.py           # Delta time travel utilities
│   ├── geospatial/
│   │   ├── gadm_ingestion.py       # GADM admin boundaries
│   │   ├── osm_ingestion.py        # OSM building extraction
│   │   └── spatial_operations.py   # Spatial joins & aggregations
│   ├── features/
│   │   └── quality_features.py     # Feature engineering for ML
│   ├── ml/
│   │   ├── train_quality_model.py  # Model training
│   │   ├── predict_quality_risk.py # Inference script
│   │   └── model_utils.py          # Model utilities
│   └── pipelines/
│       ├── batch_pipeline.py       # End-to-end batch pipeline
│       └── streaming_pipeline.py   # End-to-end streaming pipeline
├── notebooks/
│   ├── 01_exploration.ipynb        # Data exploration
│   ├── 02_quality_analysis.ipynb   # Quality analysis
│   └── 03_ml_evaluation.ipynb      # ML model evaluation
├── tests/
│   ├── test_ingestion.py
│   ├── test_quality.py
│   └── test_governance.py
├── data/
│   ├── raw/                        # Raw source data
│   └── lakehouse/                  # Delta Lake tables
│       ├── bronze/
│       ├── silver/
│       └── gold/
├── docker-compose.yml              # Spark cluster setup
├── requirements.txt                # Python dependencies
├── Makefile                        # Convenience commands
└── README.md

```

---

## 🔍 Data Quality Rules

### Completeness
- Missing rate per household per day/week
- Threshold: Alert if >10% missing per day, >25% per week

### Temporal Coherence
- Half-hourly cadence must be consistent (30-minute intervals)
- Detect gaps, duplicates, out-of-order records
- Alert if >5% temporal anomalies

### Business Rules
- Negative consumption invalid (hard constraint)
- Extreme spikes: z-score >5 or absolute >50 kWh/hh (contextual)
- Cross-source consistency: consumption should correlate with weather extremes

### Schema Validation
- Type checks (numeric, datetime, categorical)
- Allowed ranges for energy values (0-100 kWh/hh)
- Categorical enums for tariffs (Std/ToU), ACORN groups

### Quality Scoring Formula

```
Quality Score (0-100) = 
  40 * (1 - completeness_error_rate) +
  25 * (1 - temporal_coherence_error_rate) +
  20 * (1 - business_rule_violation_rate) +
  15 * schema_validity_score

Where:
  - completeness_error_rate = missing_records / expected_records
  - temporal_coherence_error_rate = anomalies / total_records
  - business_rule_violation_rate = violations / total_records
  - schema_validity_score = valid_records / total_records
```

---

## 📊 Example Queries

### Top Areas by Low Quality Score

```sql
SELECT 
    acorn_grouped,
    DATE_TRUNC('day', score_date) as day,
    AVG(quality_score) as avg_score,
    COUNT(*) as household_count
FROM gold.quality_scores
WHERE score_date >= CURRENT_DATE - INTERVAL 7 DAYS
GROUP BY acorn_grouped, DATE_TRUNC('day', score_date)
ORDER BY avg_score ASC
LIMIT 20;
```

### Incidents by Type Over Time

```sql
SELECT 
    DATE_TRUNC('day', incident_timestamp) as day,
    rule_name,
    severity,
    COUNT(*) as incident_count,
    COUNT(DISTINCT entity_id) as affected_households
FROM gold.quality_incidents
WHERE incident_timestamp >= CURRENT_DATE - INTERVAL 30 DAYS
GROUP BY day, rule_name, severity
ORDER BY day DESC, incident_count DESC;
```

### Quality Risk Predictions

```sql
SELECT 
    prediction_date,
    entity_id,
    risk_score,
    risk_category,
    top_features,
    model_version
FROM gold.quality_risk_predictions
WHERE prediction_date = CURRENT_DATE + INTERVAL 1 DAY
ORDER BY risk_score DESC
LIMIT 100;
```

### Building Aggregation by Admin Area

```sql
SELECT 
    g3.NAME_3 as admin_area,
    COUNT(b.id) as building_count,
    SUM(ST_Area(b.geometry)) as total_area_m2
FROM gold.osm_buildings b
JOIN gold.gadm_level3 g3 
    ON ST_Within(ST_Transform(b.geometry, 4326), g3.geom)
WHERE g3.NAME_0 = 'United Kingdom'
GROUP BY g3.NAME_3
ORDER BY building_count DESC
LIMIT 50;
```

---

## 🔄 Time Travel & Versioning

Delta Lake provides time travel capabilities. Access historical versions:

```python
from src.governance.versioning import get_table_version

# View table at a specific version
df = spark.read.format("delta").option("versionAsOf", 5).load("lakehouse/gold/consumption")

# View table at a specific timestamp
df = spark.read.format("delta").option("timestampAsOf", "2024-01-01").load("lakehouse/gold/consumption")

# Get version history
history = spark.sql("DESCRIBE HISTORY delta.`lakehouse/gold/consumption`")
history.show()
```

---

## 📈 ML Model: Quality Risk Prediction

### Problem Definition
Predict whether a future time window (e.g., next day) will have low quality (quality score < threshold OR high incident count).

### Features
- Historical missing rate (7-day, 30-day rolling)
- Incident counts by type (temporal, completeness, business)
- Volatility metrics (consumption std dev, z-score)
- Weather statistics (temperature, humidity extremes)
- Household segment (ACORN group encoding)
- Time-of-year signals (month, day_of_week, is_holiday)

### Model Outputs
- `quality_risk_predictions` Delta table
- Feature importance visualization
- Model metrics (AUC, F1, calibration curve)
- Explainability fields (SHAP values or feature contributions)

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_quality.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📝 Configuration

Edit `config/config.yaml` to customize:
- Data paths
- Table names
- Quality thresholds
- ML model parameters
- Spark configuration

---

## 🐳 Docker Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f spark-master
docker-compose logs -f spark-worker

# Stop services
docker-compose down

# Rebuild images
docker-compose build --no-cache
```

---

## 📚 Additional Resources

- **Delta Lake Documentation**: https://delta.io
- **Great Expectations**: https://greatexpectations.io
- **Apache Sedona**: https://sedona.apache.org
- **Spark Structured Streaming**: https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html

---

## 🎯 Acceptance Criteria Status

- ✅ End-to-end batch pipeline with Bronze/Silver/Gold layers
- ✅ Streaming pipeline with Structured Streaming
- ✅ Great Expectations + custom quality checks
- ✅ Quality scoring and incident logging
- ✅ Geospatial ingestion (GADM + OSM) with spatial operations
- ✅ ML model for quality risk prediction
- ✅ Schema evolution and audit logging
- ✅ Time travel/versioning support
- ✅ Comprehensive documentation and example queries

---

## 👥 Authors

Urban Energy Trust - London Pilot Team

---

## 📄 License

MIT License
