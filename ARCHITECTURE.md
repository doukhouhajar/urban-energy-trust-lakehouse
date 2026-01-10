# Architecture Documentation

## Overview

The Urban Energy Trust Lakehouse implements a production-grade data architecture following the Bronze/Silver/Gold medallion pattern with comprehensive data quality validation, governance, and ML-powered quality risk prediction.

## Architecture Layers

### Bronze Layer
**Purpose:** Raw data ingestion with minimal transformation

**Tables:**
- `household_info` - Household metadata (ACORN, tariff types)
- `halfhourly_consumption` - Raw half-hourly consumption data
- `daily_consumption` - Raw daily consumption aggregations
- `weather_hourly` - Hourly weather data (DarkSky)
- `weather_daily` - Daily weather summaries
- `bank_holidays` - UK bank holidays
- `acorn_details` - ACORN socio-economic details
- `gadm_level2` - Administrative boundaries (Level 2 - counties)
- `gadm_level3` - Administrative boundaries (Level 3 - districts)
- `osm_buildings` - OSM building footprints

**Characteristics:**
- Preserves original data structure
- Adds ingestion metadata columns (`_ingestion_timestamp`, `_source`, `_batch_id`)
- No data cleaning or validation
- Supports both batch and streaming ingestion

### Silver Layer
**Purpose:** Cleaned, validated, and standardized data

**Tables:**
- `halfhourly_consumption` - Cleaned consumption data
- `household_enriched` - Consumption + household metadata
- `weather_enriched` - Consumption + weather data

**Transformations:**
- Deduplication
- Temporal coherence fixes
- Missing value imputation (forward fill within household)
- Negative value removal (hard constraint)
- Outlier removal (configurable thresholds)
- Schema standardization
- Derived columns (date, hour, day_of_week, etc.)

### Gold Layer
**Purpose:** Analytics-ready trusted tables with quality metrics

**Tables:**
- `consumption_analytics` - Daily aggregations per household
- `quality_scores` - Quality scores (0-100) per household per day
- `quality_incidents` - Detailed quality incident logs
- `quality_risk_predictions` - ML predictions for future quality risk
- `gadm_level2` / `gadm_level3` - Cleaned administrative boundaries
- `osm_buildings` - Processed building data
- `building_aggregations` - Buildings aggregated by admin area
- `audit_log` - Pipeline execution audit trail

**Characteristics:**
- Trusted, production-ready data
- Quality scores computed using weighted formula
- Incident tracking with severity levels
- ML predictions for proactive quality management

## Data Quality Framework

### Quality Rules

1. **Completeness**
   - Missing rate per household per day/week
   - Thresholds: 10% per day, 25% per week
   - Metrics: `completeness_rate`, `missing_rate`

2. **Temporal Coherence**
   - Half-hourly cadence consistency (30-minute intervals)
   - Gap detection
   - Duplicate detection
   - Out-of-order record detection
   - Threshold: 5% anomalies triggers alert

3. **Business Rules**
   - Negative consumption invalid (hard constraint)
   - Extreme spikes: z-score > 5 or absolute > 50 kWh/hh
   - Cross-source consistency checks (consumption vs weather)

4. **Schema Validation**
   - Type checks (numeric, datetime, categorical)
   - Allowed ranges (0-100 kWh/hh)
   - Categorical enums (tariffs: Std/ToU, ACORN groups)

### Quality Scoring Formula

```
Quality Score (0-100) = 
  40 * (1 - completeness_error_rate) +
  25 * (1 - temporal_coherence_error_rate) +
  20 * (1 - business_rule_violation_rate) +
  15 * schema_validity_score

Categories:
  - Excellent: ≥90
  - Good: 80-89
  - Fair: 70-79
  - Poor: <70
```

### Incident Severity Levels

- **Critical**: Hard constraint violations (negative consumption, schema errors)
- **Warning**: Threshold violations (high missing rate, temporal anomalies)

## Governance Features

### Schema Evolution
- **Mode**: Merge (add new columns), Fail (raise error), Overwrite (replace schema)
- **Implementation**: `handle_schema_evolution()` in `governance/schema_registry.py`
- Delta Lake automatically handles schema merging when `mergeSchema=true`

### Versioning & Time Travel
- **History**: Track all table versions with metadata
- **Time Travel**: Read data at specific version or timestamp
- **Restore**: Restore table to previous version
- **Implementation**: Delta Lake native features + utilities in `governance/versioning.py`

### Audit Logging
- **Scope**: Every pipeline run logged with:
  - Input/output table versions
  - Row counts
  - Rejected rows
  - Quality score summaries
  - Status (SUCCESS/FAILED/PARTIAL)
  - Error messages (if failed)
- **Storage**: Delta table `gold.audit_log`
- **Retention**: Configurable (default: 365 days)

## ML Model: Quality Risk Prediction

### Problem Definition
**Predict:** Will a future time window (e.g., next day) have low quality?

**Target Label:**
- `is_high_risk = 1` if:
  - Next day quality score < threshold (default: 70), OR
  - Next day incident count ≥ threshold (default: 5)
- `is_high_risk = 0` otherwise

### Features

**Historical Metrics:**
- Missing rate (7-day, 30-day rolling windows)
- Incident counts by type (7-day, 30-day)
- Quality score trends (7-day average)
- Consumption volatility (std dev, coefficient of variation)

**Contextual Features:**
- Weather statistics (temperature, humidity extremes)
- Household segment (ACORN group encoding)
- Time-of-year signals (month, day_of_week, is_holiday)
- Tariff type (Std vs ToU)

### Model Architecture
- **Algorithm**: LightGBM (Gradient Boosting)
- **Type**: Binary Classification
- **Metrics**: AUC, F1 Score, Feature Importance
- **Output**: Risk score (0-1), risk category (low/medium/high), top features

### Model Storage
- **Format**: Pickle (`.pkl`)
- **Location**: `models/quality_risk_model/`
- **Versioning**: Timestamp-based (`model_YYYYMMDD_HHMMSS.pkl`)
- **Metadata**: JSON files with metrics, feature info, training parameters

## Streaming Architecture

### Structured Streaming Pipeline

**Source:**
- File-based streaming (new CSV files dropped in directory)
- Kafka (optional, commented in code)

**Process:**
1. Read new files with `readStream`
2. Transform (parse timestamps, clean data)
3. Add ingestion metadata
4. Write to Bronze Delta table with `writeStream`

**Checkpoints:**
- Spark Structured Streaming checkpoints for fault tolerance
- Location: `data/streaming_checkpoint/`

**Features:**
- Exactly-once semantics (Delta Lake + checkpoints)
- Automatic recovery on restart
- Configurable trigger interval (default: 30 seconds)
- Max files per trigger (default: 10)

## Geospatial Processing

### GADM Ingestion
- **Source**: GeoPackage (`.gpkg`) format
- **Layers**: Level 2 (counties), Level 3 (districts)
- **Processing**: GeoPandas → Pandas → Spark DataFrame
- **Geometry**: Stored as WKB (Well-Known Binary) in Delta tables

### OSM Ingestion
- **Source**: OSM PBF (Protocol Buffer Format)
- **Extraction**: Building footprints with tags (building, landuse, amenity)
- **Processing**: pyosmium (Python-based) or osmosis (command-line tool)
- **Geometry**: WKB format in Delta tables

### Spatial Operations
- **Join Type**: Spatial overlay (ST_Within, ST_Intersects)
- **Example**: Buildings aggregated by admin area (Level 3)
- **Output**: Building count, total area per admin area
- **Note**: Full implementation requires Apache Sedona for Spark-based spatial joins

## Technology Stack

### Core Technologies
- **PySpark 3.5.0**: Distributed data processing
- **Delta Lake 3.0.0**: ACID transactions, time travel, schema evolution
- **Great Expectations 0.18.8**: Data quality validation framework
- **LightGBM 4.1.0**: ML model for quality risk prediction
- **Apache Sedona 1.5.1**: Geospatial processing (optional)

### Infrastructure
- **Spark Cluster**: Docker Compose setup (Master + Worker)
- **Hive Metastore**: PostgreSQL-backed metastore for table metadata
- **Jupyter**: Optional notebook environment

### Data Formats
- **Input**: CSV, GeoPackage (GPKG), OSM PBF
- **Storage**: Delta Lake tables (Parquet + transaction log)
- **Output**: Delta tables, ML model artifacts (Pickle)

## Pipeline Execution

### Batch Pipeline Flow

```
1. Bronze Ingestion
   ├── Household Info
   ├── Consumption Data (half-hourly, daily)
   ├── Weather Data
   ├── ACORN Details
   ├── Bank Holidays
   └── Geospatial (GADM, OSM)

2. Silver Transformations
   ├── Data Cleaning
   ├── Deduplication
   ├── Missing Value Imputation
   ├── Schema Standardization
   └── Enrichment (household + weather)

3. Quality Checks
   ├── Completeness
   ├── Temporal Coherence
   ├── Business Rules
   └── Schema Validation

4. Quality Scoring
   ├── Compute Scores (0-100)
   ├── Generate Incidents
   └── Write to Gold

5. Gold Transformations
   ├── Analytics Aggregations
   ├── Building Aggregations
   └── Audit Logging

6. ML (Optional)
   ├── Feature Engineering
   ├── Model Training
   └── Quality Risk Predictions
```

### Streaming Pipeline Flow

```
1. Source Directory Watch
   └── New CSV files dropped

2. Stream Processing
   ├── Read new files (readStream)
   ├── Transform
   ├── Add metadata
   └── Write to Bronze (writeStream)

3. Continuous Quality (Optional)
   └── Stream quality checks (future enhancement)
```

## Performance Considerations

### Optimizations
- **Partitioning**: Bronze tables partitioned by `source_block` or date
- **Z-Ordering**: Gold tables optimized for common query patterns
- **Caching**: Frequently accessed tables cached in memory
- **Delta Optimize**: Periodic compaction and Z-ordering

### Scalability
- **Horizontal Scaling**: Spark worker nodes (Docker Compose)
- **Data Skew**: Configurable partitioning strategy
- **Memory Management**: Configurable Spark driver/executor memory

## Monitoring & Observability

### Audit Log
- Every pipeline run logged with metrics
- Query: `SELECT * FROM gold.audit_log ORDER BY run_timestamp DESC`

### Quality Metrics
- Daily quality scores per household
- Incident counts by type, severity
- Trends over time

### ML Model Metrics
- Model performance (AUC, F1)
- Feature importance
- Prediction distributions

## Security Considerations

### Data Access
- Delta Lake ACLs (future enhancement)
- Audit trail for all data access

### Credentials
- Environment variables for sensitive config
- `.env` file support (excluded from git)

## Future Enhancements

1. **Real-time Quality Monitoring**: Stream quality checks in real-time
2. **Automated Remediation**: Auto-fix common quality issues
3. **Data Lineage**: Track data flow across transformations
4. **Catalog Integration**: Integrate with data catalogs (Hive Metastore, DataHub)
5. **API Layer**: REST API for querying lakehouse
6. **Dashboards**: Grafana/Kibana dashboards for quality metrics
7. **Airflow Integration**: Orchestrate pipelines with Airflow DAGs
