# Quick Start Guide

## Prerequisites

1. **Python 3.9+** installed
2. **Java 8 or 11** installed (for Spark)
3. **Docker & Docker Compose** (optional, for Spark cluster)

## Setup

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt
```

### 2. Start Infrastructure (Optional)

For local development, Spark runs in local mode. For distributed execution:

```bash
# Start Docker Compose services (Spark cluster + Hive metastore)
docker-compose up -d

# Check services are running
docker-compose ps

# View logs
docker-compose logs -f spark-master
```

## Running the Pipeline

### Batch Pipeline (Full End-to-End)

```bash
# Using Python module
python -m src.pipelines.batch_pipeline

# Or using Makefile
make batch-run
```

This will:
1. Ingest raw data to Bronze layer
2. Transform to Silver layer (cleaning, validation)
3. Run quality checks and compute scores
4. Create Gold layer analytics tables
5. Write audit logs

### Streaming Pipeline

```bash
# Start streaming ingestion
python -m src.pipelines.streaming_pipeline

# Or using Makefile
make streaming-run
```

Drop new CSV files into `data/streaming_source/` to process them in real-time.

### Train ML Model

```bash
# Train quality risk prediction model
python -m src.ml.train_quality_model

# Or using Makefile
make ml-train
```

### Run Quality Risk Predictions

```bash
# Generate predictions for future time windows
python -m src.ml.predict_quality_risk
```

### Example Queries

```bash
# Run example queries from README
python examples/example_queries.py
```

## Verifying Results

### Check Lakehouse Tables

```python
from src.utils.spark_session import get_or_create_spark_session
from src.utils.config import load_config
from pyspark.sql.functions import col

config = load_config()
spark = get_or_create_spark_session(config, use_docker=False)

# Read Gold layer tables
quality_scores = spark.read.format("delta").load("data/lakehouse/gold/quality_scores")
quality_scores.show(10)

# Read quality incidents
incidents = spark.read.format("delta").load("data/lakehouse/gold/quality_incidents")
incidents.groupBy("rule_name").count().show()

spark.stop()
```

### View Delta Lake History

```python
# Get version history
history = spark.sql("DESCRIBE HISTORY delta.`data/lakehouse/gold/quality_scores`")
history.show()

# Read at specific version
df_v5 = spark.read.format("delta").option("versionAsOf", 5).load("data/lakehouse/gold/quality_scores")
df_v5.show()
```

## Configuration

Edit `config/config.yaml` to customize:
- Data paths
- Quality thresholds
- ML model parameters
- Spark settings

## Troubleshooting

### Issue: Spark can't find Delta Lake

**Solution:** Ensure Delta Lake packages are in Spark classpath. Check `config/config.yaml` has correct `spark.jars.packages`.

### Issue: OSM/GADM ingestion fails

**Solution:** 
- For GADM: Requires GeoPandas (`pip install geopandas`)
- For OSM: Requires pyosmium (`pip install pyosmium`) or use osmosis tool

### Issue: Out of memory errors

**Solution:** 
- Reduce data size in config: `ingestion.limit_blocks: 10`
- Increase Spark driver/executor memory in config
- Process data in smaller batches

### Issue: Great Expectations not working

**Solution:** GE is optional. Custom Spark checks are the primary quality validation. To use GE, install and configure separately.

## Next Steps

1. Review the README.md for detailed documentation
2. Explore notebooks in `notebooks/` directory
3. Run tests: `pytest tests/`
4. Customize quality rules in `config/config.yaml`
5. Add your own transformations in `src/transformations/`

## Getting Help

- Check README.md for detailed documentation
- Review example queries in `examples/example_queries.py`
- Examine test files in `tests/` for usage examples
