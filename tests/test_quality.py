"""Tests for quality checks"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType

from src.quality.custom_checks import (
    check_completeness, check_temporal_coherence,
    check_business_rules, check_schema_validity
)


@pytest.fixture(scope="session")
def spark():
    """Create Spark session for testing"""
    return SparkSession.builder \
        .appName("test") \
        .master("local[2]") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()


@pytest.fixture
def sample_consumption_data(spark):
    """Create sample consumption data for testing"""
    schema = StructType([
        StructField("household_id", StringType(), True),
        StructField("timestamp", TimestampType(), True),
        StructField("energy_kwh", DoubleType(), True)
    ])
    
    data = [
        ("HH001", "2024-01-01 00:00:00", 0.5),
        ("HH001", "2024-01-01 00:30:00", 0.6),
        ("HH001", "2024-01-01 01:00:00", 0.7),
        ("HH002", "2024-01-01 00:00:00", 0.8),
        ("HH002", "2024-01-01 00:30:00", None),  # Missing value
        ("HH003", "2024-01-01 00:00:00", -0.5),  # Negative (invalid)
        ("HH003", "2024-01-01 00:30:00", 100.0),  # Above max (invalid)
    ]
    
    return spark.createDataFrame(data, schema)


def test_check_completeness(spark, sample_consumption_data):
    """Test completeness check"""
    completeness_metrics, incidents = check_completeness(
        sample_consumption_data,
        partition_cols=["household_id"],
        timestamp_col="timestamp",
        expected_interval_minutes=30,
        missing_threshold=0.10
    )
    
    assert completeness_metrics is not None
    assert incidents is not None
    assert completeness_metrics.count() > 0


def test_check_business_rules(spark, sample_consumption_data):
    """Test business rules check"""
    business_metrics, incidents = check_business_rules(
        sample_consumption_data,
        partition_cols=["household_id"],
        value_col="energy_kwh",
        timestamp_col="timestamp",
        min_value=0.0,
        max_value=50.0,
        z_score_threshold=5.0
    )
    
    assert business_metrics is not None
    assert incidents is not None
    # Should detect negative and above-max values
    incidents_list = incidents.collect()
    assert len([i for i in incidents_list if "negative" in i.rule_name]) > 0


def test_check_temporal_coherence(spark, sample_consumption_data):
    """Test temporal coherence check"""
    temporal_metrics, incidents = check_temporal_coherence(
        sample_consumption_data,
        partition_cols=["household_id"],
        timestamp_col="timestamp",
        expected_interval_minutes=30
    )
    
    assert temporal_metrics is not None
    assert incidents is not None


def test_check_schema_validity(spark):
    """Test schema validity check"""
    schema = StructType([
        StructField("household_id", StringType(), True),
        StructField("timestamp", TimestampType(), True),
        StructField("energy_kwh", DoubleType(), True),
        StructField("tariff_type", StringType(), True),
        StructField("acorn_group", StringType(), True)
    ])
    
    data = [
        ("HH001", "2024-01-01 00:00:00", 0.5, "Std", "Affluent"),
        ("HH002", "2024-01-01 00:00:00", 0.8, "Invalid", "Affluent"),  # Invalid tariff
        ("HH003", "2024-01-01 00:00:00", 150.0, "Std", "Affluent"),  # Out of range
    ]
    
    df = spark.createDataFrame(data, schema)
    
    schema_metrics, incidents = check_schema_validity(
        df,
        partition_cols=["household_id"],
        timestamp_col="timestamp",
        allowed_tariffs=["Std", "ToU"],
        allowed_acorn_groups=["Affluent", "Comfortable"],
        energy_range=(0.0, 100.0)
    )
    
    assert schema_metrics is not None
    assert incidents is not None
