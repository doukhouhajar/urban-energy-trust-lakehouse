"""Tests for governance modules"""

import pytest
from pyspark.sql import SparkSession
import tempfile
import os
from delta.tables import DeltaTable

from src.governance.versioning import (
    get_table_version, read_table_at_version,
    get_table_history
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


def test_versioning(spark):
    """Test Delta Lake versioning functions"""
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType
    from pyspark.sql.functions import lit
    
    with tempfile.TemporaryDirectory() as tmpdir:
        table_path = os.path.join(tmpdir, "test_table")
        
        # Create initial table
        schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("value", StringType(), True)
        ])
        
        data = [(1, "v1"), (2, "v1")]
        df_v1 = spark.createDataFrame(data, schema)
        df_v1.write.format("delta").mode("overwrite").save(table_path)
        
        # Check version
        version_1 = get_table_version(spark, table_path)
        assert version_1 >= 0
        
        # Create version 2
        data_v2 = [(1, "v2"), (2, "v2"), (3, "v2")]
        df_v2 = spark.createDataFrame(data_v2, schema)
        df_v2.write.format("delta").mode("overwrite").save(table_path)
        
        version_2 = get_table_version(spark, table_path)
        assert version_2 > version_1
        
        # Read at specific version
        df_at_v1 = read_table_at_version(spark, table_path, version_1)
        assert df_at_v1.count() == 2
        
        # Get history
        history = get_table_history(spark, table_path, limit=10)
        assert history.count() > 0
