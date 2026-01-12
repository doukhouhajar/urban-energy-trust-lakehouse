import pytest
from pyspark.sql import SparkSession
import tempfile
import os

from src.ingestion.batch_ingestion import add_ingestion_metadata


@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder \
        .appName("test") \
        .master("local[2]") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()


def test_add_ingestion_metadata(spark):
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType
    
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("value", StringType(), True)
    ])
    
    data = [(1, "test1"), (2, "test2")]
    df = spark.createDataFrame(data, schema)
    
    df_with_metadata = add_ingestion_metadata(df, source="test_source", batch_id="batch_001")
    
    assert "_ingestion_timestamp" in df_with_metadata.columns
    assert "_source" in df_with_metadata.columns
    assert "_batch_id" in df_with_metadata.columns
    
   # check metadata values
    rows = df_with_metadata.collect()
    assert all(row._source == "test_source" for row in rows)
