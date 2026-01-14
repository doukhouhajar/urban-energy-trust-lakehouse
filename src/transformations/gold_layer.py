from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, avg, sum as spark_sum, count, min as spark_min, max as spark_max,
    stddev, percentile_approx, date_trunc, to_date
)
from typing import Dict
import os


def _join_path(base: str, *parts: str) -> str:
    """
    Join paths that work with both HDFS (hdfs://) and local filesystem.
    os.path.join() doesn't work correctly for HDFS paths or absolute paths starting with /.
    """
    # Remove trailing slashes from base
    base = base.rstrip('/')
    # Join parts with single slash
    result = base
    for part in parts:
        part = part.lstrip('/')
        if part:
            result = f"{result}/{part}"
    return result


def create_consumption_analytics(
    spark: SparkSession,
    household_enriched_path: str,
    weather_enriched_path: str,
    gold_path: str
) -> DataFrame:
    print("Creating consumption analytics table...")
    
    # Load household enriched data (has consumption + household metadata)
    household_df = spark.read.format("delta").load(household_enriched_path)
    
    # Load weather enriched data (has consumption + weather data)
    weather_df = spark.read.format("delta").load(weather_enriched_path)
    
    # Join both to get consumption + household + weather in one dataset
    # Use household_enriched as base to preserve all consumption records
    # Left join weather to add weather columns (may be null for some records)
    df = household_df.join(
        weather_df.select(
            col("household_id"),
            col("timestamp"),
            col("temperature_celsius"),
            col("humidity_ratio"),
            col("apparent_temperature_celsius"),
            col("pressure_mb"),
            col("wind_speed_ms"),
            col("weather_summary")
        ),
        on=["household_id", "timestamp"],
        how="left"
    )
    
    # daily aggregations per household
    analytics_df = df.groupBy(
        col("household_id"),
        col("date"),
        col("acorn_group"),
        col("tariff_type"),
        col("year"),
        col("month"),
        col("day_of_week")
    ).agg(
        count("*").alias("halfhourly_readings_count"),
        avg(col("energy_kwh")).alias("avg_consumption_kwh"),
        spark_sum(col("energy_kwh")).alias("total_daily_consumption_kwh"),
        spark_min(col("energy_kwh")).alias("min_consumption_kwh"),
        spark_max(col("energy_kwh")).alias("max_consumption_kwh"),
        stddev(col("energy_kwh")).alias("std_consumption_kwh"),
        avg(col("temperature_celsius")).alias("avg_temperature_celsius"),
        avg(col("humidity_ratio")).alias("avg_humidity_ratio"),
        spark_min(col("temperature_celsius")).alias("min_temperature_celsius"),
        spark_max(col("temperature_celsius")).alias("max_temperature_celsius")
    ).withColumn(
        "consumption_per_reading",
        col("total_daily_consumption_kwh") / col("halfhourly_readings_count")
    )
    
    # Remove overwriteSchema=true - it's dangerous and can silently change schemas
    analytics_df.write.format("delta").mode("overwrite").save(gold_path)
    
    print(f"Created analytics table with {analytics_df.count()} daily records")
    
    return analytics_df


def run_gold_transformations(spark: SparkSession, config: Dict) -> Dict[str, DataFrame]:
    paths = config['paths']
    
    results = {}
    

    print("GOLD LAYER TRANSFORMATIONS")

    
    # Only create local directories; HDFS locations are managed by Spark/Hadoop
    if paths['gold_root'].startswith("file://"):
        os.makedirs(paths['gold_root'][len("file://"):], exist_ok=True)
    
    print("\n1. Creating consumption analytics...")
    results['consumption_analytics'] = create_consumption_analytics(
        spark,
        _join_path(paths['silver_root'], "household_enriched"),
        _join_path(paths['silver_root'], "weather_enriched"),
        _join_path(paths['gold_root'], "consumption_analytics")
    )
    
    print("\n2. Copying geospatial tables to Gold...")
    gadm_level2 = spark.read.format("delta").load(_join_path(paths['bronze_root'], "gadm_level2"))
    gadm_level2.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(_join_path(paths['gold_root'], "gadm_level2"))
    
    gadm_level3 = spark.read.format("delta").load(_join_path(paths['bronze_root'], "gadm_level3"))
    gadm_level3.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(_join_path(paths['gold_root'], "gadm_level3"))
    
    print("GOLD TRANSFORMATIONS COMPLETE")

    
    return results
