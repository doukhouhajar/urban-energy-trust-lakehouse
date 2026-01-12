from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, avg, sum as spark_sum, count, min as spark_min, max as spark_max,
    stddev, percentile_approx, date_trunc, to_date
)
from typing import Dict
import os


def create_consumption_analytics(
    spark: SparkSession,
    silver_path: str,
    gold_path: str
) -> DataFrame:
    print("Creating consumption analytics table...")
    
    df = spark.read.format("delta").load(silver_path)
    
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
    
    analytics_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(gold_path)
    
    print(f"Created analytics table with {analytics_df.count()} daily records")
    
    return analytics_df


def run_gold_transformations(spark: SparkSession, config: Dict) -> Dict[str, DataFrame]:
    paths = config['paths']
    
    results = {}
    

    print("GOLD LAYER TRANSFORMATIONS")

    
    os.makedirs(paths['gold_root'], exist_ok=True)
    
    print("\n1. Creating consumption analytics...")
    results['consumption_analytics'] = create_consumption_analytics(
        spark,
        os.path.join(paths['silver_root'], "household_enriched"),
        os.path.join(paths['gold_root'], "consumption_analytics")
    )
    
    print("\n2. Copying geospatial tables to Gold...")
    gadm_level2 = spark.read.format("delta").load(os.path.join(paths['bronze_root'], "gadm_level2"))
    gadm_level2.write.format("delta").mode("overwrite").save(os.path.join(paths['gold_root'], "gadm_level2"))
    
    gadm_level3 = spark.read.format("delta").load(os.path.join(paths['bronze_root'], "gadm_level3"))
    gadm_level3.write.format("delta").mode("overwrite").save(os.path.join(paths['gold_root'], "gadm_level3"))
    
    print("GOLD TRANSFORMATIONS COMPLETE")

    
    return results
