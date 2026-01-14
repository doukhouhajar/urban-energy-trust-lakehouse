from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, lit, when, isnull, isnan, trim, upper, coalesce,
    date_trunc, to_date, hour, dayofweek, dayofmonth, month, year,
    lag, lead, row_number, avg, stddev
)
from pyspark.sql.window import Window
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


def clean_halfhourly_consumption(
    spark: SparkSession,
    bronze_path: str,
    silver_path: str,
    config: Dict
) -> DataFrame:
    print("Cleaning half-hourly consumption...")
    
    df = spark.read.format("delta").load(bronze_path)
    
    # 1. remove explicit duplicates
    df = df.dropDuplicates(["household_id", "timestamp"])
    
    # 2. remove negative consumption
    df = df.filter(col("energy_kwh") >= 0.0)
    
    # 3. remove extreme outliers
    max_consumption = config.get('quality', {}).get('business_rules', {}).get('max_consumption', 50.0)
    df = df.filter(col("energy_kwh") <= max_consumption)
    
    # 4. fix temporal coherence
    window_spec = Window.partitionBy("household_id").orderBy("timestamp")
    df = df.withColumn("row_num", row_number().over(window_spec))
    df = df.filter(col("row_num") == 1).drop("row_num")  # Remove duplicates from ordering
    
    # 5. add derived temporal columns
    df = df.withColumn("date", to_date(col("timestamp"))) \
           .withColumn("hour", hour(col("timestamp"))) \
           .withColumn("day_of_week", dayofweek(col("timestamp"))) \
           .withColumn("day_of_month", dayofmonth(col("timestamp"))) \
           .withColumn("month", month(col("timestamp"))) \
           .withColumn("year", year(col("timestamp")))
    
    # 6. forward fill missing values within household
    window_24h = Window.partitionBy("household_id").orderBy("timestamp").rowsBetween(-48, 0)
    df = df.withColumn(
        "energy_kwh",
        when(col("energy_kwh").isNull() | isnan(col("energy_kwh")),
             avg(col("energy_kwh")).over(window_24h)
        ).otherwise(col("energy_kwh"))
    )
    
    # 7. remove rows with still-null values after imputation
    df = df.filter(col("energy_kwh").isNotNull())
    
    # 8. standardize column names and types
    df = df.select(
        col("household_id").cast("string"),
        col("timestamp").cast("timestamp"),
        col("energy_kwh").cast("double"),
        col("date").cast("date"),
        col("hour").cast("int"),
        col("day_of_week").cast("int"),
        col("day_of_month").cast("int"),
        col("month").cast("int"),
        col("year").cast("int"),
        col("source_block").cast("string"),
        col("_ingestion_timestamp").cast("timestamp"),
        col("_source").cast("string")
    )
    
    # 9. add quality flags
    df = df.withColumn("_silver_processed", coalesce(col("energy_kwh"), lit(0.0)) > 0)
    
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(silver_path)
    
    print(f"Cleaned {df.count()} records to Silver")
    
    return df


def enrich_household_data(
    spark: SparkSession,
    consumption_silver_path: str,
    household_bronze_path: str,
    silver_path: str
) -> DataFrame:
    print("Enriching household data...")
    
    consumption_df = spark.read.format("delta").load(consumption_silver_path)
    household_df = spark.read.format("delta").load(household_bronze_path)
    
    # join household info
    enriched_df = consumption_df.join(
        household_df.select(
            col("household_id"),
            col("tariff_type"),
            col("acorn_code"),
            col("acorn_group")
        ),
        on="household_id",
        how="left"
    )
    
    # standardize categorical values
    enriched_df = enriched_df.withColumn(
        "tariff_type",
        upper(trim(col("tariff_type")))
    ).withColumn(
        "acorn_group",
        trim(col("acorn_group"))
    )
    
    enriched_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(silver_path)
    
    print(f"Enriched {enriched_df.count()} records")
    
    return enriched_df


def enrich_weather_data(
    spark: SparkSession,
    consumption_silver_path: str,
    weather_hourly_bronze_path: str,
    silver_path: str
) -> DataFrame:
    print("Enriching weather data...")
    
    consumption_df = spark.read.format("delta").load(consumption_silver_path)
    weather_df = spark.read.format("delta").load(weather_hourly_bronze_path)
    
    # round timestamps to hour for join
    consumption_with_hour = consumption_df.withColumn(
        "hour_timestamp",
        date_trunc("hour", col("timestamp"))
    )
    
    weather_with_hour = weather_df.withColumn(
        "hour_timestamp",
        date_trunc("hour", col("timestamp"))
    )
    
    enriched_df = consumption_with_hour.join(
        weather_with_hour.select(
            col("hour_timestamp"),
            col("temperature_celsius"),
            col("apparent_temperature_celsius"),
            col("humidity_ratio"),
            col("pressure_mb"),
            col("wind_speed_ms"),
            col("weather_summary")
        ),
        on="hour_timestamp",
        how="left"
    ).drop("hour_timestamp")
    
    enriched_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(silver_path)
    
    print(f"Enriched with weather data")
    
    return enriched_df


def run_silver_transformations(spark: SparkSession, config: Dict) -> Dict[str, DataFrame]:
    paths = config['paths']
    
    results = {}
    
    print("SILVER LAYER TRANSFORMATIONS")
    
    # Only create local directories; HDFS locations are managed by Spark/Hadoop
    if paths['silver_root'].startswith("file://"):
        os.makedirs(paths['silver_root'][len("file://"):], exist_ok=True)
    
    print("\n1. Cleaning half-hourly consumption...")
    results['halfhourly_consumption'] = clean_halfhourly_consumption(
        spark,
        _join_path(paths['bronze_root'], "halfhourly_consumption"),
        _join_path(paths['silver_root'], "halfhourly_consumption"),
        config
    )
    
    print("\n2. Enriching with household metadata...")
    results['household_enriched'] = enrich_household_data(
        spark,
        _join_path(paths['silver_root'], "halfhourly_consumption"),
        _join_path(paths['bronze_root'], "household_info"),
        _join_path(paths['silver_root'], "household_enriched")
    )
    
    print("\n3. Enriching with weather data...")
    results['weather_enriched'] = enrich_weather_data(
        spark,
        _join_path(paths['silver_root'], "halfhourly_consumption"),
        _join_path(paths['bronze_root'], "weather_hourly"),
        _join_path(paths['silver_root'], "weather_enriched")
    )
    
    print("SILVER TRANSFORMATIONS COMPLETE")
    
    return results
