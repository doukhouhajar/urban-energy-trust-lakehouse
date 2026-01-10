"""Silver layer transformations: cleaning, validation, standardization"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, isnull, isnan, trim, upper, coalesce,
    date_trunc, to_date, hour, dayofweek, dayofmonth, month, year,
    lag, lead, row_number, avg, stddev
)
from pyspark.sql.window import Window
from typing import Dict
import os


def clean_halfhourly_consumption(
    spark: SparkSession,
    bronze_path: str,
    silver_path: str,
    config: Dict
) -> DataFrame:
    """
    Transform Bronze half-hourly consumption to Silver layer:
    - Remove duplicates
    - Fix temporal coherence
    - Repair missing values (forward fill within household)
    - Standardize schema
    - Add derived columns
    """
    print("Cleaning half-hourly consumption...")
    
    # Read Bronze data
    df = spark.read.format("delta").load(bronze_path)
    
    # 1. Remove explicit duplicates (same household_id + timestamp)
    df = df.dropDuplicates(["household_id", "timestamp"])
    
    # 2. Remove negative consumption (hard constraint)
    df = df.filter(col("energy_kwh") >= 0.0)
    
    # 3. Remove extreme outliers (above max threshold)
    max_consumption = config.get('quality', {}).get('business_rules', {}).get('max_consumption', 50.0)
    df = df.filter(col("energy_kwh") <= max_consumption)
    
    # 4. Fix temporal coherence - order by household and timestamp
    window_spec = Window.partitionBy("household_id").orderBy("timestamp")
    df = df.withColumn("row_num", row_number().over(window_spec))
    df = df.filter(col("row_num") == 1).drop("row_num")  # Remove duplicates from ordering
    
    # 5. Add derived temporal columns
    df = df.withColumn("date", to_date(col("timestamp"))) \
           .withColumn("hour", hour(col("timestamp"))) \
           .withColumn("day_of_week", dayofweek(col("timestamp"))) \
           .withColumn("day_of_month", dayofmonth(col("timestamp"))) \
           .withColumn("month", month(col("timestamp"))) \
           .withColumn("year", year(col("timestamp")))
    
    # 6. Forward fill missing values within household (within 24-hour window)
    # This is a simplified approach - in production, use more sophisticated imputation
    window_24h = Window.partitionBy("household_id").orderBy("timestamp").rowsBetween(-48, 0)
    df = df.withColumn(
        "energy_kwh",
        when(col("energy_kwh").isNull() | isnan(col("energy_kwh")),
             avg(col("energy_kwh")).over(window_24h)
        ).otherwise(col("energy_kwh"))
    )
    
    # 7. Remove rows with still-null values after imputation
    df = df.filter(col("energy_kwh").isNotNull())
    
    # 8. Standardize column names and types
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
    
    # 9. Add quality flags (will be computed more thoroughly in quality checks)
    df = df.withColumn("_silver_processed", coalesce(col("energy_kwh"), lit(0.0)) > 0)
    
    # Write to Silver
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(silver_path)
    
    print(f"✓ Cleaned {df.count()} records to Silver")
    
    return df


def enrich_household_data(
    spark: SparkSession,
    consumption_silver_path: str,
    household_bronze_path: str,
    silver_path: str
) -> DataFrame:
    """Enrich consumption data with household metadata"""
    print("Enriching household data...")
    
    consumption_df = spark.read.format("delta").load(consumption_silver_path)
    household_df = spark.read.format("delta").load(household_bronze_path)
    
    # Join household info
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
    
    # Standardize categorical values
    enriched_df = enriched_df.withColumn(
        "tariff_type",
        upper(trim(col("tariff_type")))
    ).withColumn(
        "acorn_group",
        trim(col("acorn_group"))
    )
    
    # Write enriched data
    enriched_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(silver_path)
    
    print(f"✓ Enriched {enriched_df.count()} records")
    
    return enriched_df


def enrich_weather_data(
    spark: SparkSession,
    consumption_silver_path: str,
    weather_hourly_bronze_path: str,
    silver_path: str
) -> DataFrame:
    """Enrich consumption data with weather information"""
    print("Enriching weather data...")
    
    consumption_df = spark.read.format("delta").load(consumption_silver_path)
    weather_df = spark.read.format("delta").load(weather_hourly_bronze_path)
    
    # Round timestamps to hour for join
    consumption_with_hour = consumption_df.withColumn(
        "hour_timestamp",
        date_trunc("hour", col("timestamp"))
    )
    
    weather_with_hour = weather_df.withColumn(
        "hour_timestamp",
        date_trunc("hour", col("timestamp"))
    )
    
    # Join on hour timestamp
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
    
    # Write enriched data
    enriched_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(silver_path)
    
    print(f"✓ Enriched with weather data")
    
    return enriched_df


def run_silver_transformations(spark: SparkSession, config: Dict) -> Dict[str, DataFrame]:
    """Run all Silver layer transformations"""
    paths = config['paths']
    
    results = {}
    
    print("=" * 60)
    print("SILVER LAYER TRANSFORMATIONS")
    print("=" * 60)
    
    # Create Silver directory
    os.makedirs(paths['silver_root'], exist_ok=True)
    
    # Clean half-hourly consumption
    print("\n1. Cleaning half-hourly consumption...")
    results['halfhourly_consumption'] = clean_halfhourly_consumption(
        spark,
        os.path.join(paths['bronze_root'], "halfhourly_consumption"),
        os.path.join(paths['silver_root'], "halfhourly_consumption"),
        config
    )
    
    # Enrich with household data
    print("\n2. Enriching with household metadata...")
    results['household_enriched'] = enrich_household_data(
        spark,
        os.path.join(paths['silver_root'], "halfhourly_consumption"),
        os.path.join(paths['bronze_root'], "household_info"),
        os.path.join(paths['silver_root'], "household_enriched")
    )
    
    # Enrich with weather data
    print("\n3. Enriching with weather data...")
    results['weather_enriched'] = enrich_weather_data(
        spark,
        os.path.join(paths['silver_root'], "halfhourly_consumption"),
        os.path.join(paths['bronze_root'], "weather_hourly"),
        os.path.join(paths['silver_root'], "weather_enriched")
    )
    
    print("\n" + "=" * 60)
    print("SILVER TRANSFORMATIONS COMPLETE")
    print("=" * 60)
    
    return results
