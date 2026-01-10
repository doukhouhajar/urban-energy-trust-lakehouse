"""Batch data ingestion to Bronze layer"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *
from pyspark.sql.functions import (
    col, lit, current_timestamp, to_timestamp, trim,
    when, regexp_replace, monotonically_increasing_id
)
from typing import Dict, Optional
import os
from pathlib import Path
from delta.tables import DeltaTable


def add_ingestion_metadata(df: DataFrame, source: str, batch_id: Optional[str] = None) -> DataFrame:
    """Add ingestion metadata columns to DataFrame"""
    df = df.withColumn("_ingestion_timestamp", current_timestamp()) \
           .withColumn("_source", lit(source))
    
    if batch_id:
        df = df.withColumn("_batch_id", lit(batch_id))
    else:
        df = df.withColumn("_batch_id", monotonically_increasing_id())
    
    return df


def ingest_household_info(spark: SparkSession, source_path: str, target_path: str) -> DataFrame:
    """Ingest household information CSV to Bronze"""
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(source_path)
    
    # Clean column names
    df = df.select(
        col("LCLid").alias("household_id"),
        col("stdorToU").alias("tariff_type"),
        col("Acorn").alias("acorn_code"),
        col("Acorn_grouped").alias("acorn_group"),
        col("file").alias("source_block")
    )
    
    df = add_ingestion_metadata(df, source="informations_households.csv")
    
    # Write to Delta Bronze
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(target_path)
    
    return df


def ingest_halfhourly_consumption(
    spark: SparkSession,
    source_dir: str,
    target_path: str,
    limit_blocks: Optional[int] = None
) -> DataFrame:
    """Ingest half-hourly consumption data from multiple CSV files"""
    # Get all block CSV files
    block_files = sorted(Path(source_dir).glob("block_*.csv"))
    
    if limit_blocks:
        block_files = block_files[:limit_blocks]
    
    if not block_files:
        raise ValueError(f"No block files found in {source_dir}")
    
    print(f"Ingesting {len(block_files)} block files...")
    
    # Define schema for consistency
    schema = StructType([
        StructField("LCLid", StringType(), True),
        StructField("tstp", StringType(), True),
        StructField("energy(kWh/hh)", StringType(), True)  # Read as string, parse later
    ])
    
    dfs = []
    for i, block_file in enumerate(block_files):
        print(f"Processing block {i+1}/{len(block_files)}: {block_file.name}")
        
        df = spark.read \
            .option("header", "true") \
            .schema(schema) \
            .csv(str(block_file))
        
        # Extract block number from filename
        block_num = block_file.stem.split("_")[-1]
        df = df.withColumn("source_block", lit(block_num))
        
        dfs.append(df)
    
    # Union all blocks
    if len(dfs) == 1:
        combined_df = dfs[0]
    else:
        combined_df = dfs[0]
        for df in dfs[1:]:
            combined_df = combined_df.unionByName(df, allowMissingColumns=True)
    
    # Clean and transform
    combined_df = combined_df.select(
        col("LCLid").alias("household_id"),
        # Parse timestamp - handle format: 2012-10-12 00:30:00.0000000
        to_timestamp(
            regexp_replace(col("tstp"), "\.0+$", ""),
            "yyyy-MM-dd HH:mm:ss"
        ).alias("timestamp"),
        # Parse energy - handle leading/trailing spaces and convert to double
        regexp_replace(trim(col("energy(kWh/hh)")), "^\\s*", "").cast(DoubleType()).alias("energy_kwh")
    ) \
    .withColumn("source_block", col("source_block"))
    
    # Add ingestion metadata
    combined_df = add_ingestion_metadata(combined_df, source="halfhourly_dataset")
    
    # Write to Delta Bronze with partitioning
    combined_df.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .partitionBy("source_block") \
        .save(target_path)
    
    return combined_df


def ingest_weather_hourly(
    spark: SparkSession,
    source_path: str,
    target_path: str
) -> DataFrame:
    """Ingest hourly weather data"""
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(source_path)
    
    # Clean and standardize columns
    df = df.select(
        col("time").alias("timestamp"),
        col("temperature").alias("temperature_celsius"),
        col("apparentTemperature").alias("apparent_temperature_celsius"),
        col("humidity").alias("humidity_ratio"),
        col("pressure").alias("pressure_mb"),
        col("windSpeed").alias("wind_speed_ms"),
        col("windBearing").alias("wind_bearing_degrees"),
        col("visibility").alias("visibility_km"),
        col("dewPoint").alias("dew_point_celsius"),
        col("precipType").alias("precipitation_type"),
        col("icon").alias("weather_icon"),
        col("summary").alias("weather_summary")
    )
    
    # Parse timestamp
    df = df.withColumn(
        "timestamp",
        to_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss")
    )
    
    df = add_ingestion_metadata(df, source="weather_hourly_darksky.csv")
    
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(target_path)
    
    return df


def ingest_weather_daily(
    spark: SparkSession,
    source_path: str,
    target_path: str
) -> DataFrame:
    """Ingest daily weather data"""
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(source_path)
    
    # Parse date and select relevant columns
    df = df.withColumn(
        "date",
        to_timestamp(col("time"), "yyyy-MM-dd HH:mm:ss").cast("date")
    )
    
    # Select key daily metrics
    daily_cols = [
        col("date"),
        col("temperatureHigh").alias("temp_high_celsius"),
        col("temperatureLow").alias("temp_low_celsius"),
        col("temperatureMin").alias("temp_min_celsius"),
        col("temperatureMax").alias("temp_max_celsius"),
        col("precipIntensity").alias("precipitation_intensity_mmh"),
        col("precipProbability").alias("precipitation_probability"),
        col("humidity").alias("humidity_ratio"),
        col("pressure").alias("pressure_mb"),
        col("windSpeed").alias("wind_speed_ms"),
        col("windGust").alias("wind_gust_ms"),
        col("visibility").alias("visibility_km"),
        col("summary").alias("weather_summary")
    ]
    
    # Select only columns that exist
    available_cols = [c for c in daily_cols if c._jc.toString().split("`")[1] in df.columns]
    df = df.select(*available_cols)
    
    df = add_ingestion_metadata(df, source="weather_daily_darksky.csv")
    
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(target_path)
    
    return df


def ingest_acorn_details(
    spark: SparkSession,
    source_path: str,
    target_path: str
) -> DataFrame:
    """Ingest ACORN socio-economic details"""
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(source_path)
    
    df = add_ingestion_metadata(df, source="acorn_details.csv")
    
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(target_path)
    
    return df


def ingest_bank_holidays(
    spark: SparkSession,
    source_path: str,
    target_path: str
) -> DataFrame:
    """Ingest UK bank holidays"""
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(source_path)
    
    # Standardize date column
    if "Bank holidays" in df.columns:
        df = df.withColumnRenamed("Bank holidays", "holiday_date")
    
    df = df.withColumn(
        "holiday_date",
        to_timestamp(col("holiday_date"), "yyyy-MM-dd").cast("date")
    )
    
    df = add_ingestion_metadata(df, source="uk_bank_holidays.csv")
    
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(target_path)
    
    return df


def run_bronze_ingestion(spark: SparkSession, config: Dict) -> Dict[str, DataFrame]:
    """Run complete Bronze layer ingestion"""
    paths = config['paths']
    tables = config['tables']
    
    results = {}
    
    # Create Bronze directory
    os.makedirs(paths['bronze_root'], exist_ok=True)
    
    print("=" * 60)
    print("BRONZE LAYER INGESTION")
    print("=" * 60)
    
    # Ingest household info
    print("\n1. Ingesting household information...")
    results['household_info'] = ingest_household_info(
        spark,
        paths['smart_meters']['household_info'],
        os.path.join(paths['bronze_root'], "household_info")
    )
    print(f"   ✓ Ingested {results['household_info'].count()} households")
    
    # Ingest ACORN details
    print("\n2. Ingesting ACORN details...")
    results['acorn_details'] = ingest_acorn_details(
        spark,
        paths['smart_meters']['acorn_details'],
        os.path.join(paths['bronze_root'], "acorn_details")
    )
    print(f"   ✓ Ingested ACORN details")
    
    # Ingest half-hourly consumption (with limit for initial testing)
    print("\n3. Ingesting half-hourly consumption...")
    limit_blocks = config.get('ingestion', {}).get('limit_blocks', None)
    results['halfhourly_consumption'] = ingest_halfhourly_consumption(
        spark,
        paths['smart_meters']['halfhourly_dir'],
        os.path.join(paths['bronze_root'], "halfhourly_consumption"),
        limit_blocks=limit_blocks
    )
    print(f"   ✓ Ingested half-hourly consumption data")
    
    # Ingest weather data
    print("\n4. Ingesting weather data...")
    results['weather_hourly'] = ingest_weather_hourly(
        spark,
        paths['smart_meters']['weather_hourly'],
        os.path.join(paths['bronze_root'], "weather_hourly")
    )
    print(f"   ✓ Ingested hourly weather data")
    
    results['weather_daily'] = ingest_weather_daily(
        spark,
        paths['smart_meters']['weather_daily'],
        os.path.join(paths['bronze_root'], "weather_daily")
    )
    print(f"   ✓ Ingested daily weather data")
    
    # Ingest bank holidays
    if os.path.exists(paths['smart_meters']['bank_holidays']):
        print("\n5. Ingesting bank holidays...")
        results['bank_holidays'] = ingest_bank_holidays(
            spark,
            paths['smart_meters']['bank_holidays'],
            os.path.join(paths['bronze_root'], "bank_holidays")
        )
        print(f"   ✓ Ingested bank holidays")
    
    print("\n" + "=" * 60)
    print("BRONZE INGESTION COMPLETE")
    print("=" * 60)
    
    return results
