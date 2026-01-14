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


def add_ingestion_metadata(df: DataFrame, source: str, batch_id: Optional[str] = None) -> DataFrame:
    df = df.withColumn("_ingestion_timestamp", current_timestamp()) \
           .withColumn("_source", lit(source))
    
    if batch_id:
        df = df.withColumn("_batch_id", lit(batch_id))
    else:
        df = df.withColumn("_batch_id", monotonically_increasing_id())
    
    return df


def ingest_household_info(spark: SparkSession, source_path: str, target_path: str) -> DataFrame:
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(source_path)
    
    df = df.select(
        col("LCLid").alias("household_id"),
        col("stdorToU").alias("tariff_type"),
        col("Acorn").alias("acorn_code"),
        col("Acorn_grouped").alias("acorn_group"),
        col("file").alias("source_block")
    )
    
    df = add_ingestion_metadata(df, source="informations_households.csv")
    
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(target_path)
    
    return df


def ingest_halfhourly_consumption(
    spark: SparkSession,
    source_dir: str,
    target_path: str,
    limit_blocks: Optional[int] = None
) -> DataFrame:
    # Use Spark's filesystem API to list files (works with both local and HDFS)
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    fs = spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.get(
        spark.sparkContext._jvm.java.net.URI(source_dir), hadoop_conf
    )
    
    # List all block_*.csv files
    path = spark.sparkContext._jvm.org.apache.hadoop.fs.Path(source_dir)
    statuses = fs.listStatus(path)
    block_files = []
    for status in statuses:
        file_name = status.getPath().getName()
        if file_name.startswith("block_") and file_name.endswith(".csv"):
            full_path = str(status.getPath())
            block_files.append((full_path, file_name))
    
    block_files = sorted(block_files, key=lambda x: x[1])
    
    if limit_blocks:
        block_files = block_files[:limit_blocks]
    
    if not block_files:
        raise ValueError(f"No block files found in {source_dir}")
    
    print(f"Ingesting {len(block_files)} block files...")
    
    schema = StructType([
        StructField("LCLid", StringType(), True),
        StructField("tstp", StringType(), True),
        StructField("energy(kWh/hh)", StringType(), True)
    ])
    
    dfs = []
    for i, (block_file_path, block_file_name) in enumerate(block_files):
        print(f"Processing block {i+1}/{len(block_files)}: {block_file_name}")
        
        df = spark.read \
            .option("header", "true") \
            .schema(schema) \
            .csv(block_file_path)
        
        block_num = block_file_name.replace("block_", "").replace(".csv", "")
        df = df.withColumn("source_block", lit(block_num))
        
        dfs.append(df)
    
    if len(dfs) == 1:
        combined_df = dfs[0]
    else:
        combined_df = dfs[0]
        for df in dfs[1:]:
            combined_df = combined_df.unionByName(df, allowMissingColumns=True)
    
    combined_df = combined_df.select(
        col("LCLid").alias("household_id"),
        to_timestamp(
            regexp_replace(col("tstp"), "\.0+$", ""),
            "yyyy-MM-dd HH:mm:ss"
        ).alias("timestamp"),
        #   handle leading/trailing spaces and convert to double
        regexp_replace(trim(col("energy(kWh/hh)")), "^\\s*", "").cast(DoubleType()).alias("energy_kwh"),
        col("source_block")
    )
    
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
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(source_path)
    
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
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(source_path)
    
    df = df.withColumn(
        "date",
        to_timestamp(col("time"), "yyyy-MM-dd HH:mm:ss").cast("date")
    )
    
    available_cols = []
    if "date" in df.columns:
        available_cols.append(col("date"))
    if "temperatureHigh" in df.columns:
        available_cols.append(col("temperatureHigh").alias("temp_high_celsius"))
    if "temperatureLow" in df.columns:
        available_cols.append(col("temperatureLow").alias("temp_low_celsius"))
    if "temperatureMin" in df.columns:
        available_cols.append(col("temperatureMin").alias("temp_min_celsius"))
    if "temperatureMax" in df.columns:
        available_cols.append(col("temperatureMax").alias("temp_max_celsius"))
    if "precipIntensity" in df.columns:
        available_cols.append(col("precipIntensity").alias("precipitation_intensity_mmh"))
    if "precipProbability" in df.columns:
        available_cols.append(col("precipProbability").alias("precipitation_probability"))
    if "humidity" in df.columns:
        available_cols.append(col("humidity").alias("humidity_ratio"))
    if "pressure" in df.columns:
        available_cols.append(col("pressure").alias("pressure_mb"))
    if "windSpeed" in df.columns:
        available_cols.append(col("windSpeed").alias("wind_speed_ms"))
    if "windGust" in df.columns:
        available_cols.append(col("windGust").alias("wind_gust_ms"))
    if "visibility" in df.columns:
        available_cols.append(col("visibility").alias("visibility_km"))
    if "summary" in df.columns:
        available_cols.append(col("summary").alias("weather_summary"))
    
    if available_cols:
        df = df.select(*available_cols)
    
    df = add_ingestion_metadata(df, source="weather_daily_darksky.csv")
    
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(target_path)
    
    return df


def ingest_acorn_details(
    spark: SparkSession,
    source_path: str,
    target_path: str
) -> DataFrame:
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(source_path)
    
    # Clean column names - replace invalid characters with underscores
    for col_name in df.columns:
        clean_name = col_name.replace(" ", "_").replace(",", "_").replace(";", "_") \
                            .replace("{", "_").replace("}", "_").replace("(", "_") \
                            .replace(")", "_").replace("\n", "_").replace("\t", "_") \
                            .replace("=", "_").replace("/", "_").replace("\\", "_")
        if clean_name != col_name:
            df = df.withColumnRenamed(col_name, clean_name)
    
    df = add_ingestion_metadata(df, source="acorn_details.csv")
    
    # Enable column mapping for Delta table to handle any remaining special characters
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
        .option("delta.columnMapping.mode", "name") \
        .option("delta.minReaderVersion", "2") \
        .option("delta.minWriterVersion", "5") \
        .save(target_path)
    
    return df


def ingest_bank_holidays(
    spark: SparkSession,
    source_path: str,
    target_path: str
) -> DataFrame:
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(source_path)
    
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
    paths = config['paths']
    tables = config['tables']
    
    results = {}
    bronze_root = paths['bronze_root']
    # Only create local directories; HDFS locations are managed by Spark/Hadoop
    if bronze_root.startswith("file://"):
        local_bronze_root = bronze_root[len("file://"):]
        os.makedirs(local_bronze_root, exist_ok=True)
    
    print("BRONZE LAYER INGESTION")
    
    print("\n1. Ingesting household information...")
    results['household_info'] = ingest_household_info(
        spark,
        paths['smart_meters']['household_info'],
        _join_path(paths['bronze_root'], "household_info")
    )
    print(f"   Ingested {results['household_info'].count()} households")
    
    print("\n2. Ingesting ACORN details...")
    results['acorn_details'] = ingest_acorn_details(
        spark,
        paths['smart_meters']['acorn_details'],
        _join_path(paths['bronze_root'], "acorn_details")
    )
    print(f"   Ingested ACORN details")
    
    print("\n3. Ingesting half-hourly consumption...")
    limit_blocks = config.get('ingestion', {}).get('limit_blocks', None)
    results['halfhourly_consumption'] = ingest_halfhourly_consumption(
        spark,
        paths['smart_meters']['halfhourly_dir'],
        _join_path(paths['bronze_root'], "halfhourly_consumption"),
        limit_blocks=limit_blocks
    )
    print(f"   Ingested half-hourly consumption data")
    
    print("\n4. Ingesting weather data...")
    results['weather_hourly'] = ingest_weather_hourly(
        spark,
        paths['smart_meters']['weather_hourly'],
        _join_path(paths['bronze_root'], "weather_hourly")
    )
    print(f"   Ingested hourly weather data")
    
    results['weather_daily'] = ingest_weather_daily(
        spark,
        paths['smart_meters']['weather_daily'],
        _join_path(paths['bronze_root'], "weather_daily")
    )
    print(f"   Ingested daily weather data")
    
    # Check if bank holidays file exists (works with both local and HDFS)
    bank_holidays_path = paths['smart_meters']['bank_holidays']
    try:
        hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
        fs = spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark.sparkContext._jvm.java.net.URI(bank_holidays_path), hadoop_conf
        )
        hdfs_path = spark.sparkContext._jvm.org.apache.hadoop.fs.Path(bank_holidays_path)
        if fs.exists(hdfs_path):
            print("\n5. Ingesting bank holidays...")
            results['bank_holidays'] = ingest_bank_holidays(
                spark,
                bank_holidays_path,
                _join_path(paths['bronze_root'], "bank_holidays")
            )
            print(f"   Ingested bank holidays")
    except Exception as e:
        print(f"Warning: Could not check bank holidays file: {e}")
    
    print("BRONZE INGESTION COMPLETE")
    
    return results
