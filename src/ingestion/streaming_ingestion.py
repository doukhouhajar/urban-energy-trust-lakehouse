from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.streaming import StreamingQuery
from pyspark.sql.types import *
from pyspark.sql.functions import (
    col, lit, current_timestamp, to_timestamp, trim,
    regexp_replace, monotonically_increasing_id
)
from typing import Dict, Optional
import os


def add_ingestion_metadata_streaming(df: DataFrame, source: str) -> DataFrame:
    return df.withColumn("_ingestion_timestamp", current_timestamp()) \
             .withColumn("_source", lit(source)) \
             .withColumn("_streaming_batch_id", monotonically_increasing_id())


def stream_halfhourly_consumption(
    spark: SparkSession,
    source_dir: str,
    target_path: str,
    checkpoint_location: str,
    trigger_interval: str = "30 seconds"
) -> StreamingQuery:
    schema = StructType([
        StructField("LCLid", StringType(), True),
        StructField("tstp", StringType(), True),
        StructField("energy(kWh/hh)", StringType(), True)
    ])
    
    stream_df = spark.readStream \
        .option("header", "true") \
        .option("maxFilesPerTrigger", 10) \
        .schema(schema) \
        .csv(source_dir)
    
    transformed_df = stream_df.select(
        col("LCLid").alias("household_id"),
        to_timestamp(
            regexp_replace(col("tstp"), "\.0+$", ""),
            "yyyy-MM-dd HH:mm:ss"
        ).alias("timestamp"),
        regexp_replace(trim(col("energy(kWh/hh)")), "^\\s*", "").cast(DoubleType()).alias("energy_kwh")
    )
    
    transformed_df = add_ingestion_metadata_streaming(transformed_df, source="halfhourly_streaming")
    
    query = transformed_df.writeStream \
        .format("delta") \
        .outputMode("append") \
        .option("checkpointLocation", checkpoint_location) \
        .option("path", target_path) \
        .trigger(processingTime=trigger_interval) \
        .start()
    
    return query


def stream_weather_hourly(
    spark: SparkSession,
    source_dir: str,
    target_path: str,
    checkpoint_location: str,
    trigger_interval: str = "30 seconds"
) -> StreamingQuery:
    stream_df = spark.readStream \
        .option("header", "true") \
        .option("maxFilesPerTrigger", 10) \
        .schema(
            StructType([
                StructField("time", TimestampType(), True),
                StructField("temperature", DoubleType(), True),
                StructField("apparentTemperature", DoubleType(), True),
                StructField("humidity", DoubleType(), True),
                StructField("pressure", DoubleType(), True),
                StructField("windSpeed", DoubleType(), True),
                StructField("windBearing", DoubleType(), True),
                StructField("visibility", DoubleType(), True),
                StructField("dewPoint", DoubleType(), True),
                StructField("precipType", StringType(), True),
                StructField("icon", StringType(), True),
                StructField("summary", StringType(), True)
            ])
        ) \
        .csv(source_dir)
    
    transformed_df = stream_df.select(
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
    
    transformed_df = add_ingestion_metadata_streaming(transformed_df, source="weather_hourly_streaming")
    
    query = transformed_df.writeStream \
        .format("delta") \
        .outputMode("append") \
        .option("checkpointLocation", checkpoint_location) \
        .option("path", target_path) \
        .trigger(processingTime=trigger_interval) \
        .start()
    
    return query
