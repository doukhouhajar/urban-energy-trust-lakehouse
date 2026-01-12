from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, avg, sum as spark_sum, count, stddev, min as spark_min, max as spark_max,
    lag, lead, window, date_trunc, to_date, dayofweek, month,
    when, lit, coalesce, datediff, current_date, expr
)
from pyspark.sql.window import Window
from typing import Dict, List
import pandas as pd
import os


def compute_historical_features(
    spark: SparkSession,
    quality_scores_path: str,
    quality_incidents_path: str,
    consumption_silver_path: str,
    feature_windows: List[int] = [7, 30]
) -> DataFrame:
    """
    historical features for quality risk prediction:
    - Missing rate (rolling windows)
    - Incident counts by type (rolling windows)
    - Consumption volatility (rolling windows)
    - Weather statistics (rolling windows)
    """
    print("Computing historical features...")
    
    quality_scores = spark.read.format("delta").load(quality_scores_path)
    quality_incidents = spark.read.format("delta").load(quality_incidents_path)
    consumption = spark.read.format("delta").load(consumption_silver_path)
    
    scores_features = quality_scores.select(
        col("household_id"),
        col("score_date"),
        col("completeness_score"),
        col("missing_rate"),
        col("quality_score")
    )
    
    incident_features = quality_incidents.groupBy(
        col("entity_id").alias("household_id"),
        date_trunc("day", col("incident_timestamp")).alias("score_date")
    ).agg(
        count("*").alias("total_incidents"),
        spark_sum(when(col("rule_name").contains("completeness"), 1).otherwise(0)).alias("completeness_incidents"),
        spark_sum(when(col("rule_name").contains("temporal"), 1).otherwise(0)).alias("temporal_incidents"),
        spark_sum(when(col("rule_name").contains("business"), 1).otherwise(0)).alias("business_incidents"),
        spark_sum(when(col("rule_name").contains("schema"), 1).otherwise(0)).alias("schema_incidents"),
        spark_sum(when(col("severity") == "critical", 1).otherwise(0)).alias("critical_incidents"),
        spark_sum(when(col("severity") == "warning", 1).otherwise(0)).alias("warning_incidents")
    )
    
    consumption_features = consumption.groupBy(
        col("household_id"),
        date_trunc("day", col("timestamp")).alias("score_date")
    ).agg(
        count("*").alias("reading_count"),
        avg(col("energy_kwh")).alias("avg_consumption"),
        stddev(col("energy_kwh")).alias("std_consumption"),
        spark_min(col("energy_kwh")).alias("min_consumption"),
        spark_max(col("energy_kwh")).alias("max_consumption"),
        avg(col("temperature_celsius")).alias("avg_temperature"),
        spark_min(col("temperature_celsius")).alias("min_temperature"),
        spark_max(col("temperature_celsius")).alias("max_temperature"),
        dayofweek(col("timestamp")).alias("day_of_week"),
        month(col("timestamp")).alias("month")
    ).withColumn(
        "consumption_cv",  # Coefficient of variation
        when(col("avg_consumption") > 0, col("std_consumption") / col("avg_consumption")).otherwise(0.0)
    )
    
    features_df = scores_features \
        .join(incident_features, on=["household_id", "score_date"], how="outer") \
        .join(consumption_features, on=["household_id", "score_date"], how="outer")
    
    features_df = features_df.fillna({
        "total_incidents": 0,
        "completeness_incidents": 0,
        "temporal_incidents": 0,
        "business_incidents": 0,
        "schema_incidents": 0,
        "critical_incidents": 0,
        "warning_incidents": 0,
        "reading_count": 0,
        "avg_consumption": 0.0,
        "std_consumption": 0.0,
        "consumption_cv": 0.0
    })
    
    window_spec = Window.partitionBy("household_id").orderBy("score_date").rowsBetween(-feature_windows[0], 0)
    window_30_spec = Window.partitionBy("household_id").orderBy("score_date").rowsBetween(-feature_windows[1], 0)
    
    features_df = features_df.withColumn(
        "missing_rate_7d",
        avg(col("missing_rate")).over(window_spec)
    ).withColumn(
        "incident_count_7d",
        spark_sum(col("total_incidents")).over(window_spec)
    ).withColumn(
        "incident_count_30d",
        spark_sum(col("total_incidents")).over(window_30_spec)
    ).withColumn(
        "avg_quality_score_7d",
        avg(col("quality_score")).over(window_spec)
    ).withColumn(
        "std_consumption_7d",
        avg(col("std_consumption")).over(window_spec)
    )
    
    return features_df


def create_target_labels(
    spark: SparkSession,
    quality_scores_path: str,
    quality_incidents_path: str,
    target_window_days: int = 1,
    low_quality_threshold: float = 70.0,
    high_incident_threshold: int = 5
) -> DataFrame:
    """
    Label = 1 if next day/week has low quality score OR high incident count
    """
    print("Creating target labels...")
    
    quality_scores = spark.read.format("delta").load(quality_scores_path)
    quality_incidents = spark.read.format("delta").load(quality_incidents_path)
    
    window_spec = Window.partitionBy("household_id").orderBy("score_date")
    future_scores = quality_scores.withColumn(
        "future_score_date",
        lead(col("score_date"), target_window_days).over(window_spec)
    ).withColumn(
        "future_quality_score",
        lead(col("quality_score"), target_window_days).over(window_spec)
    ).withColumn(
        "future_is_low_quality",
        lead(col("is_low_quality"), target_window_days).over(window_spec)
    )
    
    future_incidents = quality_incidents.groupBy(
        col("entity_id").alias("household_id"),
        date_trunc("day", col("incident_timestamp")).alias("incident_date")
    ).agg(
        count("*").alias("incident_count")
    )
    
    labels_df = future_scores.select(
        col("household_id"),
        col("score_date"),
        col("future_quality_score"),
        col("future_is_low_quality")
    ).join(
        future_incidents.select(
            col("household_id"),
            col("incident_date").alias("future_score_date"),
            col("incident_count").alias("future_incident_count")
        ),
        on=["household_id", "future_score_date"],
        how="left"
    ).withColumn(
        "future_incident_count",
        coalesce(col("future_incident_count"), lit(0))
    ).withColumn(
        "is_high_risk",  # Target label
        when(
            (col("future_quality_score") < low_quality_threshold) |
            (col("future_incident_count") >= high_incident_threshold),
            1
        ).otherwise(0)
    ).select(
        col("household_id"),
        col("score_date"),
        col("is_high_risk").alias("target")
    )
    
    return labels_df


def prepare_ml_features(
    spark: SparkSession,
    config: Dict
) -> DataFrame:
    paths = config['paths']
    ml_config = config.get('ml', {}).get('quality_risk', {})
    
    features_df = compute_historical_features(
        spark,
        os.path.join(paths['gold_root'], "quality_scores"),
        os.path.join(paths['gold_root'], "quality_incidents"),
        os.path.join(paths['silver_root'], "household_enriched"),
        feature_windows=ml_config.get('feature_windows', {}).get('missing_rate_days', [7, 30])
    )
    
    labels_df = create_target_labels(
        spark,
        os.path.join(paths['gold_root'], "quality_scores"),
        os.path.join(paths['gold_root'], "quality_incidents"),
        target_window_days=ml_config.get('target_window_days', 1),
        low_quality_threshold=ml_config.get('low_quality_threshold', 70.0),
        high_incident_threshold=ml_config.get('high_incident_threshold', 5)
    )
    
    ml_dataset = features_df.join(
        labels_df,
        on=["household_id", "score_date"],
        how="inner"
    ).filter(
        col("target").isNotNull()  # Only keep rows with labels
    )
    
    return ml_dataset
