"""Quality scoring and aggregation"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, lit, avg, count, sum as spark_sum,
    concat_ws, coalesce, date_trunc
)
from typing import Dict


def compute_quality_scores(
    completeness_metrics: DataFrame,
    temporal_metrics: DataFrame,
    business_metrics: DataFrame,
    schema_metrics: DataFrame,
    config: Dict
) -> DataFrame:
    """
    Compute overall quality scores (0-100) from individual metric DataFrames
    
    Formula:
    Quality Score = 
        40 * (1 - completeness_error_rate) +
        25 * (1 - temporal_coherence_error_rate) +
        20 * (1 - business_rule_violation_rate) +
        15 * schema_validity_score
    """
    quality_config = config.get('quality', {}).get('scoring', {})
    
    weights = {
        'completeness': quality_config.get('completeness_weight', 0.40),
        'temporal': quality_config.get('temporal_coherence_weight', 0.25),
        'business': quality_config.get('business_rules_weight', 0.20),
        'schema': quality_config.get('schema_validity_weight', 0.15)
    }
    
    # Join all metrics on partition columns and score_date
    # Assume all have: partition_cols + score_date
    partition_cols = ["household_id"]  # Default, can be parameterized
    
    # Prepare completeness component (0-1)
    completeness_component = completeness_metrics.select(
        col("household_id"),
        col("score_date"),
        (1 - coalesce(col("missing_rate"), lit(0.0))).alias("completeness_score")
    )
    
    # Prepare temporal component (0-1)
    temporal_component = temporal_metrics.select(
        col("household_id"),
        col("score_date"),
        (1 - coalesce(col("anomaly_rate"), lit(0.0))).alias("temporal_score")
    )
    
    # Prepare business component (0-1)
    business_component = business_metrics.select(
        col("household_id"),
        col("score_date"),
        (1 - coalesce(col("violation_rate"), lit(0.0))).alias("business_score")
    )
    
    # Prepare schema component (0-1)
    schema_component = schema_metrics.select(
        col("household_id"),
        col("score_date"),
        coalesce(col("schema_validity_score"), lit(1.0)).alias("schema_score")
    )
    
    # Join all components
    scores_df = completeness_component \
        .join(temporal_component, on=["household_id", "score_date"], how="outer") \
        .join(business_component, on=["household_id", "score_date"], how="outer") \
        .join(schema_component, on=["household_id", "score_date"], how="outer")
    
    # Fill nulls with 1.0 (perfect score if metric not computed)
    scores_df = scores_df.fillna({
        "completeness_score": 1.0,
        "temporal_score": 1.0,
        "business_score": 1.0,
        "schema_score": 1.0
    })
    
    # Compute weighted quality score (0-100)
    scores_df = scores_df.withColumn(
        "quality_score",
        (
            weights['completeness'] * col("completeness_score") +
            weights['temporal'] * col("temporal_score") +
            weights['business'] * col("business_score") +
            weights['schema'] * col("schema_score")
        ) * 100.0
    ).withColumn(
        "quality_category",
        when(col("quality_score") >= 90, "excellent")
        .when(col("quality_score") >= 80, "good")
        .when(col("quality_score") >= 70, "fair")
        .otherwise("poor")
    ).withColumn(
        "is_low_quality",
        when(col("quality_score") < quality_config.get('low_quality_threshold', 70.0), 1).otherwise(0)
    )
    
    return scores_df


def aggregate_quality_scores_by_partition(
    quality_scores: DataFrame,
    partition_cols: list,
    time_window: str = "day"
) -> DataFrame:
    """
    Aggregate quality scores by partition (e.g., ACORN group, block, area)
    """
    agg_df = quality_scores.groupBy(partition_cols + [
        date_trunc(time_window, col("score_date")).alias(f"{time_window}_window")
    ]).agg(
        avg(col("quality_score")).alias("avg_quality_score"),
        count("*").alias("household_count"),
        spark_sum(col("is_low_quality")).alias("low_quality_count"),
        avg(col("completeness_score")).alias("avg_completeness"),
        avg(col("temporal_score")).alias("avg_temporal"),
        avg(col("business_score")).alias("avg_business"),
        avg(col("schema_score")).alias("avg_schema")
    ).withColumn(
        "low_quality_rate",
        col("low_quality_count") / col("household_count")
    )
    
    return agg_df


def write_quality_scores(
    quality_scores: DataFrame,
    target_path: str,
    mode: str = "append"
):
    """Write quality scores to Delta table"""
    quality_scores.write \
        .format("delta") \
        .mode(mode) \
        .option("mergeSchema", "true") \
        .save(target_path)


def write_quality_incidents(
    incidents: DataFrame,
    target_path: str,
    mode: str = "append"
):
    """Write quality incidents to Delta table"""
    incidents.write \
        .format("delta") \
        .mode(mode) \
        .option("mergeSchema", "true") \
        .save(target_path)
