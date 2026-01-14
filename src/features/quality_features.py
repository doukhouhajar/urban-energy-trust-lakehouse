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
    
    # quality_scores has: completeness_score (which is 1 - missing_rate)
    # So missing_rate = 1 - completeness_score
    # IMPORTANT: Do NOT include quality_score in features - it's target leakage!
    # We're predicting future quality, so we can't use current quality_score
    scores_features = quality_scores.select(
        col("household_id"),
        col("score_date"),
        col("completeness_score"),
        (1 - col("completeness_score")).alias("missing_rate"),  # Derive missing_rate from completeness_score
        col("temporal_score"),  # Include component scores
        col("business_score"),
        col("schema_score")
        # Removed quality_score - this causes overfitting/target leakage!
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
    
    # Build aggregation expressions - only include columns that exist
    agg_exprs = [
        count("*").alias("reading_count"),
        avg(col("energy_kwh")).alias("avg_consumption"),
        stddev(col("energy_kwh")).alias("std_consumption"),
        spark_min(col("energy_kwh")).alias("min_consumption"),
        spark_max(col("energy_kwh")).alias("max_consumption")
    ]
    
    # Add weather columns if they exist
    if "temperature_celsius" in consumption.columns:
        agg_exprs.extend([
            avg(col("temperature_celsius")).alias("avg_temperature"),
            spark_min(col("temperature_celsius")).alias("min_temperature"),
            spark_max(col("temperature_celsius")).alias("max_temperature")
        ])
    
    # Group by household_id and date (truncated to day)
    consumption_features = consumption.groupBy(
        col("household_id"),
        date_trunc("day", col("timestamp")).alias("score_date")
    ).agg(*agg_exprs).withColumn(
        "consumption_cv",  # Coefficient of variation
        when(col("avg_consumption") > 0, col("std_consumption") / col("avg_consumption")).otherwise(0.0)
    ).withColumn(
        # Compute day_of_week and month from score_date (which is already date_trunc'd)
        "day_of_week",
        dayofweek(col("score_date"))
    ).withColumn(
        "month",
        month(col("score_date"))
    )
    
    # CRITICAL: Remove ALL current-day component scores - they cause target leakage!
    # The target is future quality, which is calculated from the same component scores.
    # Using current component scores = using the target to predict itself!
    # We can ONLY use historical rolling averages (7d, 30d) as features.
    # Keep only missing_rate for historical calculations, but remove current-day scores
    scores_features_for_ml = scores_features.select(
        col("household_id"),
        col("score_date"),
        # DO NOT include current-day component scores - they cause overfitting!
        # Only include missing_rate for historical rolling calculations
        col("missing_rate")
        # Removed: completeness_score, temporal_score, business_score, schema_score
        # These are calculated from the same data as the target = target leakage!
    )
    
    features_df = scores_features_for_ml \
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
        "consumption_cv": 0.0,
        "missing_rate": 0.0
        # Removed: completeness_score, temporal_score, business_score, schema_score
        # These are not included as features anymore to prevent target leakage
    })
    
    window_spec = Window.partitionBy("household_id").orderBy("score_date").rowsBetween(-feature_windows[0], 0)
    window_30_spec = Window.partitionBy("household_id").orderBy("score_date").rowsBetween(-feature_windows[1], 0)
    
    # Load historical scores ONLY for calculating rolling averages (not as direct features)
    historical_scores = quality_scores.select(
        col("household_id"),
        col("score_date"),
        col("completeness_score"),
        col("temporal_score"),
        col("business_score"),
        col("schema_score")
    )
    
    # Join historical scores to calculate rolling averages
    # Note: Rolling averages that include current day are OK (they're averages, not exact values)
    features_df = features_df.join(
        historical_scores.alias("hist"),
        on=["household_id", "score_date"],
        how="left"
    )
    
    # Calculate rolling features - using averages is OK (less leakage than exact values)
    features_df = features_df.withColumn(
        "missing_rate_7d",
        avg(col("missing_rate")).over(window_spec)
    ).withColumn(
        "missing_rate_30d",
        avg(col("missing_rate")).over(window_30_spec)
    ).withColumn(
        "incident_count_7d",
        spark_sum(col("total_incidents")).over(window_spec)
    ).withColumn(
        "incident_count_30d",
        spark_sum(col("total_incidents")).over(window_30_spec)
    ).withColumn(
        "avg_completeness_score_7d",
        avg(col("hist.completeness_score")).over(window_spec)
    ).withColumn(
        "avg_completeness_score_30d",
        avg(col("hist.completeness_score")).over(window_30_spec)
    ).withColumn(
        "avg_temporal_score_7d",
        avg(col("hist.temporal_score")).over(window_spec)
    ).withColumn(
        "avg_business_score_7d",
        avg(col("hist.business_score")).over(window_spec)
    ).withColumn(
        "avg_schema_score_7d",
        avg(col("hist.schema_score")).over(window_spec)
    ).withColumn(
        "std_consumption_7d",
        avg(col("std_consumption")).over(window_spec)
    ).withColumn(
        "avg_consumption_7d",
        avg(col("avg_consumption")).over(window_spec)
    ).withColumn(
        "consumption_trend_7d",
        col("avg_consumption") - avg(col("avg_consumption")).over(window_spec)
    ).withColumn(
        "critical_incidents_7d",
        spark_sum(col("critical_incidents")).over(window_spec)
    ).withColumn(
        "completeness_incidents_7d",
        spark_sum(col("completeness_incidents")).over(window_spec)
    ).withColumn(
        "temporal_incidents_7d",
        spark_sum(col("temporal_incidents")).over(window_spec)
    ).withColumn(
        "business_incidents_7d",
        spark_sum(col("business_incidents")).over(window_spec)
    ).drop("hist.household_id", "hist.score_date", "hist.completeness_score", 
           "hist.temporal_score", "hist.business_score", "hist.schema_score")
    
    return features_df


def create_target_labels(
    spark: SparkSession,
    quality_scores_path: str,
    quality_incidents_path: str,
    target_window_days: int = 1,
    low_quality_threshold: float = 70.0,
    high_incident_threshold: int = 5,
    auto_adjust_threshold: bool = True
) -> DataFrame:
    """
    Label = 1 if next day/week has low quality score OR high incident count
    """
    print("Creating target labels...")
    
    quality_scores = spark.read.format("delta").load(quality_scores_path)
    quality_incidents = spark.read.format("delta").load(quality_incidents_path)
    
    # Debug: Check what data we have
    total_scores = quality_scores.count()
    unique_dates = quality_scores.select("score_date").distinct().orderBy("score_date")
    date_count = unique_dates.count()
    
    print(f"   Total quality scores: {total_scores}")
    print(f"   Unique dates: {date_count}")
    
    if date_count > 0:
        print("   Date range:")
        unique_dates.show(10, truncate=False)
    
    if date_count < (target_window_days + 1):
        print(f"   WARNING: Only {date_count} days of data, but need at least {target_window_days + 1} days for future prediction")
        print(f"   Consider reducing target_window_days from {target_window_days} to {max(0, date_count - 1)}")
        if date_count <= 1:
            raise ValueError(
                f"Insufficient data for ML training: Only {date_count} day(s) of quality scores. "
                f"Need at least {target_window_days + 1} days. "
                f"Either reduce target_window_days in config or ensure you have multiple days of consumption data."
            )
    
    # Use date arithmetic join instead of lead() - more reliable
    from pyspark.sql.functions import date_add
    
    # Auto-adjust threshold if needed to create balanced labels
    if auto_adjust_threshold:
        # Get quality score percentiles to find a balanced threshold
        # Use multiple percentiles to find one that gives ~20-40% positive class
        quality_percentiles = quality_scores.approxQuantile("quality_score", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], 0.1)
        if quality_percentiles and len(quality_percentiles) > 0:
            # Try median (50th percentile) first - should give ~50% positive if scores are well-distributed
            # If median is too high, use 40th percentile; if too low, use 60th
            median_score = quality_percentiles[3] if len(quality_percentiles) > 3 else quality_percentiles[-1]
            # Use median as threshold - more balanced than 30th percentile
            adjusted_threshold = median_score
            print(f"   Auto-adjusting threshold: {low_quality_threshold:.2f} -> {adjusted_threshold:.2f} (median/50th percentile)")
            low_quality_threshold = adjusted_threshold
    
    # Create future quality scores: shift each score_date forward by target_window_days
    # This gives us the "future" quality for prediction
    future_quality = quality_scores.select(
        col("household_id"),
        date_add(col("score_date"), target_window_days).alias("future_score_date"),
        col("quality_score").alias("future_quality_score"),
        col("is_low_quality").alias("future_is_low_quality")
    )
    
    # Get future incidents aggregated by date
    future_incidents = quality_incidents.groupBy(
        col("entity_id").alias("household_id"),
        date_trunc("day", col("incident_timestamp")).alias("incident_date")
    ).agg(
        count("*").alias("incident_count")
    )
    
    # Join: for each (household_id, score_date), find the quality at (score_date + target_window_days)
    # This creates the target labels
    # Join condition: current score_date + target_window_days = future_score_date
    labels_df = quality_scores.alias("current").select(
        col("current.household_id"),
        col("current.score_date")
    ).join(
        future_quality.alias("future"),
        (col("current.household_id") == col("future.household_id")) & 
        (date_add(col("current.score_date"), target_window_days) == col("future.future_score_date")),
        how="inner"
    ).select(
        col("current.household_id"),
        col("current.score_date"),
        col("future.future_score_date"),  # Add this so we can join with incidents
        col("future.future_quality_score"),
        col("future.future_is_low_quality")
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
            (col("future_incident_count") >= high_incident_threshold) |
            (col("future_incident_count") > 0),  # Any incident is a risk signal
            1
        ).otherwise(0)
    )
    
    # Debug: Check label distribution before filtering
    label_counts = labels_df.groupBy("is_high_risk").count().collect()
    print(f"   Label distribution before final select:")
    for row in label_counts:
        print(f"     Label {row['is_high_risk']}: {row['count']} samples")
    
    labels_df = labels_df.select(
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
        _join_path(paths['gold_root'], "quality_scores"),
        _join_path(paths['gold_root'], "quality_incidents"),
        _join_path(paths['silver_root'], "household_enriched"),
        feature_windows=ml_config.get('feature_windows', {}).get('missing_rate_days', [7, 30])
    )
    
    labels_df = create_target_labels(
        spark,
        _join_path(paths['gold_root'], "quality_scores"),
        _join_path(paths['gold_root'], "quality_incidents"),
        target_window_days=ml_config.get('target_window_days', 1),
        low_quality_threshold=ml_config.get('low_quality_threshold', 70.0),
        high_incident_threshold=ml_config.get('high_incident_threshold', 5),
        auto_adjust_threshold=True  # Automatically adjust threshold for balanced labels
    )
    
    # Debug: print row counts
    features_count = features_df.count()
    labels_count = labels_df.count()
    print(f"   Features DataFrame: {features_count} rows")
    print(f"   Labels DataFrame: {labels_count} rows")
    
    if labels_count == 0:
        print("   ERROR: Labels DataFrame is empty!")
        print("   This means no future dates are available for target labels.")
        print("   Possible causes:")
        print("   - Quality scores table has insufficient data (need at least target_window_days + 1 days)")
        print("   - All future_score_date values are null (no future data to predict)")
        print("   - Consider using a smaller target_window_days or checking your data date range")
        raise ValueError("Cannot create ML dataset: Labels DataFrame is empty. Check your quality scores data.")
    
    # Show sample labels for debugging
    print("   Sample labels:")
    labels_df.select("household_id", "score_date", "target").show(5, truncate=False)
    
    # Show sample features for debugging
    if features_count > 0:
        print("   Sample features (first few columns):")
        feature_cols_sample = [c for c in features_df.columns[:5]]
        features_df.select(*feature_cols_sample).show(5, truncate=False)
    
    ml_dataset = features_df.join(
        labels_df,
        on=["household_id", "score_date"],
        how="inner"
    ).filter(
        col("target").isNotNull()  # Only keep rows with labels
    )
    
    ml_count = ml_dataset.count()
    print(f"   Joined ML dataset: {ml_count} rows")
    
    if ml_count == 0:
        print("   ERROR: ML dataset is empty after join!")
        print("   This might be because:")
        print("   - Features and labels don't have matching (household_id, score_date) pairs")
        print("   - Date formats don't match between features and labels")
        print("   - Date ranges don't overlap")
        
        # Try to diagnose the issue
        if features_count > 0 and labels_count > 0:
            print("\n   Diagnosing join issue...")
            features_dates = features_df.select("score_date").distinct().orderBy("score_date")
            labels_dates = labels_df.select("score_date").distinct().orderBy("score_date")
            
            print("   Features date range:")
            features_dates.show(5, truncate=False)
            print("   Labels date range:")
            labels_dates.show(5, truncate=False)
        
        raise ValueError("Cannot create ML dataset: Join resulted in 0 rows. Check date alignment between features and labels.")
    
    return ml_dataset
