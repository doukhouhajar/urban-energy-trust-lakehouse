from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, isnan, isnull, count, sum as spark_sum,
    avg, stddev, min as spark_min, max as spark_max,
    window, lead, lag, row_number, date_trunc, to_date,
    abs as spark_abs, expr, lit, concat_ws
)
from pyspark.sql.window import Window
from typing import Dict, List, Tuple
from datetime import timedelta


def check_completeness(
    df: DataFrame,
    partition_cols: List[str],
    timestamp_col: str,
    expected_interval_minutes: int = 30,
    missing_threshold: float = 0.10
) -> Tuple[DataFrame, DataFrame]:
    completeness_df = df.groupBy(partition_cols + [date_trunc("day", col(timestamp_col)).alias("score_date")]) \
        .agg(
            count("*").alias("actual_records"),
            count(col(timestamp_col)).alias("non_null_records"),
            spark_min(col(timestamp_col)).alias("min_timestamp"),
            spark_max(col(timestamp_col)).alias("max_timestamp")
        )
    
    records_per_day = int(24 * 60 / expected_interval_minutes)
    
    completeness_df = completeness_df.withColumn(
        "expected_records",
        lit(records_per_day)
    ).withColumn(
        "missing_records",
        col("expected_records") - col("actual_records")
    ).withColumn(
        "completeness_rate",
        col("actual_records") / col("expected_records")
    ).withColumn(
        "missing_rate",
        1 - col("completeness_rate")
    ).withColumn(
        "completeness_error",
        when(col("missing_rate") > missing_threshold, 1).otherwise(0)
    )
    
    incidents_df = completeness_df.filter(col("missing_rate") > missing_threshold) \
        .select(
            concat_ws("_", *partition_cols).alias("entity_id"),
            col("score_date"),
            col("score_date").alias("incident_timestamp"),
            lit("completeness_high_missing_rate").alias("rule_name"),
            when(col("missing_rate") > 0.25, "critical").otherwise("warning").alias("severity"),
            concat_ws("|",
                col("actual_records"),
                col("expected_records"),
                (col("missing_rate") * 100).cast("string").alias("missing_pct")
            ).alias("evidence")
        )
    
    return completeness_df, incidents_df


def check_temporal_coherence(
    df: DataFrame,
    partition_cols: List[str],
    timestamp_col: str,
    expected_interval_minutes: int = 30
) -> Tuple[DataFrame, DataFrame]:
    window_spec = Window.partitionBy(partition_cols).orderBy(col(timestamp_col))
    
    df_with_lag = df.withColumn(
        "prev_timestamp",
        lag(col(timestamp_col), 1).over(window_spec)
    ).withColumn(
        "next_timestamp",
        lead(col(timestamp_col), 1).over(window_spec)
    )
    
    df_with_diffs = df_with_lag.withColumn(
        "time_diff_minutes",
        (col(timestamp_col).cast("long") - col("prev_timestamp").cast("long")) / 60
    ).withColumn(
        "is_gap",
        when(
            (col("time_diff_minutes").isNotNull()) &
            (spark_abs(col("time_diff_minutes") - expected_interval_minutes) > 1),
            1
        ).otherwise(0)
    ).withColumn(
        "is_duplicate",
        when(col("timestamp") == col("prev_timestamp"), 1).otherwise(0)
    ).withColumn(
        "is_out_of_order",
        when(
            col("prev_timestamp").isNotNull() &
            (col("timestamp") < col("prev_timestamp")),
            1
        ).otherwise(0)
    )
    
    temporal_metrics = df_with_diffs.groupBy(partition_cols + [
        date_trunc("day", col(timestamp_col)).alias("score_date")
    ]).agg(
        count("*").alias("total_records"),
        spark_sum(col("is_gap")).alias("gap_count"),
        spark_sum(col("is_duplicate")).alias("duplicate_count"),
        spark_sum(col("is_out_of_order")).alias("out_of_order_count")
    ).withColumn(
        "total_anomalies",
        col("gap_count") + col("duplicate_count") + col("out_of_order_count")
    ).withColumn(
        "anomaly_rate",
        col("total_anomalies") / col("total_records")
    ).withColumn(
        "temporal_coherence_error",
        when(col("anomaly_rate") > 0.05, 1).otherwise(0)
    )
    
    incidents_df = df_with_diffs.filter(
        (col("is_gap") == 1) | (col("is_duplicate") == 1) | (col("is_out_of_order") == 1)
    ).select(
        concat_ws("_", *partition_cols).alias("entity_id"),
        col(timestamp_col).alias("incident_timestamp"),
        date_trunc("day", col(timestamp_col)).alias("score_date"),
        when(col("is_gap") == 1, "temporal_gap")
            .when(col("is_duplicate") == 1, "temporal_duplicate")
            .otherwise("temporal_out_of_order").alias("rule_name"),
        lit("warning").alias("severity"),
        concat_ws("|",
            when(col("is_gap") == 1, col("time_diff_minutes")).otherwise(lit("")),
            when(col("is_duplicate") == 1, "duplicate").otherwise(lit("")),
            when(col("is_out_of_order") == 1, "out_of_order").otherwise(lit(""))
        ).alias("evidence")
    ).dropDuplicates(["entity_id", "incident_timestamp", "rule_name"])
    
    return temporal_metrics, incidents_df


def check_business_rules(
    df: DataFrame,
    partition_cols: List[str],
    value_col: str,
    timestamp_col: str,
    min_value: float = 0.0,
    max_value: float = 50.0,
    z_score_threshold: float = 5.0
) -> Tuple[DataFrame, DataFrame]:
    stats_df = df.filter(col(value_col).isNotNull()).groupBy(partition_cols).agg(
        avg(col(value_col)).alias("mean_value"),
        stddev(col(value_col)).alias("std_value"),
        spark_min(col(value_col)).alias("min_value"),
        spark_max(col(value_col)).alias("max_value")
    )
    
    df_with_stats = df.join(stats_df, on=partition_cols, how="left")
    
    df_with_violations = df_with_stats.withColumn(
        "is_negative",
        when(col(value_col) < min_value, 1).otherwise(0)
    ).withColumn(
        "is_above_max",
        when(col(value_col) > max_value, 1).otherwise(0)
    ).withColumn(
        "z_score",
        when(
            (col("std_value").isNotNull()) & (col("std_value") > 0),
            (col(value_col) - col("mean_value")) / col("std_value")
        ).otherwise(0.0)
    ).withColumn(
        "is_extreme_spike",
        when(spark_abs(col("z_score")) > z_score_threshold, 1).otherwise(0)
    )
    
    business_metrics = df_with_violations.groupBy(partition_cols + [
        date_trunc("day", col(timestamp_col)).alias("score_date")
    ]).agg(
        count("*").alias("total_records"),
        spark_sum(col("is_negative")).alias("negative_count"),
        spark_sum(col("is_above_max")).alias("above_max_count"),
        spark_sum(col("is_extreme_spike")).alias("extreme_spike_count")
    ).withColumn(
        "total_violations",
        col("negative_count") + col("above_max_count") + col("extreme_spike_count")
    ).withColumn(
        "violation_rate",
        col("total_violations") / col("total_records")
    ).withColumn(
        "business_rule_error",
        when(col("violation_rate") > 0.01, 1).otherwise(0)
    )
    
    incidents_df = df_with_violations.filter(
        (col("is_negative") == 1) | (col("is_above_max") == 1) | (col("is_extreme_spike") == 1)
    ).select(
        concat_ws("_", *partition_cols).alias("entity_id"),
        col(timestamp_col).alias("incident_timestamp"),
        date_trunc("day", col(timestamp_col)).alias("score_date"),
        when(col("is_negative") == 1, "business_negative_consumption")
            .when(col("is_above_max") == 1, "business_above_max")
            .otherwise("business_extreme_spike").alias("rule_name"),
        when(col("is_negative") == 1, "critical").otherwise("warning").alias("severity"),
        concat_ws("|",
            when(col("is_negative") == 1, col(value_col)).otherwise(lit("")),
            when(col("is_above_max") == 1, col(value_col)).otherwise(lit("")),
            when(col("is_extreme_spike") == 1, col("z_score").cast("string")).otherwise(lit(""))
        ).alias("evidence")
    ).dropDuplicates(["entity_id", "incident_timestamp", "rule_name"])
    
    return business_metrics, incidents_df


def check_schema_validity(
    df: DataFrame,
    partition_cols: List[str],
    timestamp_col: str,
    allowed_tariffs: List[str] = None,
    allowed_acorn_groups: List[str] = None,
    energy_range: Tuple[float, float] = (0.0, 100.0)
) -> Tuple[DataFrame, DataFrame]:

    schema_checks = df.withColumn(
        "has_null_timestamp",
        when(col(timestamp_col).isNull(), 1).otherwise(0)
    ).withColumn(
        "energy_in_range",
        when(
            (col("energy_kwh").isNotNull()) &
            (col("energy_kwh") >= energy_range[0]) &
            (col("energy_kwh") <= energy_range[1]),
            0
        ).otherwise(1)
    )
    
    # Add tariff and ACORN checks if columns exist
    if "tariff_type" in df.columns and allowed_tariffs:
        schema_checks = schema_checks.withColumn(
            "invalid_tariff",
            when(
                col("tariff_type").isNotNull() &
                (~col("tariff_type").isin(allowed_tariffs)),
                1
            ).otherwise(0)
        )
    else:
        schema_checks = schema_checks.withColumn("invalid_tariff", lit(0))
    
    if "acorn_group" in df.columns and allowed_acorn_groups:
        schema_checks = schema_checks.withColumn(
            "invalid_acorn",
            when(
                col("acorn_group").isNotNull() &
                (~col("acorn_group").isin(allowed_acorn_groups)),
                1
            ).otherwise(0)
        )
    else:
        schema_checks = schema_checks.withColumn("invalid_acorn", lit(0))
    
    schema_metrics = schema_checks.groupBy(partition_cols + [
        date_trunc("day", col(timestamp_col)).alias("score_date")
    ]).agg(
        count("*").alias("total_records"),
        spark_sum(col("has_null_timestamp")).alias("null_timestamp_count"),
        spark_sum(col("energy_in_range")).alias("out_of_range_count"),
        spark_sum(col("invalid_tariff")).alias("invalid_tariff_count"),
        spark_sum(col("invalid_acorn")).alias("invalid_acorn_count")
    ).withColumn(
        "total_schema_errors",
        col("null_timestamp_count") + col("out_of_range_count") +
        col("invalid_tariff_count") + col("invalid_acorn_count")
    ).withColumn(
        "schema_error_rate",
        col("total_schema_errors") / col("total_records")
    ).withColumn(
        "schema_validity_score",
        1 - col("schema_error_rate")
    )
    

    incidents_df = schema_checks.filter(
        (col("has_null_timestamp") == 1) |
        (col("energy_in_range") == 1) |
        (col("invalid_tariff") == 1) |
        (col("invalid_acorn") == 1)
    ).select(
        concat_ws("_", *partition_cols).alias("entity_id"),
        col(timestamp_col).alias("incident_timestamp"),
        date_trunc("day", col(timestamp_col)).alias("score_date"),
        when(col("has_null_timestamp") == 1, "schema_null_timestamp")
            .when(col("energy_in_range") == 1, "schema_out_of_range")
            .when(col("invalid_tariff") == 1, "schema_invalid_tariff")
            .otherwise("schema_invalid_acorn").alias("rule_name"),
        lit("warning").alias("severity"),
        lit("schema_violation").alias("evidence")
    ).dropDuplicates(["entity_id", "incident_timestamp", "rule_name"])
    
    return schema_metrics, incidents_df
