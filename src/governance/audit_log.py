"""Pipeline audit logging for traceability"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, current_timestamp, count
from typing import Dict, Optional
from delta.tables import DeltaTable
import os
import json


def log_pipeline_run(
    spark: SparkSession,
    audit_log_path: str,
    pipeline_name: str,
    config: Dict,
    input_versions: Optional[Dict[str, int]] = None,
    output_paths: Optional[Dict[str, str]] = None,
    row_counts: Optional[Dict[str, int]] = None,
    rejected_rows: Optional[int] = None,
    quality_score_summary: Optional[Dict] = None,
    status: str = "SUCCESS",
    error_message: Optional[str] = None
):
    """
    Log a pipeline run to the audit log Delta table
    
    Args:
        spark: SparkSession
        audit_log_path: Path to audit log Delta table
        pipeline_name: Name of the pipeline
        config: Configuration used
        input_versions: Dict of input table -> version number
        output_paths: Dict of output_name -> Delta table path
        row_counts: Dict of table -> row count
        rejected_rows: Number of rejected rows
        quality_score_summary: Dict with quality metrics
        status: Pipeline status (SUCCESS, FAILED, PARTIAL)
        error_message: Error message if failed
    """
    # Create audit log entry
    audit_entry = {
        "pipeline_name": pipeline_name,
        "run_timestamp": current_timestamp(),
        "status": status,
        "error_message": error_message if error_message else None,
        "config_snapshot": json.dumps(config),
        "input_versions": json.dumps(input_versions or {}),
        "output_paths": json.dumps(output_paths or {}),
        "row_counts": json.dumps(row_counts or {}),
        "rejected_rows": rejected_rows or 0,
        "quality_score_summary": json.dumps(quality_score_summary or {})
    }
    
    # Create DataFrame from audit entry
    audit_df = spark.createDataFrame([audit_entry], schema="""
        pipeline_name STRING,
        run_timestamp TIMESTAMP,
        status STRING,
        error_message STRING,
        config_snapshot STRING,
        input_versions STRING,
        output_paths STRING,
        row_counts STRING,
        rejected_rows BIGINT,
        quality_score_summary STRING
    """)
    
    # Create or append to audit log
    if os.path.exists(audit_log_path) and DeltaTable.isDeltaTable(spark, audit_log_path):
        # Append to existing table
        audit_df.write.format("delta").mode("append").save(audit_log_path)
    else:
        # Create new table
        os.makedirs(os.path.dirname(audit_log_path), exist_ok=True)
        audit_df.write.format("delta").mode("overwrite").save(audit_log_path)


def get_table_row_count(spark: SparkSession, table_path: str) -> int:
    """Get row count from a Delta table"""
    try:
        df = spark.read.format("delta").load(table_path)
        return df.count()
    except Exception as e:
        print(f"Warning: Could not get row count for {table_path}: {e}")
        return 0


def get_table_version(spark: SparkSession, table_path: str) -> int:
    """Get current version of a Delta table"""
    try:
        if DeltaTable.isDeltaTable(spark, table_path):
            delta_table = DeltaTable.forPath(spark, table_path)
            history = delta_table.history(1)
            if history.count() > 0:
                return history.select("version").first()[0]
        return 0
    except Exception as e:
        print(f"Warning: Could not get version for {table_path}: {e}")
        return 0


def get_quality_score_summary(spark: SparkSession, quality_scores_path: str) -> Dict:
    """Get summary statistics from quality scores table"""
    try:
        df = spark.read.format("delta").load(quality_scores_path)
        summary = df.agg({
            "quality_score": "avg",
            "quality_score": "min",
            "quality_score": "max",
            "is_low_quality": "sum"
        }).collect()[0]
        
        return {
            "avg_quality_score": float(summary[0]) if summary[0] else 0.0,
            "min_quality_score": float(summary[1]) if summary[1] else 0.0,
            "max_quality_score": float(summary[2]) if summary[2] else 0.0,
            "low_quality_count": int(summary[3]) if summary[3] else 0
        }
    except Exception as e:
        print(f"Warning: Could not get quality summary for {quality_scores_path}: {e}")
        return {}


def query_audit_log(
    spark: SparkSession,
    audit_log_path: str,
    pipeline_name: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
) -> DataFrame:
    """Query audit log with optional filters"""
    df = spark.read.format("delta").load(audit_log_path)
    
    if pipeline_name:
        df = df.filter(col("pipeline_name") == pipeline_name)
    
    if status:
        df = df.filter(col("status") == status)
    
    return df.orderBy(col("run_timestamp").desc()).limit(limit)
