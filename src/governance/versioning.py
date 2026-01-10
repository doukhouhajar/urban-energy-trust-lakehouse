"""Delta Lake time travel and versioning utilities"""

from pyspark.sql import SparkSession, DataFrame
from delta.tables import DeltaTable
from typing import Optional


def get_table_version(spark: SparkSession, table_path: str) -> int:
    """Get current version of a Delta table"""
    try:
        if DeltaTable.isDeltaTable(spark, table_path):
            delta_table = DeltaTable.forPath(spark, table_path)
            history = delta_table.history(1)
            if history.count() > 0:
                return history.select("version").first()[0]
        return 0
    except Exception:
        return 0


def read_table_at_version(spark: SparkSession, table_path: str, version: int) -> DataFrame:
    """Read a Delta table at a specific version (time travel)"""
    return spark.read.format("delta").option("versionAsOf", version).load(table_path)


def read_table_at_timestamp(spark: SparkSession, table_path: str, timestamp: str) -> DataFrame:
    """Read a Delta table at a specific timestamp (time travel)"""
    return spark.read.format("delta").option("timestampAsOf", timestamp).load(table_path)


def get_table_history(spark: SparkSession, table_path: str, limit: int = 20) -> DataFrame:
    """Get version history of a Delta table"""
    if DeltaTable.isDeltaTable(spark, table_path):
        delta_table = DeltaTable.forPath(spark, table_path)
        return delta_table.history(limit)
    else:
        return spark.createDataFrame([], schema="version INT, timestamp TIMESTAMP, operation STRING")


def restore_table_to_version(spark: SparkSession, table_path: str, version: int, restore_path: Optional[str] = None):
    """
    Restore a table to a specific version
    
    Args:
        spark: SparkSession
        table_path: Path to Delta table
        version: Version to restore to
        restore_path: Optional path for restored table (if None, overwrites original)
    """
    df = read_table_at_version(spark, table_path, version)
    
    target_path = restore_path if restore_path else table_path
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(target_path)


def compare_versions(
    spark: SparkSession,
    table_path: str,
    version1: int,
    version2: int
) -> DataFrame:
    """
    Compare two versions of a Delta table
    Returns differences in row counts and schema
    """
    df1 = read_table_at_version(spark, table_path, version1)
    df2 = read_table_at_version(spark, table_path, version2)
    
    count1 = df1.count()
    count2 = df2.count()
    
    # Compare schemas
    schema1 = df1.schema
    schema2 = df2.schema
    
    comparison = {
        "version1": version1,
        "version2": version2,
        "count_v1": count1,
        "count_v2": count2,
        "row_diff": count2 - count1,
        "schema_equal": schema1 == schema2
    }
    
    return spark.createDataFrame([comparison])


def vacuum_table(spark: SparkSession, table_path: str, retention_hours: int = 168):
    """
    Vacuum a Delta table to remove old files
    
    Args:
        spark: SparkSession
        table_path: Path to Delta table
        retention_hours: Retention period in hours (default 7 days)
    """
    if DeltaTable.isDeltaTable(spark, table_path):
        delta_table = DeltaTable.forPath(spark, table_path)
        delta_table.vacuum(retentionHours=retention_hours)


def optimize_table(spark: SparkSession, table_path: str, zorder_cols: Optional[list] = None):
    """
    Optimize a Delta table (Z-order clustering)
    
    Args:
        spark: SparkSession
        table_path: Path to Delta table
        zorder_cols: Columns for Z-ordering (optional)
    """
    if DeltaTable.isDeltaTable(spark, table_path):
        delta_table = DeltaTable.forPath(spark, table_path)
        if zorder_cols:
            delta_table.optimize().executeZOrderBy(zorder_cols)
        else:
            delta_table.optimize().executeCompaction()
