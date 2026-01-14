from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count, sum as spark_sum, avg, lit
from typing import Dict
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


def aggregate_buildings_by_admin_area(
    spark: SparkSession,
    buildings_path: str,
    admin_areas_path: str,
    target_path: str,
    admin_level: int = 3
) -> DataFrame:
    print(f"Aggregating buildings by admin area (level {admin_level})...")
    
    try:
        buildings_df = spark.read.format("delta").load(buildings_path)
        print(f"   Loaded buildings: {buildings_df.count()} records")
    except Exception as e:
        print(f"   Warning: Could not load buildings from {buildings_path}: {e}")
        buildings_df = None
    
    admin_areas_df = spark.read.format("delta").load(admin_areas_path)
    
    # Debug: print available columns
    print(f"   Available columns in admin_areas: {admin_areas_df.columns}")
    
    # Try to find the correct column names
    admin_col = None
    name_col = None
    
    # Try different possible column names
    possible_admin_cols = [f"gid_{admin_level}", f"GID_{admin_level}", "gid_3", "GID_3"]
    possible_name_cols = [f"NAME_{admin_level}", f"name_{admin_level}", "district", "District", "NAME_3", "name_3"]
    
    for col_name in possible_admin_cols:
        if col_name in admin_areas_df.columns:
            admin_col = col_name
            break
    
    for col_name in possible_name_cols:
        if col_name in admin_areas_df.columns:
            name_col = col_name
            break
    
    # If we still don't have columns, use what's available or create placeholders
    if admin_col is None:
        print(f"   Warning: Could not find admin ID column, using row number")
        from pyspark.sql.functions import monotonically_increasing_id
        admin_areas_df = admin_areas_df.withColumn("_row_id", monotonically_increasing_id())
        admin_col = "_row_id"
    
    if name_col is None:
        print(f"   Warning: Could not find name column, using placeholder")
        admin_areas_df = admin_areas_df.withColumn("_name_placeholder", lit(f"Area_Level_{admin_level}"))
        name_col = "_name_placeholder"
    
    # Create aggregation with available columns
    select_cols = [
        col(admin_col).alias("admin_area_id"),
        lit(0).alias("building_count"),  # count(*) from spatial join
        lit(0.0).alias("total_area_m2")   # sum(ST_Area(buildings.geometry))
    ]
    
    if name_col:
        select_cols.insert(1, col(name_col).alias("admin_area_name"))
    else:
        select_cols.insert(1, lit(f"Area_Level_{admin_level}").alias("admin_area_name"))
    
    aggregation = admin_areas_df.select(*select_cols)
    
    print(f"   Warning: Placeholder aggregation created at {target_path}")
    print(f"   To implement: Use Sedona ST_Within for spatial join, then aggregate")
    print(f"   Created {aggregation.count()} admin area records")
    
    aggregation.write.format("delta").mode("overwrite").save(target_path)
    
    return aggregation


def create_building_aggregations(
    spark: SparkSession,
    config: Dict
) -> DataFrame:
    paths = config['paths']
    
    print("=" * 60)
    print("BUILDING AGGREGATION")
    print("=" * 60)
    
    agg_level3 = aggregate_buildings_by_admin_area(
        spark,
        _join_path(paths['bronze_root'], "osm_buildings"),
        _join_path(paths['bronze_root'], "gadm_level3"),
        _join_path(paths['gold_root'], "building_aggregations"),
        admin_level=3
    )
    
    print("BUILDING AGGREGATION COMPLETE")
    
    return agg_level3
