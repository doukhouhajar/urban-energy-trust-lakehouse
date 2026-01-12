from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count, sum as spark_sum, avg
from typing import Dict
import os


def aggregate_buildings_by_admin_area(
    spark: SparkSession,
    buildings_path: str,
    admin_areas_path: str,
    target_path: str,
    admin_level: int = 3
) -> DataFrame:
    print(f"Aggregating buildings by admin area (level {admin_level})...")
    
    buildings_df = spark.read.format("delta").load(buildings_path)
    admin_areas_df = spark.read.format("delta").load(admin_areas_path)
    
    # Placeholder aggregation (use spatial)
    # If buildings have coordinates, join spatially
    
    admin_col = f"gid_{admin_level}" if f"gid_{admin_level}" in admin_areas_df.columns else "gid_3"
    name_col = f"NAME_{admin_level}" if f"NAME_{admin_level}" in admin_areas_df.columns else "district"
    

    aggregation = admin_areas_df.select(
        col(admin_col).alias("admin_area_id"),
        col(name_col).alias("admin_area_name"),
        lit(0).alias("building_count"),  # count(*) from spatial join
        lit(0.0).alias("total_area_m2")   # sum(ST_Area(buildings.geometry))
    )
    
    print(f"Warning: Placeholder aggregation created at {target_path}")
    print(f"   To implement: Use Sedona ST_Within for spatial join, then aggregate")
    
    aggregation.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(target_path)
    
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
        os.path.join(paths['bronze_root'], "osm_buildings"),
        os.path.join(paths['bronze_root'], "gadm_level3"),
        os.path.join(paths['gold_root'], "building_aggregations"),
        admin_level=3
    )
    
    print("BUILDING AGGREGATION COMPLETE")
    
    return agg_level3
