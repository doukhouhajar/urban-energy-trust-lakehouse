"""GADM administrative boundaries ingestion"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit
import geopandas as gpd
import os
from typing import Dict


def ingest_gadm_level2(
    spark: SparkSession,
    source_path: str,
    target_path: str,
    crs: str = "EPSG:4326"
) -> DataFrame:
    """
    Ingest GADM level 2 (counties) from GeoPackage to Delta
    
    Note: This uses GeoPandas to read the GPKG, then converts to Spark DataFrame
    For large files, consider using Sedona or PostGIS
    """
    print(f"Reading GADM Level 2 from {source_path}...")
    
    # Read with GeoPandas
    gdf = gpd.read_file(source_path, layer="ADM_2")
    
    print(f"Loaded {len(gdf)} Level 2 administrative areas")
    
    # Convert geometry to WKB (Well-Known Binary)
    gdf['geometry_wkb'] = gdf.geometry.apply(lambda geom: geom.wkb if geom else None)
    
    # Convert to Pandas DataFrame (drop geometry column, keep WKB)
    df_pandas = gdf.drop(columns=['geometry']).rename(columns={'geometry_wkb': 'geometry'})
    
    # Convert to Spark DataFrame
    df_spark = spark.createDataFrame(df_pandas)
    
    # Select key columns
    columns_to_select = [
        col("GID_2").alias("gid_2"),
        col("NAME_0").alias("country"),
        col("NAME_1").alias("region"),
        col("NAME_2").alias("county"),
        col("geometry").alias("geom_wkb")
    ]
    
    # Only select columns that exist
    available_cols = [c for c in columns_to_select if c._jc.toString().split("`")[-1].replace("`", "") in df_spark.columns]
    df_spark = df_spark.select(*available_cols)
    
    # Add metadata
    df_spark = df_spark.withColumn("admin_level", lit(2)) \
                       .withColumn("crs", lit(crs))
    
    # Write to Delta
    df_spark.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(target_path)
    
    print(f"✓ Ingested GADM Level 2 to {target_path}")
    
    return df_spark


def ingest_gadm_level3(
    spark: SparkSession,
    source_path: str,
    target_path: str,
    crs: str = "EPSG:4326"
) -> DataFrame:
    """
    Ingest GADM level 3 (districts/boroughs) from GeoPackage to Delta
    """
    print(f"Reading GADM Level 3 from {source_path}...")
    
    # Read with GeoPandas
    gdf = gpd.read_file(source_path, layer="ADM_3")
    
    print(f"Loaded {len(gdf)} Level 3 administrative areas")
    
    # Convert geometry to WKB
    gdf['geometry_wkb'] = gdf.geometry.apply(lambda geom: geom.wkb if geom else None)
    
    # Convert to Pandas DataFrame
    df_pandas = gdf.drop(columns=['geometry']).rename(columns={'geometry_wkb': 'geometry'})
    
    # Convert to Spark DataFrame
    df_spark = spark.createDataFrame(df_pandas)
    
    # Select key columns
    columns_to_select = [
        col("GID_3").alias("gid_3"),
        col("NAME_0").alias("country"),
        col("NAME_1").alias("region"),
        col("NAME_2").alias("county"),
        col("NAME_3").alias("district"),
        col("geometry").alias("geom_wkb")
    ]
    
    # Only select columns that exist
    available_cols = [c for c in columns_to_select if c._jc.toString().split("`")[-1].replace("`", "") in df_spark.columns]
    df_spark = df_spark.select(*available_cols)
    
    # Add metadata
    df_spark = df_spark.withColumn("admin_level", lit(3)) \
                       .withColumn("crs", lit(crs))
    
    # Write to Delta
    df_spark.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(target_path)
    
    print(f"✓ Ingested GADM Level 3 to {target_path}")
    
    return df_spark


def run_gadm_ingestion(spark: SparkSession, config: Dict) -> Dict[str, DataFrame]:
    """Run complete GADM ingestion"""
    paths = config['paths']
    geospatial_config = config.get('geospatial', {}).get('gadm', {})
    
    results = {}
    
    print("=" * 60)
    print("GADM INGESTION")
    print("=" * 60)
    
    # Ingest Level 2
    print("\n1. Ingesting GADM Level 2...")
    results['gadm_level2'] = ingest_gadm_level2(
        spark,
        paths['geospatial']['gadm_gpkg'],
        os.path.join(paths['bronze_root'], "gadm_level2"),
        crs=geospatial_config.get('crs', 'EPSG:4326')
    )
    
    # Ingest Level 3
    print("\n2. Ingesting GADM Level 3...")
    results['gadm_level3'] = ingest_gadm_level3(
        spark,
        paths['geospatial']['gadm_gpkg'],
        os.path.join(paths['bronze_root'], "gadm_level3"),
        crs=geospatial_config.get('crs', 'EPSG:4326')
    )
    
    print("\n" + "=" * 60)
    print("GADM INGESTION COMPLETE")
    print("=" * 60)
    
    return results
