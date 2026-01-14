from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit
import geopandas as gpd
import os
import tempfile
from typing import Dict


def _copy_hdfs_to_local(spark: SparkSession, hdfs_path: str) -> str:
    """Copy file from HDFS to temporary local location for GeoPandas to read."""
    if not hdfs_path.startswith(("hdfs://", "/")):
        # Already a local path
        return hdfs_path
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".gpkg")
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        # Use Spark's filesystem API to copy from HDFS to local
        hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
        fs = spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark.sparkContext._jvm.java.net.URI(hdfs_path), hadoop_conf
        )
        hdfs_file_path = spark.sparkContext._jvm.org.apache.hadoop.fs.Path(hdfs_path)
        local_file_path = spark.sparkContext._jvm.org.apache.hadoop.fs.Path(temp_path)
        
        fs.copyToLocalFile(hdfs_file_path, local_file_path)
        return temp_path
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise Exception(f"Failed to copy {hdfs_path} from HDFS: {e}")


def ingest_gadm_level2(
    spark: SparkSession,
    source_path: str,
    target_path: str,
    crs: str = "EPSG:4326"
) -> DataFrame:
    print(f"Reading GADM Level 2 from {source_path}...")
    
    # Handle HDFS paths by copying to temporary local file
    local_path = source_path
    temp_file_created = False
    
    # Check if it's an HDFS path (starts with hdfs:// or /) and not a local file:// path
    if source_path.startswith("hdfs://") or (source_path.startswith("/") and not source_path.startswith("file://")):
        # Check if file exists locally first (for backward compatibility)
        if not os.path.exists(source_path):
            local_path = _copy_hdfs_to_local(spark, source_path)
            temp_file_created = True
    
    # GADM layer names are ADM_ADM_0, ADM_ADM_1, ADM_ADM_2, etc.
    gdf = gpd.read_file(local_path, layer="ADM_ADM_2")
    
    print(f"Loaded {len(gdf)} Level 2 administrative areas")
    
    # Convert geometry to WKB (Well-Known Binary)
    gdf['geometry_wkb'] = gdf.geometry.apply(lambda geom: geom.wkb if geom else None)
    
    # Convert to Pandas DataFrame (drop geometry column, keep WKB)
    df_pandas = gdf.drop(columns=['geometry']).rename(columns={'geometry_wkb': 'geometry'})
    
    # Convert to Spark DataFrame
    df_spark = spark.createDataFrame(df_pandas)
    
    # key columns (GADM uses COUNTRY instead of NAME_0)
    # Check which columns actually exist in the DataFrame
    available_columns = df_spark.columns
    
    select_exprs = []
    if "GID_2" in available_columns:
        select_exprs.append(col("GID_2").alias("gid_2"))
    if "COUNTRY" in available_columns:
        select_exprs.append(col("COUNTRY").alias("country"))
    if "NAME_1" in available_columns:
        select_exprs.append(col("NAME_1").alias("region"))
    if "NAME_2" in available_columns:
        select_exprs.append(col("NAME_2").alias("county"))
    if "geometry" in available_columns:
        select_exprs.append(col("geometry").alias("geom_wkb"))
    
    if select_exprs:
        df_spark = df_spark.select(*select_exprs)
    else:
        print(f"   Warning: No expected columns found, keeping all columns")
    
    df_spark = df_spark.withColumn("admin_level", lit(2)) \
                       .withColumn("crs", lit(crs))
    
    # write to delta
    df_spark.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(target_path)
    
    print(f"Ingested GADM Level 2 to {target_path}")
    
    # Clean up temporary file if we created one
    if temp_file_created and os.path.exists(local_path):
        os.unlink(local_path)
    
    return df_spark


def ingest_gadm_level3(
    spark: SparkSession,
    source_path: str,
    target_path: str,
    crs: str = "EPSG:4326"
) -> DataFrame:
    print(f"Reading GADM Level 3 from {source_path}...")
    
    # Handle HDFS paths by copying to temporary local file
    local_path = source_path
    temp_file_created = False
    
    # Check if it's an HDFS path (starts with hdfs:// or /) and not a local file:// path
    if source_path.startswith("hdfs://") or (source_path.startswith("/") and not source_path.startswith("file://")):
        # Check if file exists locally first (for backward compatibility)
        if not os.path.exists(source_path):
            local_path = _copy_hdfs_to_local(spark, source_path)
            temp_file_created = True
    
    # GADM layer names are ADM_ADM_0, ADM_ADM_1, ADM_ADM_2, etc.
    gdf = gpd.read_file(local_path, layer="ADM_ADM_3")
    
    print(f"Loaded {len(gdf)} Level 3 administrative areas")
    
    # Convert geometry to WKB
    gdf['geometry_wkb'] = gdf.geometry.apply(lambda geom: geom.wkb if geom else None)
    
    df_pandas = gdf.drop(columns=['geometry']).rename(columns={'geometry_wkb': 'geometry'})
    
    df_spark = spark.createDataFrame(df_pandas)
    
    # Check which columns actually exist in the DataFrame
    available_columns = df_spark.columns
    
    select_exprs = []
    if "GID_3" in available_columns:
        select_exprs.append(col("GID_3").alias("gid_3"))
    if "COUNTRY" in available_columns:
        select_exprs.append(col("COUNTRY").alias("country"))
    if "NAME_1" in available_columns:
        select_exprs.append(col("NAME_1").alias("region"))
    if "NAME_2" in available_columns:
        select_exprs.append(col("NAME_2").alias("county"))
    if "NAME_3" in available_columns:
        select_exprs.append(col("NAME_3").alias("district"))
    if "geometry" in available_columns:
        select_exprs.append(col("geometry").alias("geom_wkb"))
    
    if select_exprs:
        df_spark = df_spark.select(*select_exprs)
    else:
        print(f"   Warning: No expected columns found, keeping all columns")
    
    df_spark = df_spark.withColumn("admin_level", lit(3)) \
                       .withColumn("crs", lit(crs))
    
    # write to delta
    df_spark.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(target_path)
    
    print(f"Ingested GADM Level 3 to {target_path}")
    
    # Clean up temporary file if we created one
    if temp_file_created and os.path.exists(local_path):
        os.unlink(local_path)
    
    return df_spark


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


def run_gadm_ingestion(spark: SparkSession, config: Dict) -> Dict[str, DataFrame]:
    paths = config['paths']
    geospatial_config = config.get('geospatial', {}).get('gadm', {})
    
    results = {}
    
    print("GADM INGESTION")
    
    print("\n1. Ingesting GADM Level 2...")
    results['gadm_level2'] = ingest_gadm_level2(
        spark,
        paths['geospatial']['gadm_gpkg'],
        _join_path(paths['bronze_root'], "gadm_level2"),
        crs=geospatial_config.get('crs', 'EPSG:4326')
    )
    
    print("\n2. Ingesting GADM Level 3...")
    results['gadm_level3'] = ingest_gadm_level3(
        spark,
        paths['geospatial']['gadm_gpkg'],
        _join_path(paths['bronze_root'], "gadm_level3"),
        crs=geospatial_config.get('crs', 'EPSG:4326')
    )
    print("GADM INGESTION COMPLETE")
    
    return results
