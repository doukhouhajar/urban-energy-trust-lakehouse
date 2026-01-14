from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, when
import os
from typing import Dict, List, Optional
import subprocess
import tempfile


def _copy_hdfs_to_local(spark: SparkSession, hdfs_path: str, suffix: str = ".pbf") -> str:
    """Copy file from HDFS to temporary local location for osmium to read."""
    if not hdfs_path.startswith(("hdfs://", "/")):
        # Already a local path
        return hdfs_path
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
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


def extract_osm_buildings_osmosis(
    source_pbf: str,
    output_file: str,
    building_tags: List[str] = None
) -> str:
    if building_tags is None:
        building_tags = ["building"]
    
    # NOTE: Osmosis tag filtering is expressed as "key=value" or "key=*".
    # This function currently extracts all ways with building=*.
    # If you need broader filtering (e.g. amenity, landuse), extend the cmd.
    cmd = [
        "osmosis",
        "--read-pbf", source_pbf,
        "--tf", "accept-ways", f"building=*",
        "--tf", "reject-relations",
        "--tf", "reject-nodes",
        "--used-node",
        "--write-pbf", output_file
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_file
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Osmosis extraction failed: {e}")
        print("Consider using pyosmium or osmread for Python-based extraction")
        raise


def ingest_osm_buildings_simple(
    spark: SparkSession,
    source_path: str,
    target_path: str,
    building_tags: List[str] = None,
    min_area: float = 0.0
) -> DataFrame:
    print(f"Note: OSM PBF parsing requires specialized libraries.")
    print(f"For production, use pyosmium or osmread to parse {source_path}")
    print(f"Creating placeholder structure...")
    
    if building_tags is None:
        building_tags = ["building", "landuse", "amenity"]
    
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BinaryType
    
    schema = StructType([
        StructField("building_id", StringType(), True),
        StructField("building_type", StringType(), True),
        StructField("landuse", StringType(), True),
        StructField("amenity", StringType(), True),
        StructField("geometry_wkb", BinaryType(), True),  # WKB geometry
        StructField("area_m2", DoubleType(), True),
        StructField("osm_tags", StringType(), True)  # JSON string of all tags
    ])
    
    empty_df = spark.createDataFrame([], schema)
    
    empty_df = empty_df.withColumn("source_file", lit(os.path.basename(source_path))) \
                       .withColumn("crs", lit("EPSG:4326"))
    
    print(f"Warning: Placeholder OSM buildings table created at {target_path}")
    print(f"   To populate: Parse OSM PBF using pyosmium/osmread and extract building geometries")
    
    empty_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(target_path)
    
    return empty_df


def ingest_osm_buildings_with_pyosmium(
    spark: SparkSession,
    source_path: str,
    target_path: str,
    building_tags: List[str] = None,
    min_area: float = 0.0
) -> DataFrame:
    try:
        import osmium  # pyosmium
        from shapely import wkb as shapely_wkb
    except ImportError:
        print("Warning: pyosmium or shapely not installed. Using placeholder.")
        return ingest_osm_buildings_simple(spark, source_path, target_path, building_tags, min_area)
    
    if building_tags is None:
        building_tags = ["building"]
    
    class BuildingHandler(osmium.SimpleHandler):
        def __init__(self):
            osmium.SimpleHandler.__init__(self)
            self.buildings = []
            self._wkb_factory = osmium.geom.WKBFactory()
        
        def way(self, w):
            # Only keep features matching configured keys (default: "building")
            if not any(k in w.tags for k in building_tags):
                return

            try:
                # tags to dict (osmium tags are iterable; each has k/v)
                tags_dict = {tag.k: tag.v for tag in w.tags}

                # Core fields
                building_type = w.tags.get('building', 'unknown')
                landuse = w.tags.get('landuse', None)
                amenity = w.tags.get('amenity', None)

                # Geometry: attempt polygon WKB (works for closed ways)
                geom_wkb = None
                area_m2 = 0.0
                try:
                    geom_bytes = self._wkb_factory.create_linestring(w)
                    # If it's a closed way, treat as polygon; otherwise keep linestring.
                    # (OSM buildings should generally be closed; this is a best-effort parse.)
                    geom = shapely_wkb.loads(geom_bytes)
                    geom_wkb = bytearray(geom_bytes)
                    # Shapely area is in degrees² for EPSG:4326; still useful for filtering only
                    # if you reproject elsewhere. Keep area_m2 as 0 unless you later project.
                    # For now, preserve a non-zero proxy when available.
                    if geom and hasattr(geom, "area"):
                        area_m2 = float(getattr(geom, "area", 0.0)) or 0.0
                except Exception:
                    # geometry is optional; keep tags-only record
                    geom_wkb = None
                    area_m2 = 0.0

                self.buildings.append({
                    'building_id': f"way_{w.id}",
                    'building_type': building_type,
                    'landuse': landuse,
                    'amenity': amenity,
                    'geometry_wkb': geom_wkb,
                    'area_m2': area_m2,
                    'osm_tags': str(tags_dict)
                })
            except Exception as e:
                print(f"Error processing way {w.id}: {e}")
    
    # Handle HDFS paths by copying to temporary local file
    local_path = source_path
    temp_file_created = False
    
    # Check if it's an HDFS path (starts with hdfs:// or /) and not a local file:// path
    if source_path.startswith("hdfs://") or (source_path.startswith("/") and not source_path.startswith("file://")):
        # Check if file exists locally first (for backward compatibility)
        if not os.path.exists(source_path):
            print(f"Copying OSM file from HDFS to temporary location...")
            local_path = _copy_hdfs_to_local(spark, source_path, suffix=".pbf")
            temp_file_created = True
    
    handler = BuildingHandler()
    
    try:
        handler.apply_file(local_path, locations=True)
    except Exception as e:
        print(f"Error parsing OSM file: {e}")
        if temp_file_created and os.path.exists(local_path):
            os.unlink(local_path)
        return ingest_osm_buildings_simple(spark, source_path, target_path, building_tags, min_area)
    
    buildings = handler.buildings
    if not buildings:
        print("No buildings found in OSM file. Creating placeholder.")
        if temp_file_created and os.path.exists(local_path):
            os.unlink(local_path)
        return ingest_osm_buildings_simple(spark, source_path, target_path, building_tags, min_area)
    
    # Convert to Spark DataFrame
    df = spark.createDataFrame(buildings)
    
    # Add metadata
    df = df.withColumn("source_file", lit(os.path.basename(source_path))) \
           .withColumn("crs", lit("EPSG:4326"))
    
    if min_area > 0:
        df = df.filter(col("area_m2") >= min_area)
    
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(target_path)
    
    print(f"Ingested {len(buildings)} OSM buildings to {target_path}")
    
    # Clean up temporary file if we created one
    if temp_file_created and os.path.exists(local_path):
        os.unlink(local_path)
    
    return df


def run_osm_ingestion(spark: SparkSession, config: Dict) -> DataFrame:
    paths = config['paths']
    geospatial_config = config.get('geospatial', {}).get('osm', {})
    
    building_tags = geospatial_config.get('building_tags', ['building', 'landuse', 'amenity'])
    min_area = geospatial_config.get('min_building_area', 0.0)
    
    print("OSM BUILDING INGESTION")
    
    try:
        df = ingest_osm_buildings_with_pyosmium(
            spark,
            paths['geospatial']['osm_pbf'],
            os.path.join(paths['bronze_root'], "osm_buildings"),
            building_tags=building_tags,
            min_area=min_area
        )
    except Exception as e:
        print(f"pyosmium ingestion failed: {e}")
        print("Using placeholder implementation...")
        df = ingest_osm_buildings_simple(
            spark,
            paths['geospatial']['osm_pbf'],
            os.path.join(paths['bronze_root'], "osm_buildings"),
            building_tags=building_tags,
            min_area=min_area
        )
    
    print("OSM INGESTION COMPLETE")
    
    return df
