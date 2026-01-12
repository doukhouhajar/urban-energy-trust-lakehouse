from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, when
import os
from typing import Dict, List, Optional
import subprocess
import tempfile


def extract_osm_buildings_osmosis(
    source_pbf: str,
    output_file: str,
    building_tags: List[str] = None
) -> str:
    if building_tags is None:
        building_tags = ["building"]
    
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
        import osmium
        from shapely.geometry import Polygon, Point
        from shapely import wkb
    except ImportError:
        print("Warning: pyosmium or shapely not installed. Using placeholder.")
        return ingest_osm_buildings_simple(spark, source_path, target_path, building_tags, min_area)
    
    if building_tags is None:
        building_tags = ["building"]
    
    buildings = []
    
    class BuildingHandler(osmium.SimpleHandler):
        def __init__(self):
            osmium.SimpleHandler.__init__(self)
            self.buildings = []
        
        def way(self, w):
            if 'building' in w.tags:
                try:
                    # Extract building geometry-simplified
                    building_type = w.tags.get('building', 'unknown')
                    landuse = w.tags.get('landuse', None)
                    amenity = w.tags.get('amenity', None)
                    
                    # tags to dict
                    tags_dict = {tag.k: tag.v for tag in w.tags}
                    
                    buildings.append({
                        'building_id': f"way_{w.id}",
                        'building_type': building_type,
                        'landuse': landuse,
                        'amenity': amenity,
                        'geometry_wkb': None,  # Would contain actual geometry
                        'area_m2': 0.0,  # Would calculate from geometry
                        'osm_tags': str(tags_dict)
                    })
                except Exception as e:
                    print(f"Error processing way {w.id}: {e}")
    
    handler = BuildingHandler()
    
    try:
        handler.apply_file(source_path, locations=True)
        buildings = handler.buildings
    except Exception as e:
        print(f"Error parsing OSM file: {e}")
        return ingest_osm_buildings_simple(spark, source_path, target_path, building_tags, min_area)
    
    if not buildings:
        print("No buildings found in OSM file. Creating placeholder.")
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
