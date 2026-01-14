import os
import tempfile

import pytest


def test_osm_ingestion_with_pyosmium_extracts_some_features():
    pytest.importorskip("osmium", reason="pyosmium (osmium) not installed")
    # Local import so test suite can still run without Spark-heavy imports unless needed.
    from pyspark.sql import SparkSession
    from src.geospatial.osm_ingestion import ingest_osm_buildings_with_pyosmium

    spark = (
        SparkSession.builder
        .appName("test-osm")
        .master("local[2]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )

    osm_xml = """<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6" generator="test">
  <node id="1" lat="51.5000" lon="-0.1000" />
  <node id="2" lat="51.5000" lon="-0.1005" />
  <node id="3" lat="51.5005" lon="-0.1005" />
  <node id="4" lat="51.5005" lon="-0.1000" />
  <way id="10">
    <nd ref="1"/>
    <nd ref="2"/>
    <nd ref="3"/>
    <nd ref="4"/>
    <nd ref="1"/>
    <tag k="building" v="yes"/>
  </way>
</osm>
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = os.path.join(tmpdir, "small.osm")
        target_path = os.path.join(tmpdir, "delta_out")

        with open(source_path, "w", encoding="utf-8") as f:
            f.write(osm_xml)

        df = ingest_osm_buildings_with_pyosmium(
            spark,
            source_path=source_path,
            target_path=target_path,
            building_tags=["building"],
            min_area=0.0,
        )

        assert df.count() >= 1
        assert "building_id" in df.columns
        assert "osm_tags" in df.columns

