from pyspark.sql import SparkSession
from typing import Dict, Optional
import os


def create_spark_session(
    app_name: str = "UrbanEnergyLakehouse",
    master: Optional[str] = None,
    config_overrides: Optional[Dict[str, str]] = None,
    enable_delta: bool = True,
    enable_sedona: bool = True
) -> SparkSession:
    builder = SparkSession.builder.appName(app_name)
    
    if master:
        builder = builder.master(master)
    
    # Spark configuration
    config = {
        "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
        "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        "spark.databricks.delta.retentionDurationCheck.enabled": "false",
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.sql.warehouse.dir": "/tmp/spark-warehouse",
        "spark.hadoop.fs.defaultFS": "file:///", 
    }
    

    packages = []
    if enable_delta:
        packages.append("io.delta:delta-spark_2.12:3.0.0")
    if enable_sedona:
        packages.extend([
            "org.apache.sedona:sedona-spark-shaded-3.0_2.12:1.5.1",
            "org.datasyslab:geotools-wrapper:1.4.0-28.2"
        ])
        config.update({
            "spark.sedona.spark.version": "3.0",
            "spark.kryo.registrator": "org.apache.sedona.viz.core.Serde.SedonaVizKryoRegistrator"
        })
    
    if packages:
        config["spark.jars.packages"] = ",".join(packages)
    

    if config_overrides:
        config.update(config_overrides)
    
    for key, value in config.items():
        builder = builder.config(key, value)
    
    spark = builder.getOrCreate()
    
    # register Sedona UDFs if enabled
    if enable_sedona:
        try:
            from sedona.register import SedonaRegistrator
            SedonaRegistrator.registerAll(spark)
        except ImportError:
            print("Warning: Apache Sedona not available. Geospatial features may not work.")
    
    return spark


def get_or_create_spark_session(
    config: Optional[Dict] = None,
    use_docker: bool = False
) -> SparkSession:
    if config is None:
        from src.utils.config import load_config
        config = load_config()
    
    spark_config = config.get('spark', {})
    app_name = spark_config.get('app_name', 'UrbanEnergyLakehouse')
    
    if use_docker:
        master = spark_config.get('master', 'spark://spark-master:7077')
    else:
        master = spark_config.get('local', 'local[*]')
    
    spark_config_dict = {}
    if 'config' in spark_config:
        spark_config_dict = {k: str(v) for k, v in spark_config['config'].items()}
    
    # disable Sedona by default to avoid dependency issues
    enable_sedona = config.get('geospatial', {}).get('enable_sedona', False)
    
    return create_spark_session(
        app_name=app_name,
        master=master,
        config_overrides=spark_config_dict,
        enable_sedona=enable_sedona
    )
