from pyspark.sql import SparkSession
import os
from datetime import datetime
from typing import Dict

from src.utils.config import load_config
from src.utils.spark_session import get_or_create_spark_session
from src.ingestion.batch_ingestion import run_bronze_ingestion
from src.geospatial.gadm_ingestion import run_gadm_ingestion
from src.geospatial.osm_ingestion import run_osm_ingestion
from src.transformations.silver_layer import run_silver_transformations
from src.transformations.gold_layer import run_gold_transformations
from src.quality.custom_checks import (
    check_completeness, check_temporal_coherence,
    check_business_rules, check_schema_validity
)
from src.quality.quality_scoring import (
    compute_quality_scores, write_quality_scores, write_quality_incidents
)
from src.governance.audit_log import (
    log_pipeline_run, get_table_row_count, get_table_version, get_quality_score_summary
)
from src.geospatial.spatial_operations import create_building_aggregations


def run_batch_pipeline(spark: SparkSession, config: Dict) -> bool:
    paths = config['paths']
    pipeline_name = "batch_pipeline"
    status = "SUCCESS"
    error_message = None
    row_counts = {}
    input_versions = {}
    output_paths = {}
    quality_summary = {}
    
    try:
        print("URBAN ENERGY TRUST LAKEHOUSE - BATCH PIPELINE")
        print(f"Started at: {datetime.now()}")
        print()
        
        print("STEP 1: BRONZE LAYER INGESTION")
        bronze_results = run_bronze_ingestion(spark, config)
        
        for key, df in bronze_results.items():
            table_path = os.path.join(paths['bronze_root'], key)
            row_counts[table_path] = df.count()
            input_versions[table_path] = get_table_version(spark, table_path)
            output_paths[f"bronze_{key}"] = table_path
        
        print()
        
        print("STEP 2: GEOSPATIAL INGESTION")
        try:
            gadm_results = run_gadm_ingestion(spark, config)
            for key, df in gadm_results.items():
                table_path = os.path.join(paths['bronze_root'], key)
                row_counts[table_path] = df.count()
                output_paths[f"geospatial_{key}"] = table_path
        except Exception as e:
            print(f"Warning: GADM ingestion failed: {e}")
            print("Continuing without geospatial data...")
            gadm_results = {}
        
        try:
            osm_results = run_osm_ingestion(spark, config)
        except Exception as e:
            print(f"Warning: OSM ingestion failed: {e}")
            print("Continuing without OSM data...")
            osm_results = None
        
        print()
        
        print("STEP 3: SILVER LAYER TRANSFORMATIONS")
        silver_results = run_silver_transformations(spark, config)
        
        for key, df in silver_results.items():
            table_path = os.path.join(paths['silver_root'], key)
            row_counts[table_path] = df.count()
            output_paths[f"silver_{key}"] = table_path
        
        print()
        
        print("STEP 4: DATA QUALITY CHECKS")
        
        silver_consumption_path = os.path.join(paths['silver_root'], "household_enriched")
        silver_df = spark.read.format("delta").load(silver_consumption_path)
        
        print("Running completeness checks...")
        completeness_metrics, completeness_incidents = check_completeness(
            silver_df,
            partition_cols=["household_id"],
            timestamp_col="timestamp",
            expected_interval_minutes=config.get('quality', {}).get('temporal_coherence', {}).get('expected_interval_minutes', 30),
            missing_threshold=config.get('quality', {}).get('completeness', {}).get('daily_missing_threshold', 0.10)
        )
        
        print("Running temporal coherence checks...")
        temporal_metrics, temporal_incidents = check_temporal_coherence(
            silver_df,
            partition_cols=["household_id"],
            timestamp_col="timestamp",
            expected_interval_minutes=config.get('quality', {}).get('temporal_coherence', {}).get('expected_interval_minutes', 30)
        )
        
        print("Running business rules checks...")
        business_metrics, business_incidents = check_business_rules(
            silver_df,
            partition_cols=["household_id"],
            value_col="energy_kwh",
            timestamp_col="timestamp",
            min_value=config.get('quality', {}).get('business_rules', {}).get('min_consumption', 0.0),
            max_value=config.get('quality', {}).get('business_rules', {}).get('max_consumption', 50.0),
            z_score_threshold=config.get('quality', {}).get('business_rules', {}).get('z_score_threshold', 5.0)
        )
        
        print("Running schema validity checks...")
        schema_metrics, schema_incidents = check_schema_validity(
            silver_df,
            partition_cols=["household_id"],
            timestamp_col="timestamp",
            allowed_tariffs=config.get('quality', {}).get('schema', {}).get('allowed_tariffs', ["Std", "ToU"]),
            allowed_acorn_groups=config.get('quality', {}).get('schema', {}).get('allowed_acorn_groups', ["Affluent", "Comfortable", "Adversity", "ACORN-"]),
            energy_range=tuple(config.get('quality', {}).get('schema', {}).get('energy_range', [0.0, 100.0]))
        )

        all_incidents = completeness_incidents.unionByName(temporal_incidents, allowMissingColumns=True) \
            .unionByName(business_incidents, allowMissingColumns=True) \
            .unionByName(schema_incidents, allowMissingColumns=True)
        

        print("Computing quality scores...")
        quality_scores = compute_quality_scores(
            completeness_metrics,
            temporal_metrics,
            business_metrics,
            schema_metrics,
            config
        )
        
        quality_scores_path = os.path.join(paths['gold_root'], "quality_scores")
        quality_incidents_path = os.path.join(paths['gold_root'], "quality_incidents")
        
        os.makedirs(paths['gold_root'], exist_ok=True)
        
        write_quality_scores(quality_scores, quality_scores_path, mode="overwrite")
        write_quality_incidents(all_incidents, quality_incidents_path, mode="overwrite")
        
        output_paths["quality_scores"] = quality_scores_path
        output_paths["quality_incidents"] = quality_incidents_path
        
        quality_summary = get_quality_score_summary(spark, quality_scores_path)
        
        print(f"   Quality scores computed: {quality_scores.count()} records")
        print(f"   Quality incidents: {all_incidents.count()} incidents")
        
        print()
        
        print("STEP 5: GOLD LAYER TRANSFORMATIONS")
        gold_results = run_gold_transformations(spark, config)
        
        print("\n6. Creating building aggregations...")
        building_agg = create_building_aggregations(spark, config)
        
        for key, df in gold_results.items():
            table_path = os.path.join(paths['gold_root'], key)
            row_counts[table_path] = df.count()
            output_paths[f"gold_{key}"] = table_path
        
        print()
        
        print("BATCH PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Finished at: {datetime.now()}")
        
    except Exception as e:
        status = "FAILED"
        error_message = str(e)
        print("\n" + "=" * 80)
        print(f"PIPELINE FAILED: {error_message}")
        raise
    
    finally:
        print("\nWriting audit log...")
        audit_log_path = os.path.join(paths['gold_root'], "audit_log")
        log_pipeline_run(
            spark,
            audit_log_path,
            pipeline_name,
            config,
            input_versions=input_versions,
            output_paths=output_paths,
            row_counts=row_counts,
            quality_score_summary=quality_summary,
            status=status,
            error_message=error_message
        )
        print("   Audit log written")
    
    return status == "SUCCESS"


def main():
    config = load_config()
    spark = get_or_create_spark_session(config, use_docker=False)
    
    try:
        success = run_batch_pipeline(spark, config)
        if success:
            print("\nBatch pipeline completed successfully!")
        else:
            print("\nBatch pipeline completed with errors")
    except Exception as e:
        print(f"\nBatch pipeline failed: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
