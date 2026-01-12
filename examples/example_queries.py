from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, sum as spark_sum, desc, current_date, expr
from src.utils.config import load_config
from src.utils.spark_session import get_or_create_spark_session


def example_top_areas_by_quality_score(spark: SparkSession, config: dict):
    paths = config['paths']
    
    quality_scores = spark.read.format("delta").load(
        f"{paths['gold_root']}/quality_scores"
    )
    
    print("TOP AREAS BY LOW QUALITY SCORE")
    
    # Join with household info to get ACORN groups
    household_info = spark.read.format("delta").load(
        f"{paths['bronze_root']}/household_info"
    )
    
    result = quality_scores.join(
        household_info.select("household_id", "acorn_group"),
        on="household_id",
        how="left"
    ).filter(
        col("score_date") >= expr("current_date() - INTERVAL 7 DAYS")
    ).groupBy(
        col("acorn_group"),
        expr("DATE_TRUNC('day', score_date)").alias("day")
    ).agg(
        avg("quality_score").alias("avg_score"),
        count("*").alias("household_count")
    ).orderBy(
        "avg_score"
    ).limit(20)
    
    result.show(truncate=False)
    return result


def example_incidents_by_type_over_time(spark: SparkSession, config: dict):
    paths = config['paths']
    
    incidents = spark.read.format("delta").load(
        f"{paths['gold_root']}/quality_incidents"
    )
    
    print("INCIDENTS BY TYPE OVER TIME")
    
    result = incidents.filter(
        col("incident_timestamp") >= expr("current_date() - INTERVAL 30 DAYS")
    ).groupBy(
        expr("DATE_TRUNC('day', incident_timestamp)").alias("day"),
        col("rule_name"),
        col("severity")
    ).agg(
        count("*").alias("incident_count"),
        count("entity_id").alias("affected_households")
    ).orderBy(
        desc("day"),
        desc("incident_count")
    )
    
    result.show(truncate=False)
    return result


def example_quality_risk_predictions(spark: SparkSession, config: dict):
    paths = config['paths']
    
    predictions = spark.read.format("delta").load(
        f"{paths['gold_root']}/quality_risk_predictions"
    )
    
    print("QUALITY RISK PREDICTIONS (Next Day)")
    
    result = predictions.filter(
        col("prediction_date") == expr("current_date() + INTERVAL 1 DAY")
    ).orderBy(
        desc("risk_score")
    ).limit(100)
    
    result.show(truncate=False)
    return result


def example_building_aggregation_by_admin_area(spark: SparkSession, config: dict):
    paths = config['paths']
    
    # this is a placeholder query
    buildings = spark.read.format("delta").load(
        f"{paths['gold_root']}/osm_buildings"
    )
    
    admin_areas = spark.read.format("delta").load(
        f"{paths['gold_root']}/gadm_level3"
    )
    
    print("BUILDING AGGREGATION BY ADMIN AREA")
    print("Note: This requires spatial joins with Sedona in production")
    print("Placeholder query structure:")
    
    # use spatial join
    result = admin_areas.select(
        col("district").alias("admin_area"),
        col("county"),
        lit(0).alias("building_count"),  # Would be count(*) from spatial join
        lit(0.0).alias("total_area_m2")   # Would be sum(ST_Area(...))
    ).orderBy(
        desc("building_count")
    ).limit(50)
    
    result.show(truncate=False)
    return result


def example_time_travel_query(spark: SparkSession, config: dict):
    paths = config['paths']
    
    print("TIME TRAVEL EXAMPLE - View Quality Scores at Previous Version")
    
    # Get current version
    current_df = spark.read.format("delta").load(
        f"{paths['gold_root']}/quality_scores"
    )
    
    # Get history
    history = spark.sql(
        f"DESCRIBE HISTORY delta.`{paths['gold_root']}/quality_scores`"
    )
    
    print("Version History:")
    history.show(truncate=False)
    
    # Read at version 0 (first version)
    try:
        version_0_df = spark.read.format("delta").option("versionAsOf", 0).load(
            f"{paths['gold_root']}/quality_scores"
        )
        
        print(f"\nCurrent version count: {current_df.count()}")
        print(f"Version 0 count: {version_0_df.count()}")
    except Exception as e:
        print(f"Note: Could not read version 0: {e}")
    
    return history


def run_all_examples():
    config = load_config()
    spark = get_or_create_spark_session(config, use_docker=False)
    
    try:
        print("URBAN ENERGY TRUST LAKEHOUSE - EXAMPLE QUERIES")

        print()
        
        example_top_areas_by_quality_score(spark, config)
        print()
        
        example_incidents_by_type_over_time(spark, config)
        print()
        
        example_quality_risk_predictions(spark, config)
        print()
        
        example_building_aggregation_by_admin_area(spark, config)
        print()
        
        example_time_travel_query(spark, config)
        print()
        

        print("ALL EXAMPLE QUERIES COMPLETED")

        
    except Exception as e:
        print(f"Error running examples: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    run_all_examples()
