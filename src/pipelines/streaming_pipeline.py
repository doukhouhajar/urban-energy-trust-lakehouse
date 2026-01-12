from pyspark.sql import SparkSession
from pyspark.sql.streaming import StreamingQuery
import os
from typing import Dict
import time

from src.utils.config import load_config
from src.utils.spark_session import get_or_create_spark_session
from src.ingestion.streaming_ingestion import stream_halfhourly_consumption


def run_streaming_pipeline(spark: SparkSession, config: Dict) -> None:
    paths = config['paths']
    streaming_config = config.get('streaming', {})
    
    print("URBAN ENERGY TRUST LAKEHOUSE - STREAMING PIPELINE")
    print("Starting streaming ingestion...")
    print(f"Source: {paths.get('streaming_source', 'data/streaming_source')}")
    print(f"Checkpoint: {paths.get('streaming_checkpoint', 'data/streaming_checkpoint')}")
    print()
    
    streaming_source = paths.get('streaming_source', 'data/streaming_source')
    os.makedirs(streaming_source, exist_ok=True)
    
    checkpoint_location = paths.get('streaming_checkpoint', 'data/streaming_checkpoint')
    os.makedirs(checkpoint_location, exist_ok=True)
    
    bronze_target = os.path.join(paths['bronze_root'], "halfhourly_consumption")
    
    print("Starting streaming query...")
    query = stream_halfhourly_consumption(
        spark,
        streaming_source,
        bronze_target,
        checkpoint_location,
        trigger_interval=streaming_config.get('trigger_interval', '30 seconds')
    )
    
    print("Streaming query started")
    print(f"  Query ID: {query.id}")
    print(f"  Status: {query.status}")
    print()
    print("Streaming pipeline is running...")
    print("Drop new CSV files into the source directory to process them.")
    print("Press Ctrl+C to stop.")
    print()
    
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\nStopping streaming query...")
        query.stop()
        print("Streaming stopped")
    
    print("STREAMING PIPELINE STOPPED")


def main():
    config = load_config()
    spark = get_or_create_spark_session(config, use_docker=False)
    
    try:
        run_streaming_pipeline(spark, config)
    except Exception as e:
        print(f"\nStreaming pipeline failed: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
