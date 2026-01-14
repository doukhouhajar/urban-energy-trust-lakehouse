#!/bin/bash

# Script to copy all raw data from local filesystem to HDFS
# Run this once before running the pipeline

set -e

echo "Copying raw data to HDFS"
echo ""

# Base HDFS path
HDFS_BASE="/urban-energy-trust-lakehouse/data/raw"

# Local data directory
LOCAL_DATA_DIR="data/raw"

# Check if HDFS is accessible
echo "Checking HDFS connection..."
if ! hdfs dfs -test -d /; then
    echo "ERROR: Cannot connect to HDFS. Make sure HDFS is running:"
    echo "  start-dfs.sh"
    exit 1
fi
echo "✓ HDFS is accessible"
echo ""

# Create base directory structure in HDFS
echo "Creating HDFS directory structure..."
hdfs dfs -mkdir -p "$HDFS_BASE/kaggle_smart_meters_london"
hdfs dfs -mkdir -p "$HDFS_BASE/kaggle_smart_meters_london/halfhourly_dataset/halfhourly_dataset"
hdfs dfs -mkdir -p "$HDFS_BASE/kaggle_smart_meters_london/daily_dataset/daily_dataset"
hdfs dfs -mkdir -p "$HDFS_BASE/gadm"
hdfs dfs -mkdir -p "$HDFS_BASE/osm"
echo "✓ Directory structure created"
echo ""

# Copy smart meter CSV files
echo "Copying smart meter data files..."
hdfs dfs -put "$LOCAL_DATA_DIR/kaggle_smart_meters_london/informations_households.csv" \
    "$HDFS_BASE/kaggle_smart_meters_london/" 2>/dev/null || echo "  (already exists)"
hdfs dfs -put "$LOCAL_DATA_DIR/kaggle_smart_meters_london/acorn_details.csv" \
    "$HDFS_BASE/kaggle_smart_meters_london/" 2>/dev/null || echo "  (already exists)"
hdfs dfs -put "$LOCAL_DATA_DIR/kaggle_smart_meters_london/weather_hourly_darksky.csv" \
    "$HDFS_BASE/kaggle_smart_meters_london/" 2>/dev/null || echo "  (already exists)"
hdfs dfs -put "$LOCAL_DATA_DIR/kaggle_smart_meters_london/weather_daily_darksky.csv" \
    "$HDFS_BASE/kaggle_smart_meters_london/" 2>/dev/null || echo "  (already exists)"
hdfs dfs -put "$LOCAL_DATA_DIR/kaggle_smart_meters_london/uk_bank_holidays.csv" \
    "$HDFS_BASE/kaggle_smart_meters_london/" 2>/dev/null || echo "  (already exists)"
echo "✓ Smart meter CSV files copied"
echo ""

# Copy half-hourly dataset blocks
echo "Copying half-hourly dataset blocks (this may take a while)..."
HALFHOURLY_SOURCE="$LOCAL_DATA_DIR/kaggle_smart_meters_london/halfhourly_dataset/halfhourly_dataset"
HALFHOURLY_TARGET="$HDFS_BASE/kaggle_smart_meters_london/halfhourly_dataset/halfhourly_dataset"

if [ -d "$HALFHOURLY_SOURCE" ]; then
    BLOCK_COUNT=$(ls -1 "$HALFHOURLY_SOURCE"/block_*.csv 2>/dev/null | wc -l)
    echo "  Found $BLOCK_COUNT block files"
    hdfs dfs -put "$HALFHOURLY_SOURCE"/block_*.csv "$HALFHOURLY_TARGET/" 2>/dev/null || echo "  (some files may already exist)"
    echo "✓ Half-hourly blocks copied"
else
    echo "  WARNING: Half-hourly dataset directory not found: $HALFHOURLY_SOURCE"
fi
echo ""

# Copy daily dataset blocks
echo "Copying daily dataset blocks (this may take a while)..."
DAILY_SOURCE="$LOCAL_DATA_DIR/kaggle_smart_meters_london/daily_dataset/daily_dataset"
DAILY_TARGET="$HDFS_BASE/kaggle_smart_meters_london/daily_dataset/daily_dataset"

if [ -d "$DAILY_SOURCE" ]; then
    BLOCK_COUNT=$(ls -1 "$DAILY_SOURCE"/block_*.csv 2>/dev/null | wc -l)
    echo "  Found $BLOCK_COUNT block files"
    hdfs dfs -put "$DAILY_SOURCE"/block_*.csv "$DAILY_TARGET/" 2>/dev/null || echo "  (some files may already exist)"
    echo "✓ Daily blocks copied"
else
    echo "  WARNING: Daily dataset directory not found: $DAILY_SOURCE"
fi
echo ""

# Copy geospatial data
echo "Copying geospatial data..."
if [ -f "$LOCAL_DATA_DIR/gadm/gadm41_GBR.gpkg" ]; then
    hdfs dfs -put "$LOCAL_DATA_DIR/gadm/gadm41_GBR.gpkg" \
        "$HDFS_BASE/gadm/" 2>/dev/null || echo "  (already exists)"
    echo "✓ GADM data copied"
else
    echo "  WARNING: GADM file not found: $LOCAL_DATA_DIR/gadm/gadm41_GBR.gpkg"
fi

# Prefer a smaller city extract to speed up OSM ingestion
if [ -f "$LOCAL_DATA_DIR/osm/london.osm.pbf" ]; then
    echo "  Copying OSM London PBF (smaller, faster ingestion)..."
    hdfs dfs -put "$LOCAL_DATA_DIR/osm/london.osm.pbf" \
        "$HDFS_BASE/osm/" 2>/dev/null || echo "  (already exists)"
    echo "✓ OSM London data copied"
elif [ -f "$LOCAL_DATA_DIR/osm/united-kingdom-260108.osm.pbf" ]; then
    echo "  Copying full UK OSM PBF (large; consider creating london.osm.pbf)..."
    hdfs dfs -put "$LOCAL_DATA_DIR/osm/united-kingdom-260108.osm.pbf" \
        "$HDFS_BASE/osm/" 2>/dev/null || echo "  (already exists)"
    echo "✓ OSM UK data copied"
else
    echo "  WARNING: OSM file not found: $LOCAL_DATA_DIR/osm/london.osm.pbf (or full UK .pbf)"
fi
echo ""

# Verify data was copied
echo "Verification"
echo "Listing HDFS raw data directory:"
hdfs dfs -ls -R "$HDFS_BASE" | head -20
echo ""
echo "Data copy complete!"
echo ""
echo "You can now run the pipeline with:"
echo "  python -m src.pipelines.batch_pipeline"
echo ""
