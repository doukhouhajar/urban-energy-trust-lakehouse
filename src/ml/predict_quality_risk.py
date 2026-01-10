"""Quality risk prediction using trained ML model"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, current_date, date_add
import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
import os
import json
from typing import Dict, Optional
from datetime import datetime

from src.utils.config import load_config
from src.utils.spark_session import get_or_create_spark_session
from src.features.quality_features import prepare_ml_features


def load_latest_model(model_path: str) -> tuple:
    """Load latest trained model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    # Find latest model file
    model_files = [f for f in os.listdir(model_path) if f.startswith("model_") and f.endswith(".pkl")]
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_path}")
    
    latest_model_file = sorted(model_files)[-1]
    model_file_path = os.path.join(model_path, latest_model_file)
    
    # Load model
    model = joblib.load(model_file_path)
    
    # Load feature info
    model_version = latest_model_file.replace("model_", "").replace(".pkl", "")
    feature_info_file = os.path.join(model_path, f"feature_info_{model_version}.json")
    
    if os.path.exists(feature_info_file):
        with open(feature_info_file, 'r') as f:
            feature_info = json.load(f)
        feature_cols = feature_info.get('feature_columns', [])
    else:
        # Fallback: try to infer from model
        feature_cols = model.feature_name()
    
    return model, feature_cols, model_version


def predict_quality_risk(
    spark: SparkSession,
    config: Dict,
    prediction_date: Optional[str] = None
) -> DataFrame:
    """
    Predict quality risk for future time windows
    
    Args:
        spark: SparkSession
        config: Configuration dictionary
        prediction_date: Date to predict for (default: tomorrow)
    
    Returns:
        DataFrame with predictions
    """
    print("=" * 60)
    print("QUALITY RISK PREDICTION")
    print("=" * 60)
    
    # Load model
    model_path = config.get('ml', {}).get('quality_risk', {}).get('model_path', 'models/quality_risk_model')
    print(f"\n1. Loading model from {model_path}...")
    
    try:
        model, feature_cols, model_version = load_latest_model(model_path)
        print(f"   ✓ Loaded model version: {model_version}")
        print(f"   Features: {len(feature_cols)}")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        raise
    
    # Prepare features
    print("\n2. Preparing features...")
    features_df = prepare_ml_features(spark, config)
    
    # Filter to prediction date if specified
    if prediction_date:
        features_df = features_df.filter(col("score_date") == prediction_date)
    else:
        # Default: predict for most recent date
        max_date = features_df.agg({"score_date": "max"}).collect()[0][0]
        features_df = features_df.filter(col("score_date") == max_date)
        print(f"   Predicting for date: {max_date}")
    
    # Convert to Pandas for inference
    print("\n3. Converting to Pandas for inference...")
    features_pandas = features_df.toPandas()
    
    if len(features_pandas) == 0:
        print("   ⚠️  No features found for prediction date")
        return spark.createDataFrame([], schema="""
            household_id STRING,
            prediction_date DATE,
            risk_score DOUBLE,
            risk_category STRING,
            model_version STRING
        """)
    
    # Handle categorical variables (match training)
    categorical_cols = ['acorn_group', 'tariff_type']
    for cat_col in categorical_cols:
        if cat_col in features_pandas.columns:
            features_pandas = pd.get_dummies(features_pandas, columns=[cat_col], prefix=cat_col)
    
    # Ensure feature columns match
    available_features = [col for col in feature_cols if col in features_pandas.columns]
    missing_features = [col for col in feature_cols if col not in features_pandas.columns]
    
    if missing_features:
        print(f"   Warning: Missing features: {missing_features}")
        # Add missing features as zeros
        for feat in missing_features:
            features_pandas[feat] = 0.0
    
    # Select and order features
    features_pandas = features_pandas.reindex(columns=feature_cols, fill_value=0.0)
    
    # Make predictions
    print("\n4. Making predictions...")
    X_pred = features_pandas[feature_cols].fillna(0.0)
    risk_scores = model.predict(X_pred)
    
    # Get feature importance for explainability
    feature_importance = model.feature_importance(importance_type='gain')
    feature_importance_dict = dict(zip(feature_cols, feature_importance))
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'household_id': features_pandas['household_id'].values if 'household_id' in features_pandas.columns else range(len(risk_scores)),
        'prediction_date': features_pandas['score_date'].values if 'score_date' in features_pandas.columns else [prediction_date or datetime.now().date()] * len(risk_scores),
        'risk_score': risk_scores,
        'risk_category': pd.cut(risk_scores, bins=[0, 0.3, 0.7, 1.0], labels=['low', 'medium', 'high']),
        'model_version': model_version,
        'top_features': [json.dumps(dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]))] * len(risk_scores)
    })
    
    # Convert back to Spark DataFrame
    predictions_spark = spark.createDataFrame(predictions_df)
    
    print(f"   ✓ Generated {len(predictions_df)} predictions")
    print(f"   High risk (>= 0.7): {(predictions_df['risk_score'] >= 0.7).sum()}")
    print(f"   Medium risk (0.3-0.7): {((predictions_df['risk_score'] >= 0.3) & (predictions_df['risk_score'] < 0.7)).sum()}")
    print(f"   Low risk (< 0.3): {(predictions_df['risk_score'] < 0.3).sum()}")
    
    # Write predictions to Gold
    print("\n5. Writing predictions to Gold...")
    paths = config['paths']
    predictions_path = os.path.join(paths['gold_root'], "quality_risk_predictions")
    
    predictions_spark.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(predictions_path)
    
    print(f"   ✓ Predictions saved to {predictions_path}")
    
    print("\n" + "=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)
    
    return predictions_spark


def main():
    """Main entry point"""
    config = load_config()
    spark = get_or_create_spark_session(config, use_docker=False)
    
    try:
        predictions_df = predict_quality_risk(spark, config)
        print("\n✓ Quality risk prediction completed successfully!")
    except Exception as e:
        print(f"\n✗ Prediction failed: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
