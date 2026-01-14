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
from src.features.quality_features import prepare_ml_features, _join_path


def load_latest_model(model_path: str) -> tuple:
    # Try multiple possible paths (absolute, relative, etc.)
    possible_paths = [
        model_path,  # Original path from config
        model_path.lstrip("/"),  # Remove leading slash if absolute
        os.path.join(os.getcwd(), model_path.lstrip("/")),  # Relative to current directory
        "models/quality_risk_model"  # Fallback relative path
    ]
    
    actual_path = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            actual_path = path
            break
    
    if actual_path is None:
        raise FileNotFoundError(
            f"Model path does not exist. Tried: {possible_paths}\n"
            f"Please check where the model was saved or update config.yaml model_path."
        )
    
    print(f"   Using model path: {actual_path}")
    
    model_files = [f for f in os.listdir(actual_path) if f.startswith("model_") and f.endswith(".pkl")]
    if not model_files:
        raise FileNotFoundError(f"No model files found in {actual_path}")
    
    latest_model_file = sorted(model_files)[-1]
    model_file_path = os.path.join(actual_path, latest_model_file)
    
    model = joblib.load(model_file_path)
    
    model_version = latest_model_file.replace("model_", "").replace(".pkl", "")
    feature_info_file = os.path.join(actual_path, f"feature_info_{model_version}.json")
    
    if os.path.exists(feature_info_file):
        with open(feature_info_file, 'r') as f:
            feature_info = json.load(f)
        feature_cols = feature_info.get('feature_columns', [])
    else:
        feature_cols = model.feature_name()
    
    return model, feature_cols, model_version


def predict_quality_risk(
    spark: SparkSession,
    config: Dict,
    prediction_date: Optional[str] = None
) -> DataFrame:
    
    model_path = config.get('ml', {}).get('quality_risk', {}).get('model_path', 'models/quality_risk_model')
    print(f"\n1. Loading model from {model_path}...")
    
    try:
        model, feature_cols, model_version = load_latest_model(model_path)
        print(f"   Loaded model version: {model_version}")
        print(f"   Features: {len(feature_cols)}")
    except Exception as e:
        print(f"   Error loading model: {e}")
        raise
    
    print("\n2. Preparing features...")
    features_df = prepare_ml_features(spark, config)
    
    if prediction_date:
        features_df = features_df.filter(col("score_date") == prediction_date)
    else:
        max_date = features_df.agg({"score_date": "max"}).collect()[0][0]
        features_df = features_df.filter(col("score_date") == max_date)
        print(f"   Predicting for date: {max_date}")
    
    print("\n3. Converting to Pandas for inference...")
    features_pandas = features_df.toPandas()
    
    if len(features_pandas) == 0:
        print("   Warning: No features found for prediction date")
        return spark.createDataFrame([], schema="""
            household_id STRING,
            prediction_date DATE,
            risk_score DOUBLE,
            risk_category STRING,
            model_version STRING
        """)
    
    categorical_cols = ['acorn_group', 'tariff_type']
    for cat_col in categorical_cols:
        if cat_col in features_pandas.columns:
            features_pandas = pd.get_dummies(features_pandas, columns=[cat_col], prefix=cat_col)
    
    available_features = [col for col in feature_cols if col in features_pandas.columns]
    missing_features = [col for col in feature_cols if col not in features_pandas.columns]
    
    if missing_features:
        print(f"   Warning: Missing features: {missing_features}")
        for feat in missing_features:
            features_pandas[feat] = 0.0
    
    features_pandas = features_pandas.reindex(columns=feature_cols, fill_value=0.0)
    
    print("\n4. Making predictions...")
    X_pred = features_pandas[feature_cols].fillna(0.0)
    risk_scores = model.predict(X_pred)
    
    feature_importance = model.feature_importance(importance_type='gain')
    feature_importance_dict = dict(zip(feature_cols, feature_importance))
    
    predictions_df = pd.DataFrame({
        'household_id': features_pandas['household_id'].values if 'household_id' in features_pandas.columns else range(len(risk_scores)),
        'prediction_date': features_pandas['score_date'].values if 'score_date' in features_pandas.columns else [prediction_date or datetime.now().date()] * len(risk_scores),
        'risk_score': risk_scores,
        'risk_category': pd.cut(risk_scores, bins=[0, 0.3, 0.7, 1.0], labels=['low', 'medium', 'high']),
        'model_version': model_version,
        'top_features': [json.dumps(dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]))] * len(risk_scores)
    })
    
    predictions_spark = spark.createDataFrame(predictions_df)
    
    print(f"   Generated {len(predictions_df)} predictions")
    print(f"   High risk (>= 0.7): {(predictions_df['risk_score'] >= 0.7).sum()}")
    print(f"   Medium risk (0.3-0.7): {((predictions_df['risk_score'] >= 0.3) & (predictions_df['risk_score'] < 0.7)).sum()}")
    print(f"   Low risk (< 0.3): {(predictions_df['risk_score'] < 0.3).sum()}")
    
    print("\n5. Writing predictions to Gold...")
    paths = config['paths']
    predictions_path = _join_path(paths['gold_root'], "quality_risk_predictions")
    
    predictions_spark.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(predictions_path)
    
    print(f"   Predictions saved to {predictions_path}")
    
    return predictions_spark


def main():
    config = load_config()
    spark = get_or_create_spark_session(config, use_docker=False)
    
    try:
        predictions_df = predict_quality_risk(spark, config)
        print("\nQuality risk prediction completed successfully!")
    except Exception as e:
        print(f"\nPrediction failed: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
