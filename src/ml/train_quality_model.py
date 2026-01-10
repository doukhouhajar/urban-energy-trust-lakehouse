"""Train ML model for quality risk prediction"""

from pyspark.sql import SparkSession
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report, roc_curve
import joblib
import os
from typing import Dict
import json
from datetime import datetime

from src.utils.config import load_config
from src.utils.spark_session import get_or_create_spark_session
from src.features.quality_features import prepare_ml_features


def train_lightgbm_model(
    train_df: pd.DataFrame,
    target_col: str,
    feature_cols: list,
    config: Dict
) -> tuple:
    """Train LightGBM model"""
    ml_config = config.get('ml', {}).get('quality_risk', {})
    lgb_params = ml_config.get('lightgbm_params', {})
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    # Train LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data],
        callbacks=[lgb.log_evaluation(period=10)]
    )
    
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, feature_cols: list) -> Dict:
    """Evaluate model performance"""
    y_pred_proba = model.predict(X_test[feature_cols])
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Compute metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Feature importance
    feature_importance = dict(zip(feature_cols, model.feature_importance(importance_type='gain')))
    
    return {
        'auc': float(auc),
        'f1': float(f1),
        'classification_report': report,
        'feature_importance': feature_importance
    }


def train_and_evaluate(spark: SparkSession, config: Dict):
    """Train and evaluate quality risk prediction model"""
    print("=" * 60)
    print("QUALITY RISK MODEL TRAINING")
    print("=" * 60)
    
    # Prepare features
    print("\n1. Preparing ML features...")
    ml_dataset = prepare_ml_features(spark, config)
    
    # Convert to Pandas for LightGBM (for large datasets, use Spark MLlib)
    print("\n2. Converting to Pandas DataFrame...")
    ml_pandas = ml_dataset.toPandas()
    
    # Select feature columns (exclude target and metadata)
    exclude_cols = ['household_id', 'score_date', 'target']
    feature_cols = [col for col in ml_pandas.columns if col not in exclude_cols]
    
    # Handle categorical variables (one-hot encoding or label encoding)
    categorical_cols = ['acorn_group', 'tariff_type']
    for cat_col in categorical_cols:
        if cat_col in ml_pandas.columns:
            ml_pandas = pd.get_dummies(ml_pandas, columns=[cat_col], prefix=cat_col)
            # Update feature columns
            feature_cols = [col for col in ml_pandas.columns if col not in exclude_cols]
    
    # Fill remaining nulls
    ml_pandas = ml_pandas.fillna(0)
    
    # Split train/test
    ml_config = config.get('ml', {}).get('quality_risk', {})
    test_size = 1 - ml_config.get('train_test_split', 0.8)
    random_state = ml_config.get('random_state', 42)
    
    train_df, test_df = train_test_split(
        ml_pandas,
        test_size=test_size,
        random_state=random_state,
        stratify=ml_pandas['target']
    )
    
    print(f"   Training samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")
    print(f"   Features: {len(feature_cols)}")
    
    # Train model
    print("\n3. Training LightGBM model...")
    model = train_lightgbm_model(train_df, 'target', feature_cols, config)
    
    # Evaluate
    print("\n4. Evaluating model...")
    metrics = evaluate_model(model, test_df, test_df['target'], feature_cols)
    
    print(f"\nModel Performance:")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    
    print(f"\nTop 10 Feature Importance:")
    sorted_importance = sorted(metrics['feature_importance'].items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_importance[:10]:
        print(f"  {feature}: {importance:.2f}")
    
    # Save model
    model_path = config.get('ml', {}).get('quality_risk', {}).get('model_path', 'models/quality_risk_model')
    os.makedirs(model_path, exist_ok=True)
    
    model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = os.path.join(model_path, f"model_{model_version}.pkl")
    metrics_file = os.path.join(model_path, f"metrics_{model_version}.json")
    
    joblib.dump(model, model_file)
    print(f"\n5. Saved model to {model_file}")
    
    # Save metrics
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   Saved metrics to {metrics_file}")
    
    # Save feature columns list
    feature_info = {
        'feature_columns': feature_cols,
        'model_version': model_version,
        'training_samples': len(train_df),
        'test_samples': len(test_df)
    }
    feature_info_file = os.path.join(model_path, f"feature_info_{model_version}.json")
    with open(feature_info_file, 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE")
    print("=" * 60)
    
    return model, metrics


def main():
    """Main entry point"""
    config = load_config()
    spark = get_or_create_spark_session(config, use_docker=False)
    
    try:
        model, metrics = train_and_evaluate(spark, config)
        print("\n✓ Model training completed successfully!")
    except Exception as e:
        print(f"\n✗ Model training failed: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
