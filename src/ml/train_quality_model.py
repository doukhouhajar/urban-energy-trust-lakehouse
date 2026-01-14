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
    ml_config = config.get('ml', {}).get('quality_risk', {})
    lgb_params = ml_config.get('lightgbm_params', {}).copy()  # Copy to avoid modifying original
    
    # Remove any target leakage features - be very aggressive!
    # Component scores (completeness_score, temporal_score, business_score, schema_score) 
    # WITHOUT _7d or _30d suffix are current-day scores = target leakage!
    # Only allow historical rolling averages (with _7d, _30d suffixes)
    leakage_keywords = [
        'quality_score', 'future_quality', 'target'
    ]
    safe_feature_cols = [f for f in feature_cols if not any(kw in f.lower() for kw in leakage_keywords)]
    
    # Remove current-day component scores (without rolling window suffix)
    # But KEEP rolling averages (with _7d, _30d suffixes)
    current_day_scores = ['completeness_score', 'temporal_score', 'business_score', 'schema_score', 'missing_rate']
    safe_feature_cols = [
        f for f in safe_feature_cols 
        if not (f.lower() in [s.lower() for s in current_day_scores] and '_7d' not in f.lower() and '_30d' not in f.lower())
    ]
    
    print(f"   After leakage removal: {len(safe_feature_cols)} features")
    if len(safe_feature_cols) < len(feature_cols):
        removed = set(feature_cols) - set(safe_feature_cols)
        print(f"   Removed {len(removed)} features with target leakage: {list(removed)[:10]}")
    
    if len(safe_feature_cols) < len(feature_cols):
        removed = set(feature_cols) - set(safe_feature_cols)
        print(f"   Removed {len(removed)} features with target leakage: {list(removed)[:5]}")
        feature_cols = safe_feature_cols
    
    # Add strong regularization to prevent overfitting
    lgb_params.setdefault('lambda_l1', 1.0)  # Increased L1 regularization
    lgb_params.setdefault('lambda_l2', 1.0)  # Increased L2 regularization
    lgb_params.setdefault('min_data_in_leaf', 50)  # Increased minimum samples in leaf
    lgb_params.setdefault('max_depth', 3)  # Reduced tree depth (was 5)
    lgb_params.setdefault('num_leaves', 15)  # Reduced leaves (was 31)
    lgb_params.setdefault('feature_fraction', 0.6)  # Use only 60% of features per tree (was 0.8)
    lgb_params.setdefault('bagging_fraction', 0.7)  # Use 70% of data per tree (was 0.8)
    lgb_params.setdefault('bagging_freq', 5)  # Bagging frequency
    lgb_params.setdefault('min_gain_to_split', 0.1)  # Minimum gain to split
    
    # Handle class imbalance with scale_pos_weight
    # Calculate the ratio of negative to positive samples
    negative_count = (train_df[target_col] == 0).sum()
    positive_count = (train_df[target_col] == 1).sum()
    if positive_count > 0:
        scale_pos_weight = negative_count / positive_count
        lgb_params['scale_pos_weight'] = scale_pos_weight
        print(f"   Class imbalance: {negative_count} negative, {positive_count} positive")
        print(f"   Setting scale_pos_weight: {scale_pos_weight:.2f}")
    else:
        print("   Warning: No positive samples in training set!")
    
    # Use validation set for early stopping to detect overfitting
    # Split training data into train and validation BEFORE creating Dataset
    from sklearn.model_selection import train_test_split
    X_train_full = train_df[safe_feature_cols]
    y_train_full = train_df[target_col]
    
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, 
        stratify=y_train_full if y_train_full.nunique() > 1 else None
    )
    
    train_data = lgb.Dataset(X_train_split, label=y_train_split)
    val_data = lgb.Dataset(X_val_split, label=y_val_split, reference=train_data)
    
    # Use fewer rounds and early stopping with validation set
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=100,  # More rounds but with early stopping
        valid_sets=[train_data, val_data],
        valid_names=['train', 'eval'],
        callbacks=[
            lgb.log_evaluation(period=10),
            lgb.early_stopping(stopping_rounds=20)  # More patience for early stopping
        ]
    )
    
    return model, safe_feature_cols


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, feature_cols: list) -> Dict:
    y_pred_proba = model.predict(X_test[feature_cols])
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Handle AUC calculation when there's only one class
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
        if np.isnan(auc):
            print("   Warning: AUC is NaN (only one class in test set). Setting to 0.5.")
            auc = 0.5
    except ValueError as e:
        print(f"   Warning: Could not calculate AUC: {e}. Setting to 0.5.")
        auc = 0.5
    
    f1 = f1_score(y_test, y_pred)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    feature_importance = dict(zip(feature_cols, model.feature_importance(importance_type='gain')))
    
    return {
        'auc': float(auc),
        'f1': float(f1),
        'classification_report': report,
        'feature_importance': feature_importance
    }


def train_and_evaluate(spark: SparkSession, config: Dict):
    print("QUALITY RISK MODEL TRAINING")
    
    print("\n1. Preparing ML features...")
    ml_dataset = prepare_ml_features(spark, config)
    
    # Check if we have any data
    row_count = ml_dataset.count()
    print(f"   ML dataset row count: {row_count}")
    
    if row_count == 0:
        raise ValueError(
            "ML dataset is empty. This could be because:\n"
            "1. No quality scores/incidents were generated\n"
            "2. No future dates available for target labels\n"
            "3. Join between features and labels resulted in no matches\n"
            "Please check your data quality pipeline output."
        )
    
    print("\n2. Converting to Pandas DataFrame...")
    ml_pandas = ml_dataset.toPandas()
    
    print(f"   Pandas DataFrame shape: {ml_pandas.shape}")
    
    if 'target' not in ml_pandas.columns:
        raise ValueError("Target column 'target' not found in ML dataset")
    
    target_counts = ml_pandas['target'].value_counts()
    print(f"   Target distribution:\n{target_counts}")
    print(f"   Target percentage:\n{target_counts / len(ml_pandas) * 100}")
    
    # Check for class imbalance - need at least 2 classes for binary classification
    unique_classes = ml_pandas['target'].nunique()
    if unique_classes < 2:
        print(f"\n   ERROR: Only {unique_classes} class(es) in target variable!")
        print(f"   All labels are: {ml_pandas['target'].iloc[0]}")
        print(f"   The auto-adjust threshold attempted to balance but all labels are still the same.")
        print(f"   This likely means all quality scores are very similar.")
        print(f"\n   Attempting to use a more aggressive threshold adjustment...")
        
        # Try to create labels using a different strategy: use incident count as primary signal
        # If we have incidents, use those to create positive labels
        print(f"   Checking if we can use incident-based labeling...")
        raise ValueError(
            f"Cannot train binary classifier: only {unique_classes} class in target. "
            f"All labels are {ml_pandas['target'].iloc[0]}. "
            f"Please check your quality scores data - they may all be above/below the threshold. "
            f"Consider using a regression model to predict quality_score directly instead of binary classification."
        )
    
    exclude_cols = ['household_id', 'score_date', 'target']
    feature_cols = [col for col in ml_pandas.columns if col not in exclude_cols]
    
    print(f"   Feature columns: {len(feature_cols)}")
    
    categorical_cols = ['acorn_group', 'tariff_type']
    for cat_col in categorical_cols:
        if cat_col in ml_pandas.columns:
            ml_pandas = pd.get_dummies(ml_pandas, columns=[cat_col], prefix=cat_col)
            feature_cols = [col for col in ml_pandas.columns if col not in exclude_cols]
    
    ml_pandas = ml_pandas.fillna(0)
    
    ml_config = config.get('ml', {}).get('quality_risk', {})
    test_size = 1 - ml_config.get('train_test_split', 0.8)
    random_state = ml_config.get('random_state', 42)
    
    # Check if we have enough samples for stratification
    min_class_count = target_counts.min()
    if min_class_count < 2:
        print(f"   Warning: One class has only {min_class_count} samples. Stratification disabled.")
        stratify = None
    else:
        stratify = ml_pandas['target']
    
    train_df, test_df = train_test_split(
        ml_pandas,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    print(f"   Training samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")
    print(f"   Initial features: {len(feature_cols)}")
    
    print("\n3. Training LightGBM model...")
    model, final_feature_cols = train_lightgbm_model(train_df, 'target', feature_cols, config)
    print(f"   Final features (after leakage removal): {len(final_feature_cols)}")
    
    print("\n4. Evaluating model...")
    metrics = evaluate_model(model, test_df, test_df['target'], final_feature_cols)
    
    print(f"\nModel Performance:")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    
    print(f"\nTop 10 Feature Importance:")
    sorted_importance = sorted(metrics['feature_importance'].items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_importance[:10]:
        print(f"  {feature}: {importance:.2f}")
    
    model_path = config.get('ml', {}).get('quality_risk', {}).get('model_path', 'models/quality_risk_model')
    # Create directory for model storage
    # Handle absolute paths that might be HDFS paths
    if model_path.startswith("file://"):
        local_path = model_path[len("file://"):]
        try:
            os.makedirs(local_path, exist_ok=True)
        except OSError as e:
            if e.errno == 30:  # Read-only file system
                # Try relative path instead
                model_path = "models/quality_risk_model"
                os.makedirs(model_path, exist_ok=True)
                print(f"   Warning: Could not create {local_path}, using {model_path} instead")
            else:
                raise
    elif model_path.startswith("hdfs://"):
        # HDFS paths are managed by Spark/Hadoop, no need to create
        pass
    elif model_path.startswith("/"):
        # Absolute local path - check if it's actually HDFS
        try:
            os.makedirs(model_path, exist_ok=True)
        except OSError as e:
            if e.errno == 30:  # Read-only file system - probably HDFS path
                # Use relative path instead
                model_path = "models/quality_risk_model"
                os.makedirs(model_path, exist_ok=True)
                print(f"   Warning: Could not create absolute path (read-only), using {model_path} instead")
            else:
                raise
    else:
        # Relative path - assume local
        os.makedirs(model_path, exist_ok=True)
    
    model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = os.path.join(model_path, f"model_{model_version}.pkl")
    metrics_file = os.path.join(model_path, f"metrics_{model_version}.json")
    
    joblib.dump(model, model_file)
    print(f"\n5. Saved model to {model_file}")
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   Saved metrics to {metrics_file}")
    
    feature_info = {
        'feature_columns': final_feature_cols,  # Use cleaned features (leakage removed)
        'model_version': model_version,
        'training_samples': len(train_df),
        'test_samples': len(test_df)
    }
    feature_info_file = os.path.join(model_path, f"feature_info_{model_version}.json")
    with open(feature_info_file, 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print("MODEL TRAINING COMPLETE")
    
    return model, metrics


def main():
    config = load_config()
    spark = get_or_create_spark_session(config, use_docker=False)
    
    try:
        model, metrics = train_and_evaluate(spark, config)
        print("\nModel training completed successfully!")
    except Exception as e:
        print(f"\nModel training failed: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
