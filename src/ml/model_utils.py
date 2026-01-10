"""Model utilities and helpers"""

import os
import json
import joblib
from typing import Dict, Optional
from datetime import datetime


def save_model_metadata(
    model_path: str,
    model_version: str,
    metrics: Dict,
    feature_info: Dict
):
    """Save model metadata to JSON file"""
    metadata = {
        "model_version": model_version,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "feature_info": feature_info
    }
    
    metadata_file = os.path.join(model_path, f"metadata_{model_version}.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_file


def load_model_metadata(model_path: str, model_version: Optional[str] = None) -> Dict:
    """Load model metadata"""
    if model_version is None:
        # Find latest metadata file
        metadata_files = [f for f in os.listdir(model_path) if f.startswith("metadata_") and f.endswith(".json")]
        if not metadata_files:
            raise FileNotFoundError(f"No metadata files found in {model_path}")
        latest_file = sorted(metadata_files)[-1]
        metadata_file = os.path.join(model_path, latest_file)
    else:
        metadata_file = os.path.join(model_path, f"metadata_{model_version}.json")
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def get_model_versions(model_path: str) -> List[str]:
    """Get list of available model versions"""
    if not os.path.exists(model_path):
        return []
    
    model_files = [f for f in os.listdir(model_path) if f.startswith("model_") and f.endswith(".pkl")]
    versions = [f.replace("model_", "").replace(".pkl", "") for f in model_files]
    return sorted(versions)
