"""Configuration loading utilities"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        # Try relative to project root
        config_file = Path(__file__).parent.parent.parent / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve relative paths
    project_root = Path(__file__).parent.parent.parent
    if 'paths' in config:
        for key, value in config['paths'].items():
            if isinstance(value, str) and not Path(value).is_absolute():
                # Don't resolve if it's a table name or special path
                if not value.startswith(('bronze.', 'silver.', 'gold.')) and \
                   not value.startswith('data/'):
                    continue
                if value.startswith('data/'):
                    config['paths'][key] = str(project_root / value)
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, str) and not Path(subvalue).is_absolute():
                        if subvalue.startswith('data/'):
                            config['paths'][key][subkey] = str(project_root / subvalue)
    
    return config


def get_spark_config(config: Dict[str, Any]) -> Dict[str, str]:
    """Extract Spark configuration from main config"""
    spark_config = config.get('spark', {}).get('config', {})
    return {str(k): str(v) for k, v in spark_config.items()}


def get_table_path(config: Dict[str, Any], layer: str, table_key: str) -> str:
    """Get full table path for a given layer and table key"""
    root_paths = {
        'bronze': config['paths']['bronze_root'],
        'silver': config['paths']['silver_root'],
        'gold': config['paths']['gold_root']
    }
    
    table_name = config['tables'][layer][table_key]
    # If table name includes layer prefix (e.g., "bronze.table"), use it
    if '.' in table_name:
        parts = table_name.split('.')
        return os.path.join(root_paths[parts[0]], parts[1])
    else:
        return os.path.join(root_paths[layer], table_name)
