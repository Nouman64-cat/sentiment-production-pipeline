"""
Config module - YAML configuration files for model hyperparameters.

Files:
    - ml_config.yaml: Classical ML model hyperparameters
    - dl_config.yaml: Deep Learning model hyperparameters

Usage:
    import yaml
    import os
    
    config_path = os.path.join("src", "config", "ml_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
"""

import os

# Config directory path for easy access
CONFIG_DIR = os.path.dirname(__file__)
ML_CONFIG_PATH = os.path.join(CONFIG_DIR, "ml_config.yaml")
DL_CONFIG_PATH = os.path.join(CONFIG_DIR, "dl_config.yaml")
