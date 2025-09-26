# src/utils.py

# Script with utility functions for saving/loading JSON and YAML files.
# Ensures directories exist when saving files, making it easier to manage configurations and metadata.

import json
from pathlib import Path
import yaml

def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)
