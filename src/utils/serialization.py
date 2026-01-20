"""
Serialization utilities for dataclasses and complex objects.
Consolidates dataclass_json_dump functions from multiple files.
"""

import json
import numpy as np
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any, Union


def dataclass_to_dict(obj: Any) -> Any:
    """
    Convert a dataclass (potentially with nested structures) to a dict.
    Handles numpy arrays, paths, and other common types.
    
    Args:
        obj: Object to convert (typically a dataclass)
        
    Returns:
        JSON-serializable dict
    """
    # ----- Dataclass -----
    if is_dataclass(obj):
        return dataclass_to_dict(asdict(obj))
    
    # ----- Numpy array -----
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # ----- Numpy scalar (np.float32, np.int64, etc.) -----
    if isinstance(obj, (np.generic,)):
        return obj.item()
    
    # ----- Path -----
    if isinstance(obj, Path):
        return str(obj)
    
    # ----- Dict -----
    if isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    
    # ----- List / Tuple -----
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(item) for item in obj]
    
    # ----- JSON-valid types -----
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    
    # ----- Other objects -----
    # Convert to string to avoid crashes
    return str(obj)


def save_to_json(obj: Any, filepath: Union[str, Path], indent: int = 4):
    """
    Save an object to JSON file.
    Handles dataclasses, numpy arrays, and nested structures.
    
    Args:
        obj: Object to save
        filepath: Path to JSON file
        indent: JSON indentation (default: 4)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(dataclass_to_dict(obj), f, indent=indent)


def load_from_json(filepath: Union[str, Path]) -> Any:
    """
    Load object from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded object (dict)
    """
    with open(filepath, 'r') as f:
        return json.load(f)


# Backward compatibility alias
dataclass_json_dump = save_to_json
