import time
import re
import yaml
import logging
import hashlib
from typing import Optional, List, Any

import dycomutils as common_utils
import pandas as pd

logger = logging.getLogger(__name__)

def format_dataframe_to_string(
    dataframe,  # The pandas DataFrame to convert into a string.
    header_map: Optional[dict] = None,  # Optional mapping from original column names to display names.
    include_headers: bool = True,  # Whether to include the header row in the output.
    column_separator: str = " | ",  # Text placed between values in the same row.
    row_separator: str = "\n",  # Text placed between rows.
    na_rep: str = "",  # Replacement text for missing values such as NaN or None.
) -> str:
    """
    Formats a pandas DataFrame into a readable string with optional header remapping.
    """
    if dataframe is None:
        return ""

    header_map = header_map or {}
    columns = list(dataframe.columns)
    lines: List[str] = []

    if include_headers and columns:
        lines.append(
            column_separator.join(str(header_map.get(column, column)) for column in columns)
        )

    if dataframe.empty:
        return row_separator.join(lines)

    append_line = lines.append
    is_missing = pd.isna

    for row in dataframe.itertuples(index=False, name=None):
        append_line(
            column_separator.join(
                na_rep if is_missing(value) else str(value)
                for value in row
            )
        )

    return row_separator.join(lines)

def clean_string_list(items: List[Any]) -> List[str]:
        cleaned_items: List[str] = []
        for item in items:
            cleaned_item = " ".join(str(item).split()).strip().strip("\"'")
            if cleaned_item and cleaned_item not in cleaned_items:
                cleaned_items.append(cleaned_item)

        return cleaned_items

def create_timestamp_id(prefix:str):
    """
    Creates a unique identifier based on the current timestamp.
    R: create_timestamp_id
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{prefix}_{timestamp}"

def generate_hashed_filename(key: str, extension: str = ".txt") -> str:
    """
    Generates a unique, file-system-safe filename from a string of any length.
    
    Args:
        key (str): The input text (e.g., a long passage) to hash.
        extension (str): The file extension to append (default is '.txt').
        
    Returns:
        str: A 64-character hexadecimal filename with the extension.
    """
    # 1. Strings must be encoded to bytes before hashing
    key_bytes: bytes = key.encode("utf-8")
    
    # 2. Create the SHA-256 hash object
    hash_object = hashlib.sha256(key_bytes)
    
    # 3. Get the readable hexadecimal representation (64 chars long)
    file_hash: str = hash_object.hexdigest()
    
    # 4. Return the safe filename
    return f"{file_hash}{extension}"

def load_config(config_path: str) -> dict:
    """
    Loads a YAML configuration file and returns it as a dictionary.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config = common_utils.config.ConfigDict(config)
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}

def time_wrapper(func):
    """
    A decorator that prints the execution time of the function it decorates.
    """
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f"Function {func.__name__} Took {total_time:.4f} seconds")
        return result
    return wrapper

def regex_add_strings(template, **kwargs) -> str:
    """
    Adds strings to a template using regular expressions.
    R: regex_add_strings
    """
    try:
        for key, value in kwargs.items():
            pattern = r"\{" + re.escape(key) + r"\}"
            template = re.sub(pattern, str(value), template)
    
        return template
    except Exception as e:
        logger.error(f"Failed to add strings to template: {e}")
        return ""
