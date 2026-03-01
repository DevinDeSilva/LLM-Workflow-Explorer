import time
import re
import yaml
import logging
import dycomutils as common_utils

logger = logging.getLogger(__name__)

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